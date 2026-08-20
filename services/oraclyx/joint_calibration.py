"""P(correct | confidence, tube coherence): calibrating on two axes instead of one.

Existing calibration maps confidence to a probability, per class. It cannot express what a track knows: a
0.55 detection that is the twentieth frame of a stable tube and a 0.55 detection that appears for one
frame in the middle of nothing get the same calibrated number, and they are not equally likely to be
right. Confidence cannot separate them because the detector never saw the other frames.

This fits a two-dimensional surface over (confidence, tube score) and enforces monotonicity in both, which
is the whole reason to constrain the fit rather than just bin it. Higher confidence should not make a
detection less likely to be right, and neither should better temporal coherence; a raw binned estimate
violates both constantly on small cells, and a violation is noise every time rather than a discovery.

WHY BINNED ISOTONIC AND NOT A 2D MONOTONE SPLINE. pygam and direct scipy are not dependencies here
(pyproject pins sklearn as the only fitting library), so a true bivariate monotone smoother is not
available. The documented fallback is a 10x10 binned grid followed by a pool-adjacent-violators pass along
each axis in turn, iterated until it stops moving. That converges to a surface monotone in both axes, and
it is honest about being a step function rather than pretending to a smoothness the data does not support.

FIT, REPORT, DO NOT ACTIVATE. Exactly the discipline in services/autolabel/gold_calibrate.py: split by
FRAME so no detection leaks between train and validation, fit on train, compare the validation ECE before
and after, and return `trustworthy` without switching anything on. A calibration that makes validation ECE
worse is reported as such and refused, because a calibration that hurts is worse than none: it launders a
raw score into something that looks measured.

The surface serves through np.interp-style lookup on the stored grid, so nothing at label time needs
sklearn.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EvalPatch, InferenceRun, Prediction

log = get_logger("joint_calibration")

N_BINS = 10
# Below this many labelled detections the grid is mostly empty cells and the monotone pass is smoothing
# noise into a shape. The fit refuses rather than producing a confident-looking surface.
MIN_SUPPORT = 200
# A cell with fewer than this many observations is filled from its neighbours rather than trusted. One
# detection in a cell gives a probability of exactly 0 or 1, and the monotone pass would propagate it.
MIN_CELL = 5
_SEED = 20260820


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the closest non-decreasing sequence in weighted least squares.

    Written out rather than taken from sklearn because it runs per row and per column of the grid, and
    IsotonicRegression's fit/predict overhead per call dominates at that size.
    """
    y = y.astype(np.float64).copy()
    w = w.astype(np.float64).copy()
    n = y.size
    if n < 2:
        return y
    # Stack of (value, weight, length) blocks, merged whenever the sequence goes backwards.
    vals, wts, lens = [], [], []
    for i in range(n):
        v, ww, ll = y[i], w[i], 1
        while vals and vals[-1] > v:
            pv, pw, pl = vals.pop(), wts.pop(), lens.pop()
            tot = pw + ww
            v = (pv * pw + v * ww) / tot if tot > 0 else (pv + v) / 2.0
            ww, ll = tot, pl + ll
        vals.append(v)
        wts.append(ww)
        lens.append(ll)
    out = np.empty(n, dtype=np.float64)
    i = 0
    for v, ll in zip(vals, lens, strict=True):
        out[i:i + ll] = v
        i += ll
    return out


def _monotone_2d(grid: np.ndarray, counts: np.ndarray, iters: int = 25) -> np.ndarray:
    """Make `grid` non-decreasing along both axes, alternating PAV passes until it settles.

    Alternating rather than solving jointly: the exact bivariate isotonic problem needs a solver that is
    not available here, and alternating projections converge to a point monotone in both axes, which is
    the property being asserted. It is iterated to a fixed point rather than run once, because one pass
    along rows can reintroduce a violation along columns.
    """
    g = grid.astype(np.float64).copy()
    w = np.maximum(counts.astype(np.float64), 1e-9)
    for _ in range(iters):
        before = g.copy()
        for i in range(g.shape[0]):
            g[i, :] = _pav(g[i, :], w[i, :])
        for j in range(g.shape[1]):
            g[:, j] = _pav(g[:, j], w[:, j])
        if np.max(np.abs(g - before)) < 1e-9:
            break
    return np.clip(g, 0.0, 1.0)


def _fill_sparse(grid: np.ndarray, counts: np.ndarray, min_cell: int) -> np.ndarray:
    """Replace thinly-observed cells with the overall rate before smoothing.

    A cell with one observation is exactly 0.0 or 1.0, and the monotone pass would spread that certainty
    across its neighbours. Filling with the marginal rate says "we do not know here" in the only way a
    grid can.
    """
    total = counts.sum()
    base = float((grid * counts).sum() / total) if total > 0 else 0.5
    return np.where(counts >= min_cell, grid, base)


def _ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: mean |predicted - observed| over equal-width probability bins."""
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        err += (m.sum() / probs.size) * abs(float(probs[m].mean()) - float(correct[m].mean()))
    return float(err)


class JointSurface:
    """A fitted (conf, tube) -> probability grid that serves without sklearn.

    Bilinear over the cell centres, so a detection between two cells gets a value between them rather
    than a step. The grid itself is a step function; interpolating it is not a claim about smoothness,
    it is what stops two nearly identical detections getting visibly different answers.
    """

    KIND = "joint-isotonic-grid-v1"

    def __init__(self, grid: list[list[float]], *, n_bins: int = N_BINS,
                 fallback: float = 0.5, meta: dict[str, Any] | None = None):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.n_bins = n_bins
        self.fallback = fallback
        self.meta = meta or {}

    def __call__(self, conf: float, tube: float | None) -> float:
        if tube is None:
            # No tube: fall back to the confidence marginal, which is the row-mean over tube bins. Using
            # the best tube bin would silently reward a detection for evidence it never provided.
            i = self._axis(conf)
            return float(np.clip(self.grid[i, :].mean(), 0.0, 1.0))
        return float(np.clip(self._bilinear(conf, tube), 0.0, 1.0))

    def _axis(self, v: float) -> int:
        return int(np.clip(int(np.clip(v, 0.0, 1.0) * self.n_bins), 0, self.n_bins - 1))

    def _bilinear(self, conf: float, tube: float) -> float:
        n = self.n_bins
        x = np.clip(conf, 0.0, 1.0) * n - 0.5
        y = np.clip(tube, 0.0, 1.0) * n - 0.5
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        x0, x1 = np.clip([x0, x0 + 1], 0, n - 1)
        y0, y1 = np.clip([y0, y0 + 1], 0, n - 1)
        g = self.grid
        return float(g[x0, y0] * (1 - fx) * (1 - fy) + g[x1, y0] * fx * (1 - fy)
                      + g[x0, y1] * (1 - fx) * fy + g[x1, y1] * fx * fy)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.KIND, "n_bins": self.n_bins, "fallback": self.fallback,
                "grid": [[round(float(v), 6) for v in row] for row in self.grid], **self.meta}

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> JointSurface:
        if blob.get("kind") != cls.KIND:
            raise ValueError(f"not a {cls.KIND} blob: {blob.get('kind')}")
        meta = {k: v for k, v in blob.items() if k not in ("kind", "n_bins", "fallback", "grid")}
        return cls(blob["grid"], n_bins=int(blob.get("n_bins", N_BINS)),
                   fallback=float(blob.get("fallback", 0.5)), meta=meta)


def fit_surface(conf: np.ndarray, tube: np.ndarray, correct: np.ndarray, *,
                frame_ids: np.ndarray | None = None, n_bins: int = N_BINS,
                min_support: int = MIN_SUPPORT, min_cell: int = MIN_CELL,
                seed: int = _SEED) -> dict[str, Any]:
    """Fit P(correct | conf, tube), validate it, and report whether it is worth using.

    The split is by FRAME when frame ids are given. Splitting by detection leaks: two boxes on one frame
    share the scene, the lighting and often the object, so a per-detection split trains and validates on
    the same evidence and reports a calibration far better than it is.
    """
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    tube = np.asarray(tube, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct).reshape(-1).astype(bool)
    if not (conf.size == tube.size == correct.size):
        raise ValueError(f"{conf.size} confidences, {tube.size} tube scores, {correct.size} outcomes")
    if conf.size < min_support:
        return {"measured": False, "trustworthy": False, "n": int(conf.size),
                "reason": f"{conf.size} labelled detections, below the {min_support} needed to fill a "
                          f"{n_bins}x{n_bins} grid without smoothing noise into a shape"}

    rng = np.random.default_rng(seed)
    if frame_ids is not None:
        fids = np.asarray(frame_ids).reshape(-1)
        uniq = np.unique(fids)
        rng.shuffle(uniq)
        train_frames = set(uniq[: max(1, int(0.7 * uniq.size))].tolist())
        train = np.array([f in train_frames for f in fids], dtype=bool)
    else:
        train = rng.random(conf.size) < 0.7
    if train.sum() < min_support // 2 or (~train).sum() < 20:
        return {"measured": False, "trustworthy": False, "n": int(conf.size),
                "reason": "the frame split leaves too little on one side to fit and validate separately"}

    def _grid(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ci = np.clip((conf[mask] * n_bins).astype(int), 0, n_bins - 1)
        ti = np.clip((tube[mask] * n_bins).astype(int), 0, n_bins - 1)
        counts = np.zeros((n_bins, n_bins))
        hits = np.zeros((n_bins, n_bins))
        np.add.at(counts, (ci, ti), 1.0)
        np.add.at(hits, (ci, ti), correct[mask].astype(float))
        raw = np.divide(hits, counts, out=np.full_like(hits, np.nan), where=counts > 0)
        return raw, counts

    raw, counts = _grid(train)
    base = float(correct[train].mean())
    filled = _fill_sparse(np.where(np.isnan(raw), base, raw), counts, min_cell)
    smooth = _monotone_2d(filled, counts)
    surface = JointSurface(smooth.tolist(), n_bins=n_bins, fallback=base)

    val = ~train
    raw_val = _ece(conf[val], correct[val])
    cal_val = _ece(np.array([surface(c, t) for c, t in zip(conf[val], tube[val], strict=True)]),
                   correct[val])
    # Degeneracy: a surface that is one number everywhere has "calibrated" by throwing the signal away.
    spread = float(smooth.max() - smooth.min())
    improved = cal_val < raw_val
    trustworthy = bool(improved and spread > 0.02)

    return {
        "measured": True, "trustworthy": trustworthy,
        "raw_val_ece": round(raw_val, 6), "cal_val_ece": round(cal_val, 6),
        "improvement": round(raw_val - cal_val, 6), "spread": round(spread, 6),
        "n": int(conf.size), "n_train": int(train.sum()), "n_val": int(val.sum()),
        "base_rate": round(base, 6), "n_bins": n_bins,
        "surface": surface,
        "reason": (None if trustworthy else
                   ("the calibrated validation ECE is no better than the raw one, so this would launder a "
                    "raw score into something that looks measured"
                    if not improved else
                    "the fitted surface is nearly constant, so it has calibrated by discarding the signal")),
    }


async def fit_from_run(db: AsyncSession, *, run_id: str, tube_by_track: dict[str, float] | None = None,
                       n_bins: int = N_BINS, min_support: int = MIN_SUPPORT) -> dict[str, Any]:
    """Fit the joint surface from a run's recorded outcomes and its tracker output.

    `tube_by_track` maps a Prediction.track_id to its tube score. Detections whose track has no score, and
    detections from a detector run with no track_id at all, are excluded rather than given a default:
    a made-up tube score would be fitted against as though it were evidence.
    """
    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"measured": False, "trustworthy": False, "reason": "inference run not found"}

    rows = (await db.execute(
        select(EvalPatch.outcome, EvalPatch.frame_id, Prediction.conf, Prediction.conf_calibrated,
               Prediction.track_id)
        .join(Prediction, Prediction.prediction_id == EvalPatch.prediction_id)
        .where(EvalPatch.run_id == run.run_id, EvalPatch.prediction_id.is_not(None)))).all()
    if not rows:
        return {"measured": False, "trustworthy": False,
                "reason": "the run has no scored predictions to calibrate against"}

    tb = tube_by_track or {}
    conf, tube, correct, fids = [], [], [], []
    n_no_track = n_no_tube = 0
    for outcome, fid, c, cal, tid in rows:
        score = cal if cal is not None else c
        if score is None:
            continue
        if tid is None:
            n_no_track += 1
            continue
        t = tb.get(str(tid))
        if t is None:
            n_no_tube += 1
            continue
        conf.append(float(score))
        tube.append(float(t))
        correct.append(outcome == "tp")
        fids.append(fid)

    if not conf:
        return {"measured": False, "trustworthy": False,
                "n_no_track": n_no_track, "n_no_tube": n_no_tube,
                "reason": ("no detection in this run has both a confidence and a tube score; a detection "
                           "run carries no track_id, so there is nothing temporal to calibrate on")}

    out = fit_surface(np.array(conf), np.array(tube), np.array(correct),
                      frame_ids=np.array(fids, dtype=object), n_bins=n_bins, min_support=min_support)
    out.update({"run_id": run_id, "model_version": run.model_version, "gold_id": run.gold_id,
                "n_no_track": n_no_track, "n_no_tube": n_no_tube,
                "fitted_at": datetime.now(UTC).isoformat()})
    log.info("joint_calibration.fit", run=run_id, measured=out["measured"],
             trustworthy=out.get("trustworthy"), n=out.get("n"),
             raw_ece=out.get("raw_val_ece"), cal_ece=out.get("cal_val_ece"),
             no_track=n_no_track, no_tube=n_no_tube)
    return out


__all__ = ["JointSurface", "fit_surface", "fit_from_run", "N_BINS", "MIN_SUPPORT", "MIN_CELL"]
