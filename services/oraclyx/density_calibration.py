"""Calibration conditioned on how crowded the frame is, because one curve is wrong on both ends.

The existing calibration fits one isotonic curve per class over the whole corpus. That assumes a
detector's confidence means the same thing in an empty highway frame and at a crowded junction, and on
this corpus it does not: the median frame carries 27 detections above 0.25 and the sparsest carry 1. In a
crowded frame objects occlude each other, boxes overlap, and a 0.8 is far more often wrong than the same
0.8 on an empty road. Averaging the two produces a curve that is optimistic where it matters and
pessimistic where it does not, and the error cancels in the aggregate ECE so nothing reports it.

Conditioning on density fixes that, and the interesting part is what happens when there is not enough of
it. A per-(class, density) cell can easily hold a dozen detections, and an isotonic fit on a dozen points
is a step function through noise. So:

    at or above MIN_CELL_SUPPORT     fit the cell
    below it                         fall back to the per-class curve, and RECORD the fallback

The record is the point. A calibration table where some cells are conditioned and some are not, with no
way to tell which, is worse than either alone: a reader takes the whole thing as density-aware and it is
density-aware only where the data happened to be thick. Every served value carries which curve produced
it, and the report counts the cells that fell back.

WHAT DENSITY MEANS HERE. The number of detections the same run left on the frame above the operating
point, bucketed by the same fixed boundaries the blind audit stratifies on
(services/verdyx/blind_audit.py). Fixed rather than per-run quantiles, so two runs are comparable, and
shared with the audit so "dense" means one thing in this engine.

This is also the first thing in the tree to write `Prediction.conf_calibrated`, which has existed unwritten
since migration 0069.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EvalPatch, InferenceRun, Prediction

# Shared with the blind audit on purpose: "dense" must mean one thing across the engine, or a calibration
# cell and an audit stratum with the same name would describe different frames.
from services.verdyx.blind_audit import DENSITY_BOUNDS, density_stratum

log = get_logger("density_calibration")

# Below this a cell is an isotonic fit through a dozen points, which is a step function through noise.
MIN_CELL_SUPPORT = 100
MIN_CLASS_SUPPORT = 50
_SEED = 20260820


@dataclass(frozen=True)
class Curve:
    """One isotonic curve as knots, plus what it was fitted on and how much of it there was.

    Stored as knots and served with np.interp so nothing at label time needs sklearn, matching the
    existing gold_calibrate storage format.
    """

    kx: tuple[float, ...]
    ky: tuple[float, ...]
    n: int
    scope: str          # "class:<id>" or "class:<id>|density:<bucket>"
    fallback: bool      # True when this curve is the per-class one standing in for a thin cell

    def __call__(self, conf: float) -> float:
        if not self.kx:
            return float(conf)
        return float(np.clip(np.interp(conf, self.kx, self.ky), 0.0, 1.0))


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)
    return np.asarray(iso.X_thresholds_, float), np.asarray(iso.y_thresholds_, float)


def _ece(pred: np.ndarray, ys: np.ndarray, bins: int = 10) -> float:
    if pred.size == 0:
        return float("nan")
    idx = np.clip((pred * bins).astype(int), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += (m.sum() / pred.size) * abs(float(pred[m].mean()) - float(ys[m].mean()))
    return float(err)


class DensityCalibration:
    """Per (class, density) curves with a per-class fallback, and a record of which was used."""

    KIND = "density-isotonic-v1"

    def __init__(self, cells: dict[str, Curve], per_class: dict[int, Curve], *,
                 meta: dict[str, Any] | None = None):
        self._cells = cells
        self._per_class = per_class
        self.meta = meta or {}

    @staticmethod
    def key(class_id: int, bucket: str) -> str:
        return f"{class_id}|{bucket}"

    def curve_for(self, class_id: int, bucket: str | None) -> Curve | None:
        if bucket is not None:
            c = self._cells.get(self.key(class_id, bucket))
            if c is not None:
                return c
        return self._per_class.get(class_id)

    def calibrate(self, conf: float, class_id: int, bucket: str | None) -> tuple[float, str]:
        """(calibrated confidence, which curve produced it). The second value is not optional.

        A table where some cells are density-conditioned and some are not, with no way to tell which, is
        worse than either alone: a reader takes the whole thing as density-aware and it is only where the
        data happened to be thick.
        """
        c = self.curve_for(class_id, bucket)
        if c is None:
            return float(conf), "uncalibrated"
        return c(conf), c.scope

    def to_json(self) -> dict[str, Any]:
        def _c(c: Curve) -> dict[str, Any]:
            return {"kx": [round(v, 6) for v in c.kx], "ky": [round(v, 6) for v in c.ky],
                    "n": c.n, "scope": c.scope, "fallback": c.fallback}

        return {"kind": self.KIND,
                "cells": {k: _c(v) for k, v in sorted(self._cells.items())},
                "per_class": {str(k): _c(v) for k, v in sorted(self._per_class.items())},
                **self.meta}

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> DensityCalibration:
        if blob.get("kind") != cls.KIND:
            raise ValueError(f"not a {cls.KIND} blob: {blob.get('kind')}")

        def _c(d: dict[str, Any]) -> Curve:
            return Curve(tuple(d["kx"]), tuple(d["ky"]), int(d["n"]), d["scope"], bool(d["fallback"]))

        meta = {k: v for k, v in blob.items() if k not in ("kind", "cells", "per_class")}
        return cls({k: _c(v) for k, v in blob.get("cells", {}).items()},
                   {int(k): _c(v) for k, v in blob.get("per_class", {}).items()}, meta=meta)


async def _pairs(db: AsyncSession, run: InferenceRun, score_thr: float
                 ) -> tuple[list[tuple[int, str, float, bool, Any]], dict[Any, int]]:
    """(class_id, density bucket, conf, was-right, frame_id) per scored detection, plus per-frame counts."""
    counts = dict((await db.execute(
        select(Prediction.frame_id, func.count(Prediction.prediction_id))
        .where(Prediction.run_id == run.run_id, Prediction.conf >= score_thr)
        .group_by(Prediction.frame_id))).all())

    rows = (await db.execute(
        select(EvalPatch.outcome, EvalPatch.pred_class_id, EvalPatch.frame_id, Prediction.conf)
        .join(Prediction, Prediction.prediction_id == EvalPatch.prediction_id)
        .where(EvalPatch.run_id == run.run_id, EvalPatch.prediction_id.is_not(None),
               EvalPatch.pred_class_id.is_not(None)))).all()

    out = []
    for outcome, cid, fid, conf in rows:
        if conf is None:
            continue
        out.append((int(cid), density_stratum(int(counts.get(fid, 0)), "density"), float(conf),
                    outcome == "tp", fid))
    return out, {k: int(v) for k, v in counts.items()}


def _split(frames: list[Any], val_frac: float, seed: int) -> set[Any]:
    rng = np.random.default_rng(seed)
    fr = sorted(frames, key=str)
    rng.shuffle(fr)
    return set(fr[: max(1, int(len(fr) * val_frac))])


async def fit_density_calibration(db: AsyncSession, *, run_id: str, score_thr: float = 0.25,
                                  val_frac: float = 0.3, seed: int = _SEED,
                                  min_cell: int = MIN_CELL_SUPPORT) -> dict[str, Any]:
    """Fit per (class, density) curves with a per-class fallback. Fits, validates, reports. Does not write."""
    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"measured": False, "reason": "inference run not found", "run_id": run_id}
    if (run.params or {}).get("reconstructed"):
        return {"measured": False, "run_id": run_id,
                "reason": "this run is reconstructed and has no real confidence distribution"}

    pairs, _counts = await _pairs(db, run, score_thr)
    if not pairs:
        return {"measured": False, "run_id": run_id,
                "reason": "the run has no scored predictions to calibrate against"}

    val_frames = _split(sorted({p[4] for p in pairs}, key=str), val_frac, seed)
    train = [p for p in pairs if p[4] not in val_frames]
    val = [p for p in pairs if p[4] in val_frames]

    by_class: dict[int, list] = {}
    by_cell: dict[str, list] = {}
    for cid, bucket, conf, ok, _fid in train:
        by_class.setdefault(cid, []).append((conf, ok))
        by_cell.setdefault(DensityCalibration.key(cid, bucket), []).append((conf, ok))

    per_class: dict[int, Curve] = {}
    for cid, obs in by_class.items():
        if len(obs) < MIN_CLASS_SUPPORT:
            continue
        x = np.array([o[0] for o in obs])
        y = np.array([float(o[1]) for o in obs])
        kx, ky = _fit_isotonic(x, y)
        per_class[cid] = Curve(tuple(kx), tuple(ky), len(obs), f"class:{cid}", fallback=False)

    # The cell inventory is taken over ALL pairs, not just the training split. A cell whose every
    # observation landed in validation appears in neither `by_cell` nor the fitted set, so it used to be
    # counted in no total at all and the coverage note undercounted its own denominator - which is the
    # exact dishonesty this module says it is avoiding.
    all_cells: dict[str, int] = {}
    for cid, bucket, _conf, _ok, _fid in pairs:
        all_cells[DensityCalibration.key(cid, bucket)] = all_cells.get(
            DensityCalibration.key(cid, bucket), 0) + 1

    cells: dict[str, Curve] = {}
    thin: list[dict[str, Any]] = []
    for key in sorted(all_cells):
        obs = by_cell.get(key, [])
        cid = int(key.split("|")[0])
        bucket = key.split("|")[1]
        if len(obs) < min_cell:
            # Recorded, not silently dropped: this is the cell where the table is not density-aware, and
            # a reader has to be able to see that rather than infer it from an absence.
            thin.append({"class_id": cid, "bucket": bucket, "n": len(obs),
                         "n_all": all_cells[key], "min": min_cell,
                         "fell_back_to": "class" if cid in per_class else "uncalibrated",
                         # Distinguishes "too few to fit" from "the split put all of them in validation",
                         # which want different answers: more labels versus a different split.
                         "reason": ("none of this cell's observations landed in the training split"
                                    if not obs else "below the cell minimum")})
            continue
        x = np.array([o[0] for o in obs])
        y = np.array([float(o[1]) for o in obs])
        kx, ky = _fit_isotonic(x, y)
        cells[key] = Curve(tuple(kx), tuple(ky), len(obs), f"class:{cid}|density:{bucket}",
                           fallback=False)

    cal = DensityCalibration(cells, per_class, meta={
        "run_id": run_id, "model_version": run.model_version, "gold_id": run.gold_id,
        "score_thr": score_thr, "density_bounds": [list(b) for b in DENSITY_BOUNDS]})

    # Validation: raw against per-class-only against density-conditioned, on held-out frames.
    if val:
        vx = np.array([p[2] for p in val])
        vy = np.array([float(p[3]) for p in val])
        flat = np.array([(per_class[p[0]](p[2]) if p[0] in per_class else p[2]) for p in val])
        dens = np.array([cal.calibrate(p[2], p[0], p[1])[0] for p in val])
        raw_ece, flat_ece, dens_ece = _ece(vx, vy), _ece(flat, vy), _ece(dens, vy)
    else:
        raw_ece = flat_ece = dens_ece = float("nan")

    # Per bucket, and this is what the verdict is judged on rather than the aggregate.
    #
    # The aggregate ECE cannot see the defect being fixed, and the fixture proves it: a per-class curve
    # that predicts the pooled base rate everywhere scores 0.0007 while being badly wrong in both buckets,
    # because it is over-confident in the dense half by exactly as much as it is under-confident in the
    # sparse half and the binning averages the two away. That cancellation IS the failure. Comparing the
    # worst bucket instead cannot cancel: a curve is only better if it is better where it is worst.
    per_bucket = {}
    worst_flat = worst_dens = 0.0
    for name, _lo, _hi in DENSITY_BOUNDS:
        sel = [p for p in val if p[1] == name]
        if not sel:
            per_bucket[name] = {"n": 0, "raw_ece": None, "per_class_ece": None, "cal_ece": None}
            continue
        sx = np.array([p[2] for p in sel])
        sy = np.array([float(p[3]) for p in sel])
        sf = np.array([(per_class[p[0]](p[2]) if p[0] in per_class else p[2]) for p in sel])
        sd = np.array([cal.calibrate(p[2], p[0], p[1])[0] for p in sel])
        f_ece, d_ece = _ece(sf, sy), _ece(sd, sy)
        worst_flat = max(worst_flat, f_ece)
        worst_dens = max(worst_dens, d_ece)
        per_bucket[name] = {"n": len(sel), "raw_ece": round(_ece(sx, sy), 6),
                            "per_class_ece": round(f_ece, 6), "cal_ece": round(d_ece, 6)}

    improves = bool(worst_dens <= worst_flat + 1e-9)
    out = {
        "measured": True, "run_id": run_id, "model_version": run.model_version,
        "gold_id": run.gold_id,
        "n_pairs": len(pairs), "n_train": len(train), "n_val": len(val),
        "n_cells": len(cells), "n_per_class": len(per_class), "n_thin_cells": len(thin),
        "thin_cells": thin,
        "raw_val_ece": round(raw_ece, 6), "per_class_val_ece": round(flat_ece, 6),
        "density_val_ece": round(dens_ece, 6),
        # The two the verdict actually rests on. An aggregate that looks good while both buckets are wrong
        # is the thing this exists to catch, so it is reported but never gated on.
        "worst_bucket_per_class_ece": round(worst_flat, 6),
        "worst_bucket_density_ece": round(worst_dens, 6),
        "per_bucket": per_bucket,
        "beats_per_class": improves,
        "trustworthy": bool(improves and cells),
        "calibration": cal,
        # Said on the result: a partly-conditioned table read as fully conditioned is the failure mode.
        "coverage_note": (f"{len(cells)} of {len(all_cells)} (class, density) cells had at least "
                          f"{min_cell} training observations; the rest serve the per-class curve and are "
                          "marked as fallbacks in every served value"),
    }
    log.info("density_calibration.fit", run=run_id, cells=len(cells), thin=len(thin),
             raw_ece=out["raw_val_ece"], class_ece=out["per_class_val_ece"],
             density_ece=out["density_val_ece"], worst_class=out["worst_bucket_per_class_ece"],
             worst_density=out["worst_bucket_density_ece"], beats=improves)
    return out


async def write_calibrated_confidence(db: AsyncSession, *, run_id: str, calibration: DensityCalibration,
                                      score_thr: float = 0.25, batch: int = 5000) -> dict[str, Any]:
    """Write `Prediction.conf_calibrated` for a run. The first thing in the tree to fill that column.

    It has existed unwritten since migration 0069, which is why every threshold fitted so far reads raw
    confidence: services/oraclyx/threshold_fit.py prefers the calibrated column and has never found one.

    This is an UPDATE on the append-only prediction plane, and it is the one kind that plane permits: the
    invariant protects the model's raw output (`conf`, `class_id`, `bbox`), and `conf_calibrated` is a
    derived column that exists to be filled. Nothing here touches a raw field.
    """
    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"error": "inference run not found", "run_id": run_id}

    counts = dict((await db.execute(
        select(Prediction.frame_id, func.count(Prediction.prediction_id))
        .where(Prediction.run_id == run.run_id, Prediction.conf >= score_thr)
        .group_by(Prediction.frame_id))).all())

    rows = (await db.execute(
        select(Prediction.prediction_id, Prediction.frame_id, Prediction.class_id, Prediction.conf)
        .where(Prediction.run_id == run.run_id, Prediction.conf.is_not(None)))).all()

    written = 0
    by_scope: dict[str, int] = {}
    pending: list[dict[str, Any]] = []
    for pid, fid, cid, conf in rows:
        bucket = density_stratum(int(counts.get(fid, 0)), "density")
        value, scope = calibration.calibrate(float(conf), int(cid), bucket)
        by_scope[scope] = by_scope.get(scope, 0) + 1
        if scope == "uncalibrated":
            # No curve covers this class at all. Writing the raw score into the calibrated column would
            # make an uncalibrated prediction indistinguishable from a calibrated one, and every consumer
            # that prefers the calibrated column would silently read raw confidence believing otherwise.
            continue
        pending.append({"prediction_id": pid, "conf_calibrated": round(value, 6)})
        if len(pending) >= batch:
            await db.execute(update(Prediction), pending)
            written += len(pending)
            pending.clear()
    if pending:
        await db.execute(update(Prediction), pending)
        written += len(pending)
    await db.commit()

    log.info("density_calibration.written", run=run_id, written=written, by_scope=by_scope)
    return {"run_id": run_id, "n_written": written, "by_scope": by_scope,
            "n_uncalibrated": by_scope.get("uncalibrated", 0)}


__all__ = ["Curve", "DensityCalibration", "fit_density_calibration", "write_calibrated_confidence",
           "MIN_CELL_SUPPORT", "MIN_CLASS_SUPPORT"]
