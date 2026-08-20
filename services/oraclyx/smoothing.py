"""Offline trajectory smoothing: the thing that makes pseudo-ground-truth better than the model that produced it.

ORACLYX exists to produce auto-truth good enough to distil from, and the whole argument for that is that it
runs offline. An online tracker sees only the past, so at frame t it is guessing; a batch pass sees the entire
clip, so frame t can be corrected by what happened afterwards. Until now the module did not use that: it
interpolated linearly between anchor boxes, which trusts every anchor exactly and cannot tell a real
displacement from a bad detection.

The difference shows on the case that matters. A detector that puts one box thirty pixels off for a single
frame produces, under interpolation, a trajectory that faithfully reproduces the error and drags its
neighbours toward it. Under a smoother the same frame is one noisy measurement disagreeing with a motion model
supported by fifty others, and it is pulled back.

Rauch-Tung-Striebel over a constant-velocity model. Forward Kalman pass, then a backward pass that revisits
every state with everything learned later. Two additions the plain textbook version does not have, both
because this runs on real detector output rather than a simulation:

**Gating.** A measurement whose innovation is wildly inconsistent with the prediction is down-weighted rather
than absorbed. Without it a single flyer box drags the whole trajectory, which is precisely the failure
smoothing was supposed to fix.

**Per-frame uncertainty.** The smoother knows how well-determined each box is, and that is worth as much as
the box: it tells the distillation path which frames to trust and gives `pseudo_label_uncertainty` a real
number instead of a constant.

Time is carried in nanoseconds and converted to seconds per step, so a variable-rate clip (this corpus's rig
frames are 28ms apart across cameras, and stride sampling leaves 333ms gaps) is handled correctly rather than
assumed uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# State is [cx, cy, w, h, vcx, vcy, vw, vh]: centre and size, each with a velocity. Boxes are smoothed in
# centre-size form rather than as corners because a corner parameterisation couples position and scale, so a
# box that grows while approaching produces correlated corner noise the filter would read as motion.
STATE_DIM = 8
MEAS_DIM = 4

# Chi-square 99.9% for 4 degrees of freedom. A measurement beyond this is not evidence about where the object
# is; it is a detector mistake, and treating it as evidence is what smoothing exists to prevent.
GATE_CHI2 = 18.467


@dataclass(frozen=True)
class SmoothingParams:
    """Defaults are a floor, not an assumption: the real value is estimated per track (see `_estimate_q`).

    A fixed process noise cannot be right across this corpus. Measured over real tracks, an object moves a
    median of 53px between frames and a p90 of 564px, because the footage is sampled at 3fps from a moving
    vehicle. A constant tuned for the median calls every fast object an outlier, and one tuned for the p90
    smooths nothing. The first version of this used 4.0 px/s^2 and gated 86% of all boxes, which is a
    smoother fighting the data rather than cleaning it.
    """

    process_pos: float = 4.0      # floor for centre acceleration, px/s^2; raised per track from observation
    process_size: float = 2.0     # floor for size change
    meas_pos: float = 3.0         # detector centre noise, px
    meas_size: float = 4.0        # detector size noise, px, larger because extents are the harder half
    gate_chi2: float = GATE_CHI2
    # Growth applied to a gated measurement's noise instead of dropping it. Down-weighting keeps the frame in
    # the trajectory with an honest uncertainty; dropping it would leave a hole the caller has to guess about.
    gate_inflate: float = 25.0
    # Whether to estimate the motion model from the track. Off only for tests that need a fixed model.
    adapt: bool = True
    # Bounds on the estimate. The floor stops a briefly-stationary object producing a model so tight that its
    # first real movement reads as an error; the ceiling stops a track with an identity switch (this corpus
    # has 2,266 flagged track inconsistencies) from inflating its noise until nothing is ever an outlier,
    # which would silently disable the gating.
    adapt_min: float = 4.0
    adapt_max: float = 4000.0


def to_centre_size(bbox) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, abs(x2 - x1), abs(y2 - y1)], dtype=float)


def to_corners(z: np.ndarray) -> list[float]:
    cx, cy, w, h = (float(v) for v in z[:4])
    w, h = max(w, 0.0), max(h, 0.0)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _transition(dt: float) -> np.ndarray:
    F = np.eye(STATE_DIM)
    for i in range(4):
        F[i, i + 4] = dt
    return F


def _process_noise(dt: float, p: SmoothingParams) -> np.ndarray:
    """Piecewise-white acceleration, so a long gap is genuinely more uncertain than a short one.

    A fixed Q would make a 333ms stride gap look as well determined as a 33ms one, which is exactly backwards
    for a corpus sampled at 3fps from 30fps footage.
    """
    Q = np.zeros((STATE_DIM, STATE_DIM))
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    for i in range(4):
        q = (p.process_pos if i < 2 else p.process_size) ** 2
        Q[i, i] = dt4 / 4.0 * q
        Q[i, i + 4] = dt3 / 2.0 * q
        Q[i + 4, i] = dt3 / 2.0 * q
        Q[i + 4, i + 4] = dt2 * q
    return Q


def _estimate_q(Z: np.ndarray, ts: np.ndarray, p: SmoothingParams) -> tuple[float, float]:
    """Acceleration scale for this track, from its own motion.

    Uses the median absolute second difference rather than the mean or the max, because a track with an
    identity switch contains a step of a thousand pixels and any non-robust statistic would let that one
    frame define the model for every other.
    """
    if len(Z) < 3:
        return p.process_pos, p.process_size
    dt = np.diff(ts) / 1e9
    dt = np.clip(dt, 1e-3, None)
    out = []
    for lo, hi in ((0, 2), (2, 4)):        # centre channels, then size channels
        v = np.diff(Z[:, lo:hi], axis=0) / dt[:, None]
        if len(v) < 2:
            out.append(p.process_pos if lo == 0 else p.process_size)
            continue
        a = np.diff(v, axis=0) / dt[1:, None]
        scale = float(np.median(np.abs(a))) if a.size else 0.0
        floor = p.process_pos if lo == 0 else p.process_size
        out.append(float(np.clip(scale, max(floor, p.adapt_min), p.adapt_max)))
    return out[0], out[1]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


def coherence(observations: list[tuple[int, list[float]]]) -> dict:
    """Whether this is one object over time, or several sharing an id.

    Measured on this corpus before trusting the smoother with it: 43% of consecutive box pairs had zero
    overlap and 50% changed class, with 57 of 60 sampled tracks containing more than one class. Those are not
    trajectories. No amount of smoothing recovers a path from a sequence that alternates between a motorcycle
    and a pole, and a smoother that ran anyway would emit a confident trajectory through the average of two
    different objects, which is exactly the plausible-wrong-answer this system keeps having to hunt down.

    So this is a precondition rather than a diagnostic: `smooth_track` refuses what fails it.
    """
    if len(observations) < 3:
        return {"coherent": True, "pairs": 0, "zero_overlap_frac": 0.0, "reversal_frac": 0.0,
                "reason": "too short to judge"}
    obs = sorted(observations, key=lambda o: int(o[0]))
    ious = [_iou(a[1], b[1]) for a, b in zip(obs, obs[1:], strict=False)]
    zero = sum(1 for v in ious if v <= 0.0)
    frac = zero / len(ious)

    # Overlap alone cannot tell a fast object from an identity switch, and at 3fps a genuinely fast object
    # clears its own box every frame. What separates them is direction: a real object keeps going roughly the
    # same way, while a track flipping between two objects reverses on almost every step. So the test is
    # whether the motion is *consistent*, not whether it is small.
    cs = np.array([to_centre_size(b)[:2] for _, b in obs])
    steps = np.diff(cs, axis=0)
    if len(steps) >= 2:
        dots = np.sum(steps[1:] * steps[:-1], axis=1)
        norms = np.linalg.norm(steps[1:], axis=1) * np.linalg.norm(steps[:-1], axis=1)
        cos = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 1e-9)
        reversal = float(np.mean(cos < -0.5))
    else:
        reversal = 0.0

    incoherent = frac > MAX_ZERO_OVERLAP_FRAC and reversal > MAX_REVERSAL_FRAC
    return {"coherent": not incoherent,
            "pairs": len(ious), "zero_overlap_frac": round(frac, 3),
            "reversal_frac": round(reversal, 3),
            "median_iou": round(float(np.median(ious)), 3),
            "reason": (None if not incoherent else
                       f"{zero} of {len(ious)} consecutive boxes do not overlap and "
                       f"{reversal:.0%} of steps reverse direction, so this is more than one object "
                       "sharing a track id rather than one moving quickly")}


# How much discontinuity a real trajectory may contain. Some is legitimate: a fast object at 3fps can leave
# the previous box entirely, and an occlusion produces a gap. The corpus sits at 43%.
MAX_ZERO_OVERLAP_FRAC = 0.25
# And how often it may reverse direction. A real object rarely does; a track alternating between two objects
# reverses on nearly every step. Both conditions must hold before a track is refused, so a fast object is
# smoothed and an oscillating one is not.
MAX_REVERSAL_FRAC = 0.4


def smooth_track(observations: list[tuple[int, list[float]]], *,
                 params: SmoothingParams | None = None,
                 require_coherent: bool = True) -> dict:
    """Smooth one track's boxes offline.

    `observations` is [(ts_ns, bbox_xyxy), ...] in any order; they are sorted here because a caller reading
    from the database has no reason to guarantee it and a filter fed out-of-order timestamps produces
    confident nonsense.

    Returns a box per observed timestamp, each with the smoothed corners, a per-frame positional standard
    deviation, and whether the original measurement was gated as an outlier.
    """
    p = params or SmoothingParams()
    obs = sorted(observations, key=lambda o: int(o[0]))
    n = len(obs)
    if n == 0:
        return {"boxes": [], "n": 0, "gated": 0, "smoothed": False,
                "reason": "no observations"}
    coh = coherence(obs)
    if require_coherent and not coh["coherent"]:
        # Refusing is the feature. Producing a smooth path through several different objects would be a
        # confident wrong answer, and pseudo-ground-truth is exactly where that is most expensive.
        return {"boxes": [{"ts_ns": int(t), "bbox": [float(v) for v in b], "std": None, "gated": False}
                          for t, b in obs],
                "n": n, "gated": 0, "smoothed": False, "coherence": coh,
                "reason": coh["reason"]}

    if n < 3:
        # Two points define a line, so there is nothing for a motion model to disagree with and smoothing
        # would return the input while implying it had been improved.
        return {"boxes": [{"ts_ns": int(t), "bbox": [float(v) for v in b], "std": None, "gated": False}
                          for t, b in obs],
                "n": n, "gated": 0, "smoothed": False,
                "reason": "fewer than three observations; nothing to smooth against"}

    Z = np.array([to_centre_size(b) for _, b in obs])
    ts = np.array([int(t) for t, _ in obs], dtype=np.int64)

    if p.adapt:
        qp, qs = _estimate_q(Z, ts, p)
        p = SmoothingParams(process_pos=qp, process_size=qs, meas_pos=p.meas_pos,
                            meas_size=p.meas_size, gate_chi2=p.gate_chi2,
                            gate_inflate=p.gate_inflate, adapt=False,
                            adapt_min=p.adapt_min, adapt_max=p.adapt_max)

    H = np.zeros((MEAS_DIM, STATE_DIM))
    H[:4, :4] = np.eye(4)
    R = np.diag([p.meas_pos ** 2, p.meas_pos ** 2, p.meas_size ** 2, p.meas_size ** 2])

    # ---- forward pass
    x = np.zeros(STATE_DIM)
    x[:4] = Z[0]
    # Seed the velocity from the first two measurements rather than starting at rest. A filter that begins
    # stationary has to catch up with a moving object over several frames, and every one of those frames
    # looks like a large innovation, so a genuinely fast object was reported as a run of outliers. The
    # velocity is directly observable from the first step; declining to use it is throwing away data.
    dt0 = max(1e-3, (ts[1] - ts[0]) / 1e9)
    x[4:] = (Z[1] - Z[0]) / dt0
    P = np.eye(STATE_DIM) * 1e3          # the first state is essentially unknown beyond its measurement
    xf, Pf, xp, Pp, Fs = [], [], [], [], []
    gated = np.zeros(n, dtype=bool)

    for k in range(n):
        dt = 0.0 if k == 0 else max(1e-3, (ts[k] - ts[k - 1]) / 1e9)
        F = _transition(dt)
        if k == 0:
            x_pred, P_pred = x.copy(), P.copy()
        else:
            x_pred = F @ x
            P_pred = F @ P @ F.T + _process_noise(dt, p)
        Fs.append(F)
        xp.append(x_pred.copy())
        Pp.append(P_pred.copy())

        y = Z[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        # Mahalanobis distance of the innovation: how surprised the model is by this measurement.
        try:
            d2 = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            d2 = 0.0
        Rk = R
        if k > 0 and d2 > p.gate_chi2:
            gated[k] = True
            Rk = R * p.gate_inflate
            S = H @ P_pred @ H.T + Rk

        K = P_pred @ H.T @ np.linalg.pinv(S)
        x = x_pred + K @ y
        P = (np.eye(STATE_DIM) - K @ H) @ P_pred
        xf.append(x.copy())
        Pf.append(P.copy())

    # ---- backward (RTS) pass: every state revisited with everything learned afterwards
    # Pre-allocated and filled backwards, so every slot is written before the read loop below. Annotated
    # because `[None] * n` types as list[None] and the smoother then reads as a sequence of None operations
    # that the RTS recursion has in fact already replaced.
    xs: list[Any] = [None] * n
    Ps: list[Any] = [None] * n
    xs[-1], Ps[-1] = xf[-1], Pf[-1]
    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        C = Pf[k] @ F.T @ np.linalg.pinv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T

    boxes = []
    for k in range(n):
        # Positional standard deviation, averaged over the two centre axes: one number a reviewer or a
        # distillation weight can use without reading a covariance matrix.
        var = float(max(Ps[k][0, 0], 0.0) + max(Ps[k][1, 1], 0.0)) / 2.0
        boxes.append({
            "ts_ns": int(ts[k]),
            "bbox": to_corners(xs[k]),
            "std": round(float(np.sqrt(var)), 3),
            "gated": bool(gated[k]),
        })

    return {"boxes": boxes, "n": n, "gated": int(gated.sum()), "smoothed": True,
            "mean_std": round(float(np.mean([b["std"] for b in boxes])), 3),
            # The model this track was smoothed under, so a surprising result can be traced to it rather
            # than guessed at.
            "process_pos": round(p.process_pos, 2), "process_size": round(p.process_size, 2),
            "coherence": coh}


def displacement(observations: list[tuple[int, list[float]]], smoothed: list[dict]) -> dict:
    """How far smoothing moved each box, so the effect is measurable rather than asserted.

    A smoother that changes nothing is not helping, and one that moves every box a long way is fighting the
    data rather than cleaning it. Both are visible here.
    """
    obs = {int(t): to_centre_size(b) for t, b in observations}
    d = [float(np.linalg.norm(to_centre_size(s["bbox"])[:2] - obs[s["ts_ns"]][:2]))
         for s in smoothed if s["ts_ns"] in obs]
    if not d:
        return {"n": 0}
    arr = np.array(d)
    return {"n": len(d), "mean_px": round(float(arr.mean()), 3),
            "p95_px": round(float(np.percentile(arr, 95)), 3),
            "max_px": round(float(arr.max()), 3)}
