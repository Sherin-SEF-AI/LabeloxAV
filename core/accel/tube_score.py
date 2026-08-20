"""Whether a detection is part of a coherent object over time, or a single confident frame.

A per-frame detector scores each frame alone, so a box that appears for one frame in the middle of nothing
gets the same confidence as one that is the twentieth frame of a stable track. Those are not equally likely
to be right, and confidence cannot say so because it never saw the other frames.

The tube score is what the track adds. It combines three things a real object does and an artifact does
not:

    continuity  the fraction of the frame span where the track actually has a box. A detection that
                appears, vanishes, and reappears is describing something that does not persist.
    stability   how much the box jitters relative to its own size, from core/accel/uncertainty.py's
                flicker measure. A real object moves smoothly; a false positive breathes and jumps.
    agreement   how consistently the track was given the same class. A tube whose class changes every
                other frame is not one object being tracked, it is a detector guessing.

WHY IT IS NOT JUST AVERAGED CONFIDENCE. Averaging over the tube would smooth a genuinely uncertain
detection into a confident one, which is the failure this is meant to catch rather than commit. The tube
score is deliberately independent of confidence, so that services/oraclyx/joint_calibration.py can fit
P(correct | conf, tube) over two axes that carry different information. Multiplying them here would throw
away the second axis before anyone could use it.

A SHORT TUBE IS NOT A BAD TUBE. A track two frames long has no jitter to measure and near-perfect
continuity by construction, and reporting 1.0 for it would rank it above a twenty-frame track with one
gap. Below `min_frames` the score is None with a reason, and the calibration treats those as their own
bucket rather than as good ones.

NumPy is the oracle; the torch path mirrors the per-track reduction and must agree to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

# Below this a track has not shown enough of itself to be scored. Two frames give one displacement and no
# variance; three is the smallest span where "smooth" and "jumpy" differ.
MIN_FRAMES = 3
_EPS = 1e-9

# How the three signals combine. Continuity is weighted highest because a gap is the least ambiguous
# evidence: jitter can come from a genuinely fast-moving object, and a class flip can come from two similar
# classes, but a track that is absent for half its span is describing something that was not there.
_W_CONTINUITY, _W_STABILITY, _W_AGREEMENT = 0.5, 0.3, 0.2


@dataclass(frozen=True)
class TubeScore:
    """One track's temporal coherence, with the three components kept separate.

    Separate because they fail differently and a single number cannot be acted on: a low score from gaps
    means the tracker dropped frames, a low score from jitter means the box is unstable, and a low score
    from class disagreement means two classes are being confused along the track. Those want three
    different fixes.

    `measured` is False, with a reason, for a track too short to have shown anything.
    """

    measured: bool
    score: float | None
    continuity: float | None
    stability: float | None
    agreement: float | None
    n_frames: int
    n_present: int
    span: int
    reason: str | None = None


def _unmeasured(n_present: int, span: int, reason: str) -> TubeScore:
    return TubeScore(measured=False, score=None, continuity=None, stability=None, agreement=None,
                     n_frames=span, n_present=n_present, span=span, reason=reason)


def _stability_np(boxes: npt.NDArray[np.float64], valid: npt.NDArray[np.bool_]) -> float:
    """1 - normalised jitter, in [0, 1]. Displacement measured relative to the box's own size, so a
    distant object crossing a few pixels is not scored as more stable than a near one crossing many."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    size = np.sqrt(np.clip(w * h, _EPS, None))

    pair = valid[1:] & valid[:-1]
    if not pair.any():
        return 0.0
    scale = np.maximum((size[1:] + size[:-1]) / 2.0, _EPS)
    # Second difference, not first: an object moving steadily across the frame has a large first
    # difference and is perfectly stable. What marks an artifact is the motion CHANGING frame to frame.
    d = np.stack([np.diff(cx), np.diff(cy)], axis=0) / scale
    if d.shape[1] < 2:
        jitter = float(np.abs(d[:, pair]).mean()) if pair.any() else 0.0
    else:
        accel = np.abs(np.diff(d, axis=1))
        m = pair[1:] & pair[:-1]
        jitter = float(accel[:, m].mean()) if m.any() else float(np.abs(d[:, pair]).mean())
    return float(np.clip(1.0 - jitter, 0.0, 1.0))


def tube_score(boxes: npt.ArrayLike, valid: npt.ArrayLike | None = None,
               class_ids: npt.ArrayLike | None = None, *, min_frames: int = MIN_FRAMES) -> TubeScore:
    """Score one track. `boxes` is (T, 4) xyxy over the track's frame span; `valid` marks present frames."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    span = b.shape[0]
    v = (np.ones(span, dtype=bool) if valid is None
         else np.asarray(valid, dtype=bool).reshape(-1))
    if v.size != span:
        raise ValueError(f"{span} boxes against {v.size} validity flags")
    n_present = int(v.sum())
    if span < min_frames or n_present < min_frames:
        return _unmeasured(n_present, span,
                           f"the track spans {span} frames with {n_present} present, below the "
                           f"{min_frames} needed to tell a smooth object from a jumpy one")

    continuity = n_present / span
    stability = _stability_np(b, v)

    if class_ids is None:
        agreement = 1.0
    else:
        c = np.asarray(class_ids, dtype=np.int64).reshape(-1)[v]
        if c.size == 0:
            agreement = 1.0
        else:
            vals, counts = np.unique(c, return_counts=True)
            agreement = float(counts.max() / c.size)

    score = (_W_CONTINUITY * continuity + _W_STABILITY * stability + _W_AGREEMENT * agreement)
    return TubeScore(measured=True, score=round(float(np.clip(score, 0.0, 1.0)), 6),
                     continuity=round(continuity, 6), stability=round(stability, 6),
                     agreement=round(agreement, 6), n_frames=span, n_present=n_present, span=span)


def _batch_np(cont: npt.NDArray[np.float64], stab: npt.NDArray[np.float64],
              agree: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = _W_CONTINUITY * cont + _W_STABILITY * stab + _W_AGREEMENT * agree
    return out


def _batch_torch(cont: npt.NDArray[np.float64], stab: npt.NDArray[np.float64],
                 agree: npt.NDArray[np.float64],
                 device: str) -> npt.NDArray[np.float64]:  # pragma: no cover - GPU path
    c = torch.as_tensor(cont, device=device, dtype=torch.float64)
    s = torch.as_tensor(stab, device=device, dtype=torch.float64)
    a = torch.as_tensor(agree, device=device, dtype=torch.float64)
    out: npt.NDArray[np.float64] = (_W_CONTINUITY * c + _W_STABILITY * s + _W_AGREEMENT * a).cpu().numpy()
    return out


def score_tracks(tracks: list[dict[str, Any]], *, min_frames: int = MIN_FRAMES,
                 device: str | None = None) -> dict[str, Any]:
    """Score many tracks. Each entry is {"boxes", optional "valid", optional "class_ids"}.

    Returns {"scores" (NaN where unmeasurable), "measured", "detail", "n_unmeasured"}. NaN rather than
    zero for the same reason the margin scorer uses it: a track too short to score is not a bad track, and
    a zero would sort it below every genuinely incoherent one.
    """
    detail = [tube_score(t["boxes"], t.get("valid"), t.get("class_ids"), min_frames=min_frames)
              for t in tracks]
    n = len(detail)
    cont = np.zeros(n)
    stab = np.zeros(n)
    agree = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for i, d in enumerate(detail):
        if not d.measured:
            continue
        ok[i] = True
        cont[i], stab[i], agree[i] = d.continuity, d.stability, d.agreement

    if _HAS_TORCH and device is not None and str(device) != "cpu" and torch.cuda.is_available():
        raw = _batch_torch(cont, stab, agree, str(device))  # pragma: no cover - GPU path
    else:
        raw = _batch_np(cont, stab, agree)
    return {"scores": np.where(ok, np.clip(raw, 0.0, 1.0), np.nan), "measured": ok,
            "detail": tuple(detail), "n_unmeasured": int((~ok).sum())}


__all__ = ["TubeScore", "tube_score", "score_tracks", "MIN_FRAMES"]
