"""Detections that are reflections of a real object in the bonnet, not objects.

A dashcam looks out over a glossy bonnet, and on a bright day the bonnet mirrors the scene above it. The
detector duly finds a pedestrian there, at a plausible size, with a plausible confidence, upside down. The
hood mask (services/autolabel/ego_mask.py) removes detections that sit inside the bonnet region, but only
where a hood mask was estimated at all, and it removes nothing on a windscreen reflection or on wet tarmac.

A reflection has a property no real object has: it is a vertically mirrored copy of something directly
above it. So the test is a correlation, not a heuristic about position or size:

    take the candidate patch, flip it vertically, and slide it up the column looking for the source.
    A high normalised cross-correlation at some offset means the candidate IS that thing, reflected.

NORMALISED, and that word is doing work. A raw correlation is dominated by brightness: a reflection is
darker and lower-contrast than its source, so a plain dot product scores it poorly exactly when it is most
obviously a reflection. Subtracting each patch's mean and dividing by its standard deviation compares
SHAPE, which is what survives the reflection, and discards brightness, which does not.

WHAT IT REFUSES. A flat patch has no shape to correlate: its standard deviation is near zero, the
normalisation divides by nothing, and the correlation becomes noise amplified to 1.0. Those are returned
unmeasured rather than as confident twins, because a uniformly grey patch is the single most common thing
in a road scene and calling all of them reflections would delete a lot of real road.

THE SEARCH IS UPWARD ONLY. A reflection appears BELOW its source, always, because the reflecting surface
is below the camera axis. Searching both directions doubles the false-positive rate for nothing.

NumPy is the oracle; the torch path mirrors the correlation stack and must agree to 1e-6.
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

_EPS = 1e-6
# Below this the patch is flat enough that the normalisation is dividing by noise.
MIN_STD = 3.0
# Correlation at which a candidate is called a reflection. High on purpose: deleting a real VRU costs far
# more than keeping a reflection that a reviewer will reject.
TWIN_NCC = 0.72


@dataclass(frozen=True)
class TwinVerdict:
    """Whether a detection is a reflection of something above it, and what was found.

    `ncc` is the best normalised cross-correlation over the searched offsets, and `offset_px` how far
    above the candidate its source sat. Both are reported so a threshold can be re-chosen later from the
    distribution rather than from taste.
    """

    measured: bool
    is_twin: bool
    ncc: float | None
    offset_px: int | None
    reason: str | None = None


def _norm(patch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    m = float(patch.mean())
    s = float(patch.std())
    if s < MIN_STD:
        return None
    out: npt.NDArray[np.float64] = (patch - m) / s
    return out


def _ncc_np(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    return float((a * b).mean())


def _ncc_torch(a: npt.NDArray[np.float64], stack: npt.NDArray[np.float64],
               device: str) -> npt.NDArray[np.float64]:  # pragma: no cover - GPU path
    ta = torch.as_tensor(a, device=device, dtype=torch.float64)
    ts = torch.as_tensor(stack, device=device, dtype=torch.float64)
    out: npt.NDArray[np.float64] = (ts * ta).mean(dim=(1, 2)).cpu().numpy()
    return out


def reflection_twin(gray: npt.ArrayLike, box: npt.ArrayLike, *, search_px: int | None = None,
                    step: int = 2, ncc_thr: float = TWIN_NCC,
                    device: str | None = None) -> TwinVerdict:
    """Is the detection in `box` a vertically mirrored copy of something directly above it?

    `gray` is the full grayscale frame. `search_px` is how far up to look, defaulting to three box heights,
    which covers a bonnet reflection at any realistic camera height without wandering into the sky.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("gray must be a single-channel (H, W) image")
    b = np.asarray(box, dtype=np.float64).reshape(4)
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(g.shape[1], x2), min(g.shape[0], y2)
    h, w = y2 - y1, x2 - x1
    if h < 4 or w < 4:
        return TwinVerdict(False, False, None, None,
                           "the box is too small to correlate; a few pixels match anything")

    cand = _norm(g[y1:y2, x1:x2])
    if cand is None:
        return TwinVerdict(False, False, None, None,
                           "the candidate patch is nearly uniform, so the correlation would be noise "
                           "amplified to 1.0 rather than a measurement")
    flipped = cand[::-1, :]

    reach = search_px if search_px is not None else 3 * h
    offsets = list(range(h, min(reach, y1) + 1, max(1, step)))
    if not offsets:
        return TwinVerdict(False, False, None, None,
                           "there is nothing above the box to be a reflection of")

    patches, kept = [], []
    for off in offsets:
        top = y1 - off
        src = _norm(g[top:top + h, x1:x2])
        if src is None:
            continue                 # a flat region above is not evidence either way
        patches.append(src)
        kept.append(off)
    if not patches:
        return TwinVerdict(False, False, None, None,
                           "every region above the box is nearly uniform, so nothing could be correlated")

    if _HAS_TORCH and device is not None and str(device) != "cpu" and torch.cuda.is_available():
        scores = _ncc_torch(flipped, np.stack(patches), str(device))  # pragma: no cover - GPU path
    else:
        scores = np.array([_ncc_np(flipped, p) for p in patches], dtype=np.float64)

    i = int(np.argmax(scores))
    best = float(scores[i])
    return TwinVerdict(True, bool(best >= ncc_thr), round(best, 6), int(kept[i]))


def screen_detections(gray: npt.ArrayLike, boxes: npt.ArrayLike, *, ncc_thr: float = TWIN_NCC,
                      device: str | None = None) -> dict[str, Any]:
    """Screen a frame's detections. Returns {"verdicts", "keep", "n_twins", "n_unmeasured"}.

    `keep` is the indices that are NOT reflections, plus every detection the test could not measure. An
    unmeasurable detection is kept on purpose: the null result is "we could not tell", and dropping on
    that would delete real objects wherever the image happens to be flat.
    """
    bs = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    verdicts = [reflection_twin(gray, bs[i], ncc_thr=ncc_thr, device=device)
                for i in range(bs.shape[0])]
    keep = [i for i, v in enumerate(verdicts) if not (v.measured and v.is_twin)]
    return {"verdicts": tuple(verdicts), "keep": keep,
            "n_twins": sum(1 for v in verdicts if v.measured and v.is_twin),
            "n_unmeasured": sum(1 for v in verdicts if not v.measured)}


__all__ = ["TwinVerdict", "reflection_twin", "screen_detections", "TWIN_NCC", "MIN_STD"]
