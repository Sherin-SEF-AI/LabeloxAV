"""Two segmenters, one mask, and an honest number for how much to trust it.

SAM 2 produces the mask and SAM 1 scores it. They are independent enough to disagree usefully: measured on
70 real objects from this corpus, their masks agree at a median IoU of 0.893, but a fifth of the time they
fall below 0.8 and the disagreement is not spread evenly. It concentrates on `rider`, `motorcycle` and
`autorickshaw` - the cases where "is the mask the person, the bike, or both?" has no single right answer,
and which the pack already marks as the one confusion clique that crosses a safety boundary.

So the second model is not a better mask, it is a second opinion, and the value is in where the two differ.
This is the same shape the engine already uses twice: cross-path agreement decides whether a fused
classification may auto-accept, and `services/calyx/ego_propagate.py` refuses to write a box at all when the
geometric and tracker estimates disagree, recording the conflict instead.

Deliberately not a fusion. Intersecting or unioning the two masks produces a third mask that is arguably
better than either and provably nothing, because there is no ground truth here to check it against. An
agreement score can be validated later against human masks; a fused mask would have quietly replaced the
thing that would have been validated.

`agreement` is None, never a number, when the verifier cannot run. A missing second opinion is not a
confident one, and `services/agent/propagate_agent.py::_appearance` sets the same precedent for exactly
this reason.
"""

from __future__ import annotations

import threading

import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("sam_hybrid")

_lock = threading.Lock()
_verifier = None
_verifier_failed = False


def _get_verifier():
    """The second segmenter, loaded once. None when it cannot be had, and it is only tried once.

    A verifier that fails to load fails for a reason that will not change during the run - missing weights,
    no CUDA - and retrying per object would pay the cost on every one of them.
    """
    global _verifier, _verifier_failed
    if _verifier is not None or _verifier_failed:
        return _verifier
    with _lock:
        if _verifier is None and not _verifier_failed:
            cfg = get_settings().models.openvocab
            try:
                from ultralytics import SAM

                _verifier = SAM(cfg.seg_verify_weights)
                log.info("sam_hybrid.verifier_loaded", weights=cfg.seg_verify_weights)
            except Exception as exc:  # noqa: BLE001 - no second opinion is not a failed segmentation
                _verifier_failed = True
                log.warning("sam_hybrid.verifier_unavailable", error=str(exc)[:160])
    return _verifier


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks. 0.0 when either is empty, which is a real disagreement rather than a
    division to guard against."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def _first_mask(result) -> np.ndarray | None:
    m = getattr(result, "masks", None)
    if m is None or m.data is None or not len(m.data):
        return None
    arr = m.data[0]
    return arr.cpu().numpy().astype(bool) if hasattr(arr, "cpu") else np.asarray(arr, dtype=bool)


def verify_mask(image_bgr: np.ndarray, box: list[float], primary_mask: np.ndarray) -> float | None:
    """How much an independent segmenter agrees with this mask, or None if it could not be asked.

    The box is re-prompted rather than the mask re-used, so the two opinions share only the prompt. Feeding
    the first mask to the second model would make the second agree with it by construction.
    """
    v = _get_verifier()
    if v is None or primary_mask is None:
        return None
    try:
        res = v(image_bgr, bboxes=[list(box)], device=get_settings().gpu.device, verbose=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("sam_hybrid.verify_failed", error=str(exc)[:160])
        return None
    other = _first_mask(res[0]) if res else None
    if other is None:
        # The verifier saw the crop and found nothing where the primary found something. That is the
        # strongest disagreement available, and reporting it as 0.0 rather than None is the point.
        return 0.0
    return mask_iou(primary_mask, other)


def agrees(agreement: float | None, threshold: float | None = None) -> bool | None:
    """Whether a mask cleared the agreement floor. None in, None out - an unasked verifier has no verdict."""
    if agreement is None:
        return None
    thr = get_settings().models.openvocab.seg_agree_iou if threshold is None else threshold
    return agreement >= thr
