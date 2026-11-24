"""Lazy SAM service for interactive click-to-segment in the annotation workspace. One model is
loaded on first use and kept resident; a click becomes a single point/box prompt, returned as
polygons the canvas can draw."""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.paths.path_b_openvocab import polygons_from_mask

log = get_logger("sam_service")

def _model_key() -> str:
    from services.autolabel.paths.path_b_openvocab import sam_model_key

    return sam_model_key(get_settings().models.openvocab.seg_weights)


def _get_model():
    """The segmenter, shared with the auto-label path and the describe tool through the registry.

    It was a module global here and a second and third copy of the same checkpoint there. Idle, it can now
    be evicted when a job needs the room, and it reloads on the next click.
    """
    from core.model_manager import manager
    from services.autolabel.paths.path_b_openvocab import load_sam

    cfg = get_settings()
    return manager().get(_model_key(), lambda: load_sam(cfg.models.openvocab.seg_weights),
                         device=cfg.gpu.device)


def segment(
    image_bgr: np.ndarray,
    points: list[list[float]] | None = None,
    labels: list[int] | None = None,
    box: list[float] | None = None,
    precise: bool = False,
) -> dict:
    model = _get_model()
    dev = get_settings().gpu.device
    kwargs: dict = {"device": dev, "verbose": False}
    if box is not None:
        kwargs["bboxes"] = [box]
    if points:
        kwargs["points"] = points
        kwargs["labels"] = labels or [1] * len(points)

    from core.model_manager import manager

    # One prompt at a time on the shared instance: the predictor keeps the embedded image on itself.
    with manager().infer_lock(_model_key()):
        res = model.predict(source=image_bgr, **kwargs)
    masks = res[0].masks
    if masks is None or masks.data is None or len(masks.data) == 0:
        return {"polygons": [], "bbox": None}

    m = masks.data[0].cpu().numpy().astype(bool)
    # Panoptic wants segments that tile without overlap: follow the visible edge tightly (a fixed pixel
    # tolerance, so a large stuff region is no coarser than a small one) and keep interior holes where a
    # vehicle occludes the region. Semantic keeps the lighter perimeter-relative simplification.
    polys = polygons_from_mask(m, keep_holes=True, epsilon_px=1.5) if precise else polygons_from_mask(m)
    ys, xs = np.where(m)
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if xs.size else None
    return {"polygons": polys, "bbox": bbox}
