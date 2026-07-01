"""Lazy SAM service for interactive click-to-segment in the annotation workspace. One model is
loaded on first use and kept resident; a click becomes a single point/box prompt, returned as
polygons the canvas can draw."""

from __future__ import annotations

import threading

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.paths.path_b_sam3 import polygons_from_mask

log = get_logger("sam_service")

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from ultralytics import SAM

                cfg = get_settings().models.openvocab
                _model = SAM(cfg.seg_weights)
                log.info("sam_service.loaded", weights=cfg.seg_weights)
    return _model


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
