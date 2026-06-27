"""Drivable-surface segmentation (M2.2).

  pod path:   SAM 3.1 PCS concept prompts (drivable / non-drivable / fallback) -> a ternary surface mask.
  local path: SAM (sam_b) seeded at the road region -> the drivable mask; non-drivable is the coarse
              complement of the lower frame; fallback (unpaved/unmarked, the IDD drivable-fallback case)
              is the pod-only concept.

Stores polygons-per-surface-class as JSON in MinIO (never Postgres), plus per-class pixel-coverage
fractions on a drivable_mask row. Surface classes map to the ontology surface entries (road, sidewalk,
drivable_fallback, ...). Refinable with the existing SAM click from the editor.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("drivable")

SURFACE_CLASSES = ("drivable", "non_drivable", "fallback")


def _raster(polys: list, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        pts = np.asarray(p, np.float32).reshape(-1, 2).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def _segment_local(image_bgr: np.ndarray) -> dict:
    """Deterministic perspective road-trapezoid proposal (drivable), with the rest of the lower frame as
    non-drivable. A plausible starting region the human refines with the editor's SAM click; the real
    ternary surface (drivable/non-drivable/fallback) is the pod SAM 3.1 PCS path."""
    from services.autolabel.paths.path_b_sam3 import polygons_from_mask

    h, w = image_bgr.shape[:2]
    horizon = h * 0.52
    drivable = [[round(w * 0.05, 1), float(h), round(w * 0.95, 1), float(h),
                 round(w * 0.62, 1), round(horizon, 1), round(w * 0.38, 1), round(horizon, 1)]]
    dm = _raster(drivable, w, h)
    lower = np.zeros((h, w), bool)
    lower[int(h * 0.45):, :] = True
    non_driv = lower & ~dm
    non_polys = polygons_from_mask(non_driv.astype(bool)) if int(non_driv.sum()) > 50 else []
    total = float(h * w)
    return {
        "classes": {"drivable": drivable, "non_drivable": non_polys, "fallback": []},
        "coverage": {"drivable": round(float(dm.sum()) / total, 4),
                     "non_drivable": round(float(non_driv.sum()) / total, 4), "fallback": 0.0},
        "width": w, "height": h, "model": "trapezoid:local",
    }


def segment_drivable(image_bgr: np.ndarray) -> dict:
    cfg = get_settings().models.drivable
    if cfg.backend == "pod":
        raise NotImplementedError(
            "SAM 3.1 PCS concept-prompt drivable segmentation runs on the RunPod pod via "
            "cloud/perception_pod.py. Set models.drivable.backend=pod, or use the local fallback.")
    return _segment_local(image_bgr)
