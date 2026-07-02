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

# The real local model. Default is Mask2Former fine-tuned on Mapillary Vistas: Mapillary is globally diverse
# (developing-country roads included), so it segments Indian dashcam roads far better than Cityscapes, which
# under-segments them. Set LBX_DRIVABLE_LOCAL_MODEL to a SegFormer-Cityscapes tag for a lighter/faster (but
# lower-quality on Indian roads) option. This mirrors the pod's model choice (cloud/perception_pod.py).
import os

_SEG_MODEL = os.environ.get("LBX_DRIVABLE_LOCAL_MODEL", "facebook/mask2former-swin-large-mapillary-vistas-semantic")
# Substring rules over the model's own id2label -> ternary surface (identical to the pod's mapping).
_DRIVE = ("road", "driveway", "crosswalk", "bike lane", "bike-lane", "service lane", "service-lane", "parking")
_NONDR = ("sidewalk", "curb", "pedestrian")
_FALL = ("terrain", "pothole", "unpaved", "dirt")
_CITY_SURFACE = {0: "drivable", 1: "non_drivable", 9: "fallback"}   # cityscapes road / sidewalk / terrain
_seg_state: dict = {}


def _build_surface_map(id2label: dict) -> dict:
    """Map a model's own label ids to the ternary surface via substring rules (railroad excluded)."""
    out: dict = {}
    for i, lab in id2label.items():
        low = str(lab).lower()
        if "rail" in low:
            continue
        if any(k in low for k in _DRIVE):
            out[int(i)] = "drivable"
        elif any(k in low for k in _NONDR):
            out[int(i)] = "non_drivable"
        elif any(k in low for k in _FALL):
            out[int(i)] = "fallback"
    return out


def _seg_model():
    if "model" not in _seg_state:
        import torch

        dev = get_settings().gpu.device if torch.cuda.is_available() else "cpu"
        if "mask2former" in _SEG_MODEL:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            proc = AutoImageProcessor.from_pretrained(_SEG_MODEL)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(_SEG_MODEL).to(dev).eval()
            surf = _build_surface_map(model.config.id2label)
            _seg_state.update(kind="mask2former", proc=proc, model=model, torch=torch, dev=dev, surf=surf)
        else:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            proc = SegformerImageProcessor.from_pretrained(_SEG_MODEL)
            model = SegformerForSemanticSegmentation.from_pretrained(_SEG_MODEL).to(dev).eval()
            _seg_state.update(kind="segformer", proc=proc, model=model, torch=torch, dev=dev, surf=_CITY_SURFACE)
        log.info("drivable.seg_loaded", model=_SEG_MODEL, device=dev, kind=_seg_state["kind"])
    return _seg_state


def _surface_polys(mask: np.ndarray) -> list[list[float]]:
    """Boolean surface mask -> area-filtered simplified polygons (drops speckle so the road reads clean)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 250:
            continue
        eps = 0.004 * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) >= 3:
            polys.append([round(float(v), 1) for xy in ap for v in xy])
    return polys


def _segment_seg(image_bgr: np.ndarray) -> dict:
    """Real drivable surface from the semantic segmenter: the road pixels themselves, not a geometric guess."""
    from PIL import Image

    s = _seg_model()
    h, w = image_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inp = s["proc"](images=pil, return_tensors="pt").to(s["dev"])
    with s["torch"].inference_mode():
        out = s["model"](**inp)
    if s["kind"] == "mask2former":
        labels = s["proc"].post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0].cpu().numpy()
        tag = "mask2former-mapillary:local"
    else:
        up = s["torch"].nn.functional.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)
        labels = up.argmax(dim=1)[0].cpu().numpy()
        tag = "segformer-cityscapes:local"
    classes: dict = {}
    cov: dict = {}
    total = float(h * w) or 1.0
    for cls in SURFACE_CLASSES:
        ids = [i for i, c in s["surf"].items() if c == cls]
        m = np.isin(labels, ids) if ids else np.zeros_like(labels, bool)
        classes[cls] = _surface_polys(m)
        cov[cls] = round(float(m.sum()) / total, 4)
    return {"classes": classes, "coverage": cov, "width": w, "height": h, "model": tag}


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
    # Prefer the real SegFormer-Cityscapes road segmentation; the geometric trapezoid is only a safety net
    # when the model cannot run (no GPU / weights unavailable), never the default -- it can't match a road.
    try:
        return _segment_seg(image_bgr)
    except Exception as exc:  # noqa: BLE001
        log.warning("drivable.seg_unavailable", error=str(exc))
        return _segment_local(image_bgr)
