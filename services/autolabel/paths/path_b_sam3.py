"""Path B: open-vocabulary detect + segment of the long tail, driven by ontology concept phrases.

Spec target is SAM 3.1 Promptable Concept Segmentation (one model, text -> detect+segment+track).
That model is not yet in the installed Ultralytics, so Path B is realized today as the spec's own
D1 fallback shape: YOLO-World (open-vocab text -> boxes over the ontology phrases) followed by SAM
(box -> mask). Same Path B contract: a concept phrase yields boxes plus masks, no retraining.

To swap in real SAM 3.1 PCS later, replace this class's load()/infer() with the PCS predictor and
point models.openvocab at sam3.pt; the fusion contract (RawDetection with class + box + mask) is
unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.base import RawDetection, mask_to_bbox

log = get_logger("path_b")


class Sam3Path:
    name = "path_b_sam3"

    def __init__(self, supported_ids: set[int] | None = None) -> None:
        self.settings = get_settings()
        self.onto = get_ontology()
        self._world = None
        self._sam = None
        # M-Q.0: prompt only with the grounded supported set. An ungrounded class name in the open-vocab
        # concept list is the hallucination source (the model finds water_bottles because it is asked to),
        # so anything outside the set is dropped here and folds to fallback downstream. None means all
        # classes (back-compat for callers that do not pass the grounded set).
        classes = self.onto.classes if supported_ids is None else [c for c in self.onto.classes if c.id in supported_ids]
        # India/rare classes first so the open-vocab budget favors where Path A is weakest.
        self._classes = sorted(classes, key=lambda c: (not c.india, c.id))
        self._phrases = [c.name.replace("_", " ") for c in self._classes]
        self.model_version = (
            f"{self.settings.models.openvocab.detector_weights}+{self.settings.models.openvocab.seg_weights}"
        )

    def load(self) -> None:
        from ultralytics import SAM, YOLOWorld

        cfg = self.settings.models.openvocab
        self._world = YOLOWorld(cfg.detector_weights)
        self._world.set_classes(self._phrases)
        self._sam = SAM(cfg.seg_weights)
        log.info(
            "path_b.loaded",
            detector=cfg.detector_weights,
            segmenter=cfg.seg_weights,
            concepts=len(self._phrases),
        )

    def unload(self) -> None:
        self._world = None
        self._sam = None

    def infer(self, image_bgr: np.ndarray) -> list[RawDetection]:
        if self._world is None or self._sam is None:
            raise RuntimeError("Sam3Path not loaded")
        cfg = self.settings.models.openvocab
        dev = self.settings.gpu.device

        wres = self._world.predict(
            source=image_bgr, imgsz=self.settings.models.yolo.imgsz, half=cfg.half,
            conf=cfg.conf, device=dev, verbose=False,
        )
        r0 = wres[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clsidx = r0.boxes.cls.cpu().numpy().astype(int)

        # Keep the most confident boxes within the per-frame budget.
        order = np.argsort(-confs)[: cfg.max_boxes]
        xyxy, confs, clsidx = xyxy[order], confs[order], clsidx[order]

        masks = self._segment(image_bgr, xyxy, dev)

        dets: list[RawDetection] = []
        for i in range(len(xyxy)):
            ci = clsidx[i]
            spec = self._classes[ci] if 0 <= ci < len(self._classes) else self._classes[-1]
            mask = masks[i] if masks is not None and i < len(masks) else None
            bbox = mask_to_bbox(mask) if mask is not None else tuple(float(v) for v in xyxy[i])
            if bbox is None:
                bbox = tuple(float(v) for v in xyxy[i])
            dets.append(
                RawDetection(
                    path=self.name,
                    bbox=bbox,
                    conf=float(confs[i]),
                    class_name=spec.name,
                    class_id=spec.id,
                    model_version=self.model_version,
                    mask=mask,
                )
            )
        return dets

    def _segment(self, image_bgr: np.ndarray, xyxy: np.ndarray, dev: str) -> np.ndarray | None:
        if len(xyxy) == 0:
            return None
        sres = self._sam.predict(source=image_bgr, bboxes=xyxy, device=dev, verbose=False)
        m = sres[0].masks
        if m is None or m.data is None:
            return None
        return m.data.cpu().numpy().astype(bool)


def polygons_from_mask(mask: np.ndarray, epsilon_frac: float = 0.005, keep_holes: bool = False) -> list[list[float]]:
    """Compact polygon encoding for persistence, flattened [x,y,...] per ring.

    Default (keep_holes=False) keeps only external contours, which is right for a solid object. With
    keep_holes=True the interior contours are returned too, so a mask with a hole (an erased region, or a
    stuff region cut by an occluding vehicle) round-trips: consumers interpret the ring list with the
    even-odd rule, so a ring nested inside another reads as a hole rather than a second filled blob.
    """
    m = (mask.astype(np.uint8)) * 255
    mode = cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        polys.append([float(v) for pt in approx.reshape(-1, 2) for v in pt])
    return polys
