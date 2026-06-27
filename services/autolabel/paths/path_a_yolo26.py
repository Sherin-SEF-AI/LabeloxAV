"""Path A: YOLO26, the hot deterministic detector. NMS-free, end-to-end. Runs on every frame,
strong on the head distribution. Detections are mapped from the COCO supervocabulary into the
India ontology where a mapping exists; everything else lands in a fallback class so nothing the
detector sees is silently dropped.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.base import RawDetection

log = get_logger("path_a")

# COCO class name -> ontology class name. Ambiguous COCO 'car' defaults to sedan (the head guess);
# fusion and the VLM refine it. Unmapped detections become object_fallback (never dropped).
COCO_TO_ONTOLOGY: dict[str, str] = {
    "person": "pedestrian",
    "bicycle": "cycle",
    "motorcycle": "motorcycle",
    "car": "sedan",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_signal",
    "stop sign": "traffic_sign",
    "fire hydrant": "pole",
    "bench": "object_fallback",
    "cat": "object_fallback",
    "dog": "dog",
    "horse": "cattle",
    "sheep": "goat",
    "cow": "cattle",
    "elephant": "object_fallback",
    "bird": "object_fallback",
}


class YoloPath:
    name = "path_a_yolo26"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.onto = get_ontology()
        self._model = None
        self.model_version = self.settings.models.yolo.weights

    def load(self) -> None:
        from ultralytics import YOLO

        cfg = self.settings.models.yolo
        self._model = YOLO(cfg.weights)
        log.info("path_a.loaded", weights=cfg.weights, device=self.settings.gpu.device)

    def unload(self) -> None:
        self._model = None

    def _to_ontology(self, model_name: str) -> tuple[str, int]:
        # A fine-tuned/promoted model emits ontology class names directly; a stock COCO model emits
        # COCO names that we map. Support both so a promoted model is immediately usable in Path A.
        if self.onto.has_name(model_name):
            name = model_name
        else:
            name = COCO_TO_ONTOLOGY.get(model_name, "object_fallback")
            if not self.onto.has_name(name):
                name = "object_fallback"
        return name, self.onto.by_name(name).id

    def infer(self, image_bgr: np.ndarray) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("YoloPath not loaded")
        cfg = self.settings.models.yolo
        results = self._model.predict(
            source=image_bgr,
            imgsz=cfg.imgsz,
            half=cfg.half,
            conf=cfg.conf,
            device=self.settings.gpu.device,
            verbose=False,
        )
        dets: list[RawDetection] = []
        names = results[0].names if results else {}
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                coco_name = names.get(int(b.cls[0]), "object")
                cls_name, cls_id = self._to_ontology(coco_name)
                dets.append(
                    RawDetection(
                        path=self.name,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        conf=conf,
                        class_name=cls_name,
                        class_id=cls_id,
                        model_version=self.model_version,
                        extra={"coco_name": coco_name},
                    )
                )
        return dets
