"""Face and license-plate detectors for Gate A. Both return [(x1, y1, x2, y2, score)] in pixels and
run on CPU by default so PII never competes with the 16 GB autolabel GPU budget.

Face: OpenCV YuNet (cv2.FaceDetectorYN), no new dependency (opencv is already a base dep), swappable
.onnx. Plate: a config-pointed Ultralytics YOLO weight (the exact model is treated as swappable like
the YOLO26/SAM3 substitution). When a model file is missing the detector reports unavailable and
returns no regions; the anonymizer decides whether that is fatal (it is, when the gate is enabled).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.logging import get_logger

log = get_logger("pii_detectors")

Region = tuple[float, float, float, float, float]  # x1, y1, x2, y2, score


class FaceDetector:
    def __init__(self, weights: str, conf: float) -> None:
        self.weights = weights
        self.conf = conf
        self._det = None
        if Path(weights).exists():
            # input_size is set per-frame; score/nms/top_k from defaults + our conf.
            self._det = cv2.FaceDetectorYN.create(weights, "", (320, 320), conf, 0.3, 5000)
            log.info("pii.face_detector_loaded", weights=weights)
        else:
            log.warning("pii.face_detector_unavailable", weights=weights)

    @property
    def available(self) -> bool:
        return self._det is not None

    def detect(self, image_bgr: np.ndarray) -> list[Region]:
        if self._det is None:
            return []
        h, w = image_bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image_bgr)
        out: list[Region] = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                score = float(f[-1])
                if score >= self.conf:
                    out.append((x, y, x + fw, y + fh, score))
        return out


class PlateDetector:
    def __init__(self, weights: str, conf: float, device: str) -> None:
        self.weights = weights
        self.conf = conf
        self.device = device
        self._model = None
        if Path(weights).exists():
            try:
                from ultralytics import YOLO

                self._model = YOLO(weights)
                log.info("pii.plate_detector_loaded", weights=weights)
            except Exception as exc:  # noqa: BLE001
                log.warning("pii.plate_detector_load_failed", weights=weights, error=str(exc))
        else:
            log.warning("pii.plate_detector_unavailable", weights=weights)

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, image_bgr: np.ndarray) -> list[Region]:
        if self._model is None:
            return []
        res = self._model.predict(image_bgr, conf=self.conf, device=self.device, verbose=False)
        out: list[Region] = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(b.conf[0])))
        return out
