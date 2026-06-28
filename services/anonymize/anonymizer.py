"""PiiAnonymizer: detect faces + plates, irreversibly blur them in place, report an audit record.

Used by the ingest plane before any frame reaches the object store (Gate A). The blur mutates the
numpy array in place, so the subsequent JPEG encode stores an already-anonymized frame: no clean
copy ever lands in storage. Detectors are injectable for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from core.config import PiiSettings, get_settings
from core.logging import get_logger
from services.anonymize.detectors import FaceDetector, PlateDetector

log = get_logger("anonymizer")


@dataclass
class PiiResult:
    n_faces: int = 0
    n_plates: int = 0
    regions: list[dict] = field(default_factory=list)
    method_version: str = ""


class PiiAnonymizer:
    def __init__(
        self,
        cfg: PiiSettings,
        face_detector: FaceDetector | None = None,
        plate_detector: PlateDetector | None = None,
    ) -> None:
        self.cfg = cfg
        self.face = face_detector if face_detector is not None else FaceDetector(cfg.face_weights, cfg.face_conf)
        self.plate = (
            plate_detector if plate_detector is not None else PlateDetector(cfg.plate_weights, cfg.plate_conf, cfg.device)
        )
        # Fail loud when the gate is on but a required detector is unavailable: storing un-anonymized
        # frames would create a legally-unsellable dataset (DPDPA). Faces are always required. Plates are
        # required by default (plate_mandatory) so an absent plate model can never silently pass plates
        # through in the clear; opting out is an explicit, audited choice for face-only corpora.
        if cfg.enabled and not self.face.available:
            raise RuntimeError(
                "PII gate enabled but the face detector is unavailable. Run `make pii-models` "
                "or set LBX_PII__ENABLED=false (audited opt-out)."
            )
        if cfg.enabled and cfg.plate_mandatory and not self.plate.available:
            raise RuntimeError(
                f"PII gate enabled but the license-plate detector is unavailable; plates would reach the "
                f"object store un-blurred (DPDPA). Provide plate weights at {cfg.plate_weights} "
                f"(run `make pii-models`), or set LBX_PII__PLATE_MANDATORY=false for a provably face-only "
                f"corpus (audited opt-out)."
            )
        self.method_version = (
            f"{Path(cfg.face_weights).stem}+{Path(cfg.plate_weights).stem}@{cfg.blur_method}-k{cfg.kernel}"
        )

    def _blur_region(self, image_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> None:
        h, w = image_bgr.shape[:2]
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
        if ix2 <= ix1 or iy2 <= iy1:
            return
        roi = image_bgr[iy1:iy2, ix1:ix2]
        if self.cfg.blur_method == "pixelate":
            rh, rw = roi.shape[:2]
            block = max(2, self.cfg.kernel // 4)
            small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR)
            image_bgr[iy1:iy2, ix1:ix2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            k = self.cfg.kernel if self.cfg.kernel % 2 == 1 else self.cfg.kernel + 1
            image_bgr[iy1:iy2, ix1:ix2] = cv2.GaussianBlur(roi, (k, k), 0)

    def anonymize(self, image_bgr: np.ndarray) -> PiiResult:
        regions: list[dict] = []
        faces = self.face.detect(image_bgr)
        plates = self.plate.detect(image_bgr)
        for x1, y1, x2, y2, s in faces:
            self._blur_region(image_bgr, x1, y1, x2, y2)
            regions.append({"type": "face", "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)], "score": round(s, 3)})
        for x1, y1, x2, y2, s in plates:
            self._blur_region(image_bgr, x1, y1, x2, y2)
            regions.append({"type": "plate", "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)], "score": round(s, 3)})
        return PiiResult(n_faces=len(faces), n_plates=len(plates), regions=regions, method_version=self.method_version)


@lru_cache(maxsize=1)
def get_anonymizer() -> PiiAnonymizer:
    return PiiAnonymizer(get_settings().pii)
