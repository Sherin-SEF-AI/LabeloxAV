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
    n_text: int = 0
    # False when text was still detectable after two blur passes. Such a frame must not be stored or
    # served: it is not a frame with a caveat, it is a frame that leaks, and the release attestation would
    # otherwise assert it passed the PII gate.
    released: bool = True
    redaction_reason: str | None = None


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
            plate_detector if plate_detector is not None else PlateDetector(cfg.plate_weights, cfg.plate_conf, cfg.device,
                                        imgsz_cap=cfg.plate_imgsz_cap)
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
        # The inference resolution is part of the method, not a tuning detail. Every frame in this corpus
        # was processed at the library default of 640, which downsampled a 1920x1080 dashcam frame by three
        # and hid every small plate in it. Recording the cap is what lets a backfill find those frames
        # afterwards: without it, a frame processed by a method that could not see plates is indistinguishable
        # from one that genuinely had none.
        self.method_version = (
            f"{Path(cfg.face_weights).stem}+{Path(cfg.plate_weights).stem}"
            f"@{cfg.blur_method}-k{cfg.kernel}-imgsz{cfg.plate_imgsz_cap}"
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
        # Which image targets to redact, and in what order, comes from the active domain pack's privacy plane
        # (AV: face then plate - byte-identical to the previous hardcoded order). A pack could subset (face
        # only) or extend; an audio target (speech) is enforced on the export/audio path, not here.
        from services.domain import redaction_targets

        detectors: dict[str, FaceDetector | PlateDetector] = {"face": self.face, "plate": self.plate}
        regions: list[dict] = []
        counts: dict[str, int] = {"face": 0, "plate": 0}
        # Vehicle boxes for the text pass, collected from the plate detections themselves: a plate sits on
        # a vehicle, so a plate box is a lower bound on where vehicle text can be. The pass runs after the
        # loop so it can use them.
        plate_boxes: list[tuple[float, float, float, float]] = []
        for target in redaction_targets():
            if target.detector == "text":
                continue  # handled after the loop, where the vehicle prior exists
            det = detectors.get(target.detector)
            if det is None:
                if target.detector == "speech":
                    continue  # audio target: enforced via SpeechSegment + the DPDPA export gate
                raise RuntimeError(
                    f"PiiAnonymizer has no detector for redaction target '{target.name}' "
                    f"(detector={target.detector!r})"
                )
            found = det.detect(image_bgr)
            counts[target.detector] = counts.get(target.detector, 0) + len(found)
            for x1, y1, x2, y2, s in found:
                self._blur_region(image_bgr, x1, y1, x2, y2)
                regions.append({"type": target.name,
                                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                "score": round(s, 3)})
                if target.detector == "plate":
                    plate_boxes.append((x1, y1, x2, y2))

        # Redact-then-verify. Blurring a region and calling the frame clean assumes the blur worked, and a
        # kernel sized for a large plate leaves a small one legible. This looks again at what it blurred.
        released, text_reason, n_text = True, None, 0
        if any(t.detector == "text" for t in redaction_targets()):
            from services.anonymize.text_regions import redact_and_verify

            # Widened plate boxes as the vehicle prior: a plate box is where a plate already was, and the
            # text this pass is for is the plate that detector MISSED, which is nearby rather than on it.
            prior = [(x1 - (x2 - x1), y1 - 2 * (y2 - y1), x2 + (x2 - x1), y2 + (y2 - y1))
                     for x1, y1, x2, y2 in plate_boxes]
            out = redact_and_verify(image_bgr, prior, detector=self._text_detector(),
                                    min_score=self.cfg.text_min_score)
            released, text_reason, n_text = out.released, out.reason, out.n_redacted
            for r in out.detected:
                if r.in_vehicle:
                    regions.append({"type": "text", "bbox": [round(v, 1) for v in r.bbox],
                                    "score": r.score})

        return PiiResult(n_faces=counts.get("face", 0), n_plates=counts.get("plate", 0),
                         regions=regions, method_version=self.method_version,
                         released=released, redaction_reason=text_reason, n_text=n_text)

    def _text_detector(self):
        """Built once. An absent weights file is a refusal at redact time, not a silent pass."""
        if getattr(self, "_text_det", None) is None:
            from services.anonymize.text_regions import TextDetector

            self._text_det = TextDetector(self.cfg.text_weights)
        return self._text_det


@lru_cache(maxsize=1)
def get_anonymizer() -> PiiAnonymizer:
    return PiiAnonymizer(get_settings().pii)
