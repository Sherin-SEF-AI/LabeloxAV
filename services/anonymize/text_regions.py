"""Finding text to destroy it, and checking afterwards that it is gone.

Both OCR paths in this tree take their regions from existing annotations, so there is no text-region
detector anywhere in it. That is a gap on the privacy plane rather than on the reading one: a licence
plate the plate detector missed is still a plate, and it is still legible, and the release attestation
says the frame was redacted.

This is a DETECTOR FOR REDACTION, NOT FOR READING. It returns regions and never characters, it never
consults services/anpr/ (which refuses by capability gate and must keep refusing), and nothing here
stores what a plate said. The distinction is not stylistic: reading a plate creates a record that has to
be governed, and blurring one destroys the need for any record at all.

REDACT THEN VERIFY, which is the part that makes this worth building. Blurring a region and calling the
frame clean assumes the blur worked. A Gaussian kernel sized for a large plate leaves a small one legible,
and the failure is silent because nothing looks again. So:

    1. detect text regions
    2. blur them
    3. RE-DETECT on the blurred output
    4. anything still detected is escalated once, with a stronger kernel
    5. anything still detected after that marks the frame `redaction_failed`, and the frame is blocked
       from storage and serving

Step 5 is the point. A frame that could not be redacted is not a frame with a caveat, it is a frame that
must not leave the building, and the honest outcome is refusing to serve it rather than shipping it with a
flag nobody reads.

CONSTRAINED BY A VEHICLE PRIOR. Text is everywhere in an Indian street scene: shop signs, hoardings, bus
destination boards, political posters. Redacting all of it would destroy most of the frame's value and
none of it is personal data. So a text region counts only when it sits within a vehicle detection, which
is where a plate is, and the prior is passed in rather than assumed so a caller with no vehicle boxes gets
a documented refusal instead of a whole-frame blur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.logging import get_logger

log = get_logger("text_regions")

# How much of a text region must sit inside a vehicle box for it to be plate-like. Not 1.0: a plate at the
# edge of a tightly-drawn vehicle box overhangs it slightly.
VEHICLE_CONTAINMENT = 0.6
# The escalation ladder. The first is the ordinary kernel; the second is what a region that survived it
# gets. There is no third: a region that survives both is a failure to record, not a kernel to grow.
_KERNELS = (31, 81)


@dataclass(frozen=True)
class TextRegion:
    """One detected text region. No characters, ever, by construction rather than by policy."""

    bbox: tuple[float, float, float, float]
    score: float
    in_vehicle: bool


@dataclass(frozen=True)
class RedactionOutcome:
    """What happened to a frame: what was found, what survived, and whether it may be released.

    `released` False means the frame must not be stored or served. That is deliberately not a warning:
    a frame whose text could not be destroyed is a frame that leaks, and the only safe handling is refusal.
    """

    detected: tuple[TextRegion, ...]
    n_redacted: int
    n_escalated: int
    n_surviving: int
    released: bool
    detector_available: bool
    reason: str | None = None


class TextDetector:
    """OpenCV's DB text detector, loaded from a weights file fetched by `make pii-models`.

    Absent weights are a refusal, not a silent pass. A privacy detector that quietly does nothing is worse
    than one that is not installed, because the pipeline around it reports success either way.
    """

    def __init__(self, weights: str, *, input_size: tuple[int, int] = (736, 736),
                 binary_thr: float = 0.3, poly_thr: float = 0.5):
        self.weights = weights
        self.input_size = input_size
        self._net: Any = None
        self._error: str | None = None
        if not weights or not Path(weights).exists():
            self._error = (f"text detector weights not found at {weights or '<unset>'}; "
                           "run `make pii-models` to fetch them")
            return
        try:
            import cv2

            self._net = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(weights))
            self._net.setBinaryThreshold(binary_thr).setPolygonThreshold(poly_thr)
            self._net.setInputParams(1.0 / 255.0, input_size, (122.67891434, 116.66876762, 104.00698793))
        except Exception as exc:  # noqa: BLE001 - a broken detector must not look like a clean frame
            self._net = None
            self._error = f"text detector failed to load: {exc}"

    @property
    def available(self) -> bool:
        return self._net is not None

    @property
    def error(self) -> str | None:
        return self._error

    def detect(self, image_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Axis-aligned boxes around text, with confidence. Never characters."""
        if self._net is None:
            return []
        boxes, confs = self._net.detect(image_bgr)
        out = []
        for quad, c in zip(boxes, confs, strict=False):
            q = np.asarray(quad, dtype=float).reshape(-1, 2)
            out.append((float(q[:, 0].min()), float(q[:, 1].min()),
                        float(q[:, 0].max()), float(q[:, 1].max()), float(c)))
        return out


def _containment(inner: tuple[float, float, float, float],
                 outer: tuple[float, float, float, float]) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)
    return float(inter / area)


def _blur(image_bgr: np.ndarray, box: tuple[float, float, float, float], kernel: int) -> None:
    import cv2

    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    if x2 <= x1 or y2 <= y1:
        return
    k = kernel if kernel % 2 == 1 else kernel + 1
    image_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(image_bgr[y1:y2, x1:x2], (k, k), 0)


def redact_and_verify(image_bgr: np.ndarray, vehicle_boxes: list[tuple[float, float, float, float]], *,
                      detector: TextDetector, containment: float = VEHICLE_CONTAINMENT,
                      min_score: float = 0.3) -> RedactionOutcome:
    """Blur plate-like text, look again, escalate once, and refuse the frame if anything survives.

    `vehicle_boxes` is the prior. Without it every shop sign and hoarding in an Indian street scene would
    be redacted, which destroys the frame's value and protects nobody, so an empty list is a documented
    refusal rather than a whole-frame blur.
    """
    if not detector.available:
        return RedactionOutcome((), 0, 0, 0, released=False, detector_available=False,
                                reason=detector.error or "no text detector available")
    if not vehicle_boxes:
        return RedactionOutcome((), 0, 0, 0, released=True, detector_available=True,
                                reason="no vehicle detections on this frame, so no plate-like text region "
                                       "is defined; nothing was redacted and nothing was claimed to be")

    found = [r for r in detector.detect(image_bgr) if r[4] >= min_score]
    regions = []
    for x1, y1, x2, y2, s in found:
        box = (x1, y1, x2, y2)
        inside = any(_containment(box, v) >= containment for v in vehicle_boxes)
        regions.append(TextRegion(box, round(s, 4), inside))

    targets = [r for r in regions if r.in_vehicle]
    if not targets:
        return RedactionOutcome(tuple(regions), 0, 0, 0, released=True, detector_available=True)

    for r in targets:
        _blur(image_bgr, r.bbox, _KERNELS[0])
    n_redacted = len(targets)

    # Look again. This is the whole reason the module exists: a kernel sized for a large plate leaves a
    # small one legible, and nothing in the previous design ever checked.
    survivors = [r for r in detector.detect(image_bgr) if r[4] >= min_score
                 and any(_containment((r[0], r[1], r[2], r[3]), v) >= containment for v in vehicle_boxes)]
    n_escalated = 0
    if survivors:
        for r in survivors:
            _blur(image_bgr, (r[0], r[1], r[2], r[3]), _KERNELS[1])
        n_escalated = len(survivors)
        survivors = [r for r in detector.detect(image_bgr) if r[4] >= min_score
                     and any(_containment((r[0], r[1], r[2], r[3]), v) >= containment
                             for v in vehicle_boxes)]

    n_surviving = len(survivors)
    released = n_surviving == 0
    if not released:
        log.error("text_regions.redaction_failed", surviving=n_surviving, redacted=n_redacted,
                  escalated=n_escalated,
                  detail="text is still detectable after escalation; the frame must not be stored or served")
    return RedactionOutcome(tuple(regions), n_redacted, n_escalated, n_surviving,
                            released=released, detector_available=True,
                            reason=(None if released else
                                    f"{n_surviving} text region(s) are still detectable after two blur "
                                    "passes; this frame is blocked from storage and serving"))


__all__ = ["TextDetector", "TextRegion", "RedactionOutcome", "redact_and_verify",
           "VEHICLE_CONTAINMENT"]
