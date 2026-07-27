"""ANPR-India pipeline: detect plate regions, read each, parse the India format.

Gated on the active pack's 'anpr' capability: the pipeline REFUSES (raises) under a pack that does not
authorise plate reading, so it can never run in the AV / DPDPA context where plates are PII to be blurred.
The plate detector and the OCR reader are injectable; the format layer (services/anpr/india_format.py) is the
pure, India-specific core.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.logging import get_logger
from services.anpr.india_format import PlateParse, parse_plate

log = get_logger("anpr")

Region = tuple[float, float, float, float, float]         # x1, y1, x2, y2, detector_score
OcrReader = Callable[[np.ndarray], tuple[str, float | None]]  # crop -> (text, ocr_confidence|None)


@dataclass(frozen=True)
class PlateRead:
    bbox: tuple[float, float, float, float]
    det_conf: float
    ocr_text: str
    ocr_conf: float | None   # None when the backend reports no calibrated score
    parse: PlateParse


class AnprNotAuthorised(RuntimeError):
    """Raised when ANPR is invoked under a pack that does not declare the 'anpr' capability."""


def _require_authorised(pack_id: str | None) -> None:
    from services.domain import has_capability

    if not has_capability("anpr", pack_id):
        raise AnprNotAuthorised(
            "ANPR (plate reading) is not authorised for the active domain pack. It is a security-domain "
            "capability; a pack must declare 'anpr'. In the AV / DPDPA context plates are PII and are blurred, "
            "never read.")


def _crop(image_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return np.empty((0, 0, 3), dtype=image_bgr.dtype)
    return image_bgr[iy1:iy2, ix1:ix2]


def recognize_plates(
    image_bgr: np.ndarray,
    regions: list[Region],
    ocr: OcrReader | None = None,
    pack_id: str | None = None,
) -> list[PlateRead]:
    """Read and parse each detected plate region. `regions` are plate bounding boxes (from a PlateDetector or
    any localiser); `ocr` defaults to the configured wire seam. Regions below the configured minimum area, or
    whose OCR confidence is below the floor, are dropped. Refuses unless the active pack authorises ANPR."""
    from core.config import get_settings

    _require_authorised(pack_id)
    cfg = get_settings().anpr
    read = ocr or _default_ocr

    h, w = image_bgr.shape[:2]
    frame_area = float(max(1, h * w))
    reads: list[PlateRead] = []
    for x1, y1, x2, y2, score in regions:
        if ((x2 - x1) * (y2 - y1)) / frame_area < cfg.min_plate_area_frac:
            continue
        crop = _crop(image_bgr, (x1, y1, x2, y2))
        if crop.size == 0:
            continue
        text, ocr_conf = read(crop)
        if not text:
            continue
        # An unmeasured read cannot clear a confidence floor. It is kept (the text is still evidence) but
        # carries ocr_conf=None so every downstream sees "unscored" instead of a fabricated number, and a
        # deployment that requires a measured score can reject it explicitly.
        if ocr_conf is not None and ocr_conf < cfg.ocr_min_conf:
            continue
        if ocr_conf is None and cfg.require_measured_ocr_conf:
            continue
        reads.append(PlateRead(bbox=(x1, y1, x2, y2), det_conf=score, ocr_text=text,
                               ocr_conf=ocr_conf, parse=parse_plate(text)))
    log.info("anpr.recognized", regions=len(regions), reads=len(reads),
             valid=sum(1 for r in reads if r.parse.valid))
    return reads


def _default_ocr(crop_bgr: np.ndarray) -> tuple[str, float | None]:
    from services.anpr.ocr import default_plate_ocr

    return default_plate_ocr(crop_bgr)
