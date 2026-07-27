"""OCR and ANPR report unmeasured confidence as None, never as a fabricated constant.

The defect: the local generative-VLM wire returned `0.8 if text else 0.0` for every read. A constant on one
side of a threshold makes the threshold a no-op that merely looks like a quality gate, and the fake number
then rode into Object.ocr_conf and the ANPR event payload as if it had been measured.

Pure unit tests: the readers are patched, so no VLM, GPU, or database is involved."""
from __future__ import annotations

import numpy as np

from core.config import get_settings
from services.anpr.recognize import PlateRead, recognize_plates
from services.anpr.india_format import parse_plate


def _img(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_unscored_read_is_kept_with_none_confidence():
    # A read the backend cannot score is still evidence: it survives, but carries None so downstream sees
    # "unscored" rather than a number it might trust.
    reads = recognize_plates(
        _img(), [(10.0, 10.0, 210.0, 110.0, 0.9)], pack_id="sec",
        ocr=lambda crop: ("KA01AB1234", None),
    )
    assert len(reads) == 1
    assert reads[0].ocr_conf is None
    assert reads[0].ocr_text == "KA01AB1234"


def test_measured_read_below_the_floor_is_still_dropped():
    # The floor must keep working for a backend that does report a calibrated score.
    floor = get_settings().anpr.ocr_min_conf
    reads = recognize_plates(
        _img(), [(10.0, 10.0, 210.0, 110.0, 0.9)], pack_id="sec",
        ocr=lambda crop: ("KA01AB1234", floor - 0.1),
    )
    assert reads == []


def test_measured_read_above_the_floor_is_kept():
    floor = get_settings().anpr.ocr_min_conf
    reads = recognize_plates(
        _img(), [(10.0, 10.0, 210.0, 110.0, 0.9)], pack_id="sec",
        ocr=lambda crop: ("KA01AB1234", min(1.0, floor + 0.1)),
    )
    assert len(reads) == 1 and reads[0].ocr_conf is not None


def test_empty_text_is_dropped_regardless_of_confidence():
    reads = recognize_plates(
        _img(), [(10.0, 10.0, 210.0, 110.0, 0.9)], pack_id="sec",
        ocr=lambda crop: ("", None),
    )
    assert reads == []


def test_strict_deployment_can_require_a_measured_score(monkeypatch):
    # A deployment that must gate on a real score flips the flag and unscored reads stop being accepted.
    s = get_settings()
    prev = s.anpr.require_measured_ocr_conf
    s.anpr.require_measured_ocr_conf = True
    try:
        reads = recognize_plates(
            _img(), [(10.0, 10.0, 210.0, 110.0, 0.9)], pack_id="sec",
            ocr=lambda crop: ("KA01AB1234", None),
        )
        assert reads == []
    finally:
        s.anpr.require_measured_ocr_conf = prev


def test_event_payload_carries_null_not_a_number_for_unscored_reads():
    from services.anpr.events import _read_payload

    read = PlateRead(bbox=(1.0, 2.0, 3.0, 4.0), det_conf=0.9, ocr_text="KA01AB1234",
                     ocr_conf=None, parse=parse_plate("KA01AB1234"))
    payload = _read_payload(read, camera_id="cam-1", session_id=None, pack_id="sec")
    assert payload["ocr_conf"] is None
