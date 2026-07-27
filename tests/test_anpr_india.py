"""SEC-M7: ANPR-India.

The India plate-format kernel (pure, thorough) plus the pack-gated recognition pipeline. The gate is the key
compliance property: ANPR refuses under the AV pack (plates are DPDPA PII there), and only runs where a pack
authorises it. Fixtures are procedural (numpy + stub OCR), never a model or a downloaded image.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.anpr.india_format import STATE_CODES, is_valid_state, normalize_plate, parse_plate
from services.anpr.recognize import AnprNotAuthorised, recognize_plates
from services.domain import has_capability

# ---- the India format kernel ----------------------------------------------------------------------------

def test_standard_plate_parses_and_validates():
    p = parse_plate("KA01AB1234")
    assert p.valid and p.plate_type == "standard"
    assert (p.state_code, p.rto_district, p.series, p.number) == ("KA", "01", "AB", "1234")
    assert p.format_confidence == 1.0


def test_normalisation_strips_separators_and_case():
    assert normalize_plate("ka 01-ab.1234") == "KA01AB1234"
    p = parse_plate("mh 12 de 1433")
    assert p.valid and p.state_code == "MH" and p.number == "1433"


def test_bh_series():
    p = parse_plate("22 BH 1234 AA")
    assert p.valid and p.plate_type == "bh_series"
    assert p.state_code == "BH" and p.rto_district == "22" and p.series == "AA" and p.number == "1234"


def test_diplomatic():
    p = parse_plate("21 CD 45")
    assert p.valid and p.plate_type == "diplomatic" and p.state_code == "CD" and p.number == "45"


def test_unknown_state_code_is_well_formed_but_not_valid():
    p = parse_plate("XZ01AB1234")   # XZ is not a real RTO code
    assert p.plate_type == "standard" and not p.valid
    assert p.format_confidence == 0.5 and p.state_code == "XZ"


def test_garbage_is_invalid():
    p = parse_plate("!!!")
    assert not p.valid and p.plate_type == "invalid" and p.format_confidence == 0.0


def test_state_code_table_is_sane():
    assert is_valid_state("KA") and is_valid_state("DL") and is_valid_state("TS")
    assert not is_valid_state("ZZ") and not is_valid_state(None)
    assert {"KA", "MH", "TN", "UP", "WB", "DL"} <= STATE_CODES


# ---- the pack-gated pipeline ----------------------------------------------------------------------------

def _img():
    return np.full((200, 400, 3), 180, dtype=np.uint8)


def _region(x1=50.0, y1=50.0, x2=200.0, y2=110.0, score=0.9):
    return (x1, y1, x2, y2, score)


def test_pipeline_reads_and_parses_under_an_authorising_pack():
    reads = recognize_plates(_img(), [_region()], ocr=lambda c: ("KA01AB1234", 0.95), pack_id="sec")
    assert len(reads) == 1
    r = reads[0]
    assert r.ocr_text == "KA01AB1234" and r.parse.valid and r.parse.state_code == "KA"


def test_pipeline_refuses_under_the_av_pack():
    assert has_capability("anpr", "sec") and not has_capability("anpr", "av")
    with pytest.raises(AnprNotAuthorised):
        recognize_plates(_img(), [_region()], ocr=lambda c: ("KA01AB1234", 0.95), pack_id="av")


def test_low_ocr_confidence_is_dropped():
    reads = recognize_plates(_img(), [_region()], ocr=lambda c: ("KA01AB1234", 0.10), pack_id="sec")
    assert reads == []


def test_tiny_region_below_min_area_is_dropped():
    tiny = (10.0, 10.0, 12.0, 12.0, 0.9)   # 4 px in an 80k-px frame, below min_plate_area_frac
    reads = recognize_plates(_img(), [tiny], ocr=lambda c: ("KA01AB1234", 0.95), pack_id="sec")
    assert reads == []


def test_empty_text_is_dropped():
    reads = recognize_plates(_img(), [_region()], ocr=lambda c: ("", 0.9), pack_id="sec")
    assert reads == []
