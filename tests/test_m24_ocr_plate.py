"""M2.4 hard compliance rule: OCR must NEVER read or store license-plate text. This verifies the
plate-exclusion gate: a text region overlapping a plate bbox (from PiiAudit) is excluded, plate geometry
is taken only from plate regions (not faces), and a sign far from any plate is allowed."""

from __future__ import annotations

from services.autolabel.ocr.reader import is_plate_excluded, plate_bboxes


def test_ocr_never_reads_license_plates():
    regions = [
        {"type": "plate", "bbox": [100, 100, 200, 140], "score": 0.9},
        {"type": "face", "bbox": [300, 80, 340, 120], "score": 0.8},  # faces are not plate geometry
    ]
    plates = plate_bboxes(regions)
    assert plates == [[100, 100, 200, 140]]  # only plates surface as exclusion geometry

    # a text region sitting on the plate is EXCLUDED (never OCR'd)
    assert is_plate_excluded([105, 102, 205, 142], plates, 0.2) is True
    # a sign far from any plate is allowed
    assert is_plate_excluded([500, 400, 560, 460], plates, 0.2) is False
    # no plates recorded -> nothing excluded on that ground
    assert is_plate_excluded([105, 102, 205, 142], [], 0.2) is False
