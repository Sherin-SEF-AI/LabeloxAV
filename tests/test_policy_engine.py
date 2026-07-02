"""Policy compliance engine: declarative guideline rules over object geometry, class, and attributes."""

from __future__ import annotations

import uuid

from services.autolabel.ontology import get_ontology
from services.errordetect.policy import check_object


class _O:
    def __init__(self, class_id, bbox, attrs=None):
        self.object_id = uuid.uuid4()
        self.class_id = class_id
        self.bbox = bbox
        self.attrs = attrs or {}


def _rules(violations):
    return {r for r, _s, _reason in violations}


def test_min_box_size():
    onto = get_ontology()
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    o = _O(sedan, [100, 100, 110, 116])   # 10x16 px in a 1920x1080 frame -> too small
    assert "min_box_size" in _rules(check_object(o, [], onto, 1920, 1080))


def test_degenerate_aspect_pedestrian_wider_than_tall():
    onto = get_ontology()
    ped = next(c.id for c in onto.classes if c.name == "pedestrian")
    o = _O(ped, [100, 100, 300, 180])     # 200 wide x 80 tall -> aspect 2.5, impossible for a person
    assert "degenerate_aspect" in _rules(check_object(o, [], onto, 1920, 1080))


def test_duplicate_box():
    onto = get_ontology()
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    a = _O(sedan, [400, 400, 600, 600])
    b = _O(sedan, [405, 405, 605, 605])   # near-identical same-class box
    assert "duplicate_box" in _rules(check_object(a, [a, b], onto, 1920, 1080))


def test_invalid_attribute():
    onto = get_ontology()
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    o = _O(sedan, [400, 400, 600, 700], attrs={"occlusion": 999})   # 999 not in the enum
    assert "attr_validity" in _rules(check_object(o, [], onto, 1920, 1080))


def test_clean_object_has_no_violations():
    onto = get_ontology()
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    o = _O(sedan, [800, 500, 1000, 660])   # normal sedan box, no attrs
    assert check_object(o, [o], onto, 1920, 1080) == []
