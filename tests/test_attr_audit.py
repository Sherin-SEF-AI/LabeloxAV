"""Milestone I: attribute validation audit. Scanning (class_id, attrs) records against the real ontology
flags an unknown attribute, leaves clean objects alone, and ranks the worst offenders first."""

from __future__ import annotations

from services.autolabel.ontology import get_ontology
from services.quality.attr_audit import audit_attrs

_ONTO = get_ontology()
_CID = _ONTO.classes[0].id


def test_unknown_attribute_is_flagged():
    recs = [{"object_id": "a", "class_id": _CID, "attrs": {"__not_a_real_attr__": 1}}]
    v = audit_attrs(recs, _ONTO)
    assert len(v) == 1 and v[0]["object_id"] == "a"
    assert any("unknown attribute" in e for e in v[0]["errors"])


def test_clean_object_has_no_violation():
    assert audit_attrs([{"object_id": "b", "class_id": _CID, "attrs": {}}], _ONTO) == []


def test_worst_offender_ranks_first():
    recs = [
        {"object_id": "one_error", "class_id": _CID, "attrs": {"__bad1__": 1}},
        {"object_id": "two_errors", "class_id": _CID, "attrs": {"__bad1__": 1, "__bad2__": 2}},
    ]
    v = audit_attrs(recs, _ONTO)
    assert [x["object_id"] for x in v] == ["two_errors", "one_error"]
