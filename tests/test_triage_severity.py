"""Triage flags carry a severity, and the queue can be read by it.

The queue previously emitted its reasons as one comma joined string, which the web client re-parsed with
regexes to pick a colour. That put the taxonomy in two places and lost the thing an annotator most needs:
a geometry disagreement and a rare class are not equally urgent, and rendering them identically means the
eye cannot sort a screen of them.

The reasons are structured at the point they are decided, where the evidence already exists, so severity is
computed once rather than recovered from prose downstream.
"""

from __future__ import annotations

import uuid

import pytest

from services.api.routers.triage import _why_and_priority
from services.autolabel.ontology import get_ontology


class _Obj:
    """The fields _why_and_priority reads, without needing a database row to exist."""

    def __init__(self, class_id: int, conf: float, provenance: dict | None = None):
        self.object_id = uuid.uuid4()
        self.class_id = class_id
        self.conf = conf
        self.provenance = provenance or {}


def _class_ids():
    onto = get_ontology()
    rare = next(c for c in onto.classes if c.india or c.l1 == "fallback")
    common = next(c for c in onto.classes if not (c.india or c.l1 == "fallback"))
    return onto, common.id, rare.id


def test_flags_are_structured_not_only_prose():
    onto, common, _ = _class_ids()
    obj = _Obj(common, 0.3, {"mask_box_disagree": True})
    why, _priority, flags = _why_and_priority(obj, onto)

    assert isinstance(flags, list) and flags, "the queue has to emit flags, not only a joined string"
    codes = {f["code"] for f in flags}
    assert "mask_box" in codes
    assert every_flag_is_complete(flags)
    # The prose form stays, because existing clients and the export path read it.
    assert "mask" in why


def every_flag_is_complete(flags) -> bool:
    return all(
        set(f) >= {"code", "label", "severity"} and f["severity"] in ("high", "medium", "low")
        for f in flags
    )


def test_geometry_and_conflict_outrank_rarity_and_low_confidence():
    """The ordering an annotator scans by.

    A box whose outline disagrees with it, or that two models named differently, is a defect. A rare class
    or a soft score is context: worth knowing, not worth the same alarm.
    """
    onto, _common, rare = _class_ids()
    obj = _Obj(rare, 0.2, {
        "mask_box_disagree": True,
        "proposals": [{"verdict": "overruled"}, {"verdict": "kept"}],
    })
    _why, _priority, flags = _why_and_priority(obj, onto)

    by_code = {f["code"]: f["severity"] for f in flags}
    assert by_code["mask_box"] == "high"
    assert by_code["class_conflict"] == "high"
    assert by_code["rare_class"] == "low"
    assert by_code["low_conf"] == "medium"

    # Emitted worst first, so a client renders the ordering without re-sorting it.
    rank = {"high": 0, "medium": 1, "low": 2}
    severities = [rank[f["severity"]] for f in flags]
    assert severities == sorted(severities), f"flags must arrive worst first, got {severities}"


def test_labels_are_readable_rather_than_schema():
    """`mask != box` is a column comparison. The reader's question is whether the outline fits the box."""
    onto, common, _ = _class_ids()
    obj = _Obj(common, 0.9, {"mask_box_disagree": True})
    _why, _priority, flags = _why_and_priority(obj, onto)

    label = next(f["label"] for f in flags if f["code"] == "mask_box")
    assert "!=" not in label and "mask" not in label.lower()
    assert label == "Outline off box"


def test_a_clean_object_still_says_why_it_is_here():
    """An object in the review band with nothing wrong is not an error, and must not render as one."""
    onto, common, _ = _class_ids()
    obj = _Obj(common, 0.95)
    why, _priority, flags = _why_and_priority(obj, onto)

    assert why == "review band"
    assert [f["severity"] for f in flags] == ["low"]
    assert flags[0]["code"] == "review_band"


@pytest.mark.parametrize("conf,expected", [(0.59, True), (0.6, False), (0.95, False)])
def test_low_confidence_boundary_is_not_off_by_one(conf, expected):
    """0.6 is the documented floor and is not itself low, which a `<=` would silently change."""
    onto, common, _ = _class_ids()
    _why, _priority, flags = _why_and_priority(_Obj(common, conf), onto)
    assert ("low_conf" in {f["code"] for f in flags}) is expected


def test_priority_is_unchanged_by_the_refactor():
    """Ranking is what the queue is for, so restructuring the reasons must not move it."""
    onto, _common, rare = _class_ids()
    obj = _Obj(rare, 0.2, {
        "mask_box_disagree": True,
        "proposals": [{"verdict": "overruled"}, {"verdict": "kept"}],
    })
    _why, priority, _flags = _why_and_priority(obj, onto)
    # (1 - 0.2) * 2.0 rare * (1 + 0.5 mask_box + 0.5 conflict)
    assert priority == pytest.approx(3.2)
