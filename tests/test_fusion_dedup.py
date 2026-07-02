"""Post-fusion de-duplication: one physical object becomes one box, but distinct overlapping objects
(a rider on a two-wheeler) are both kept.
"""

from __future__ import annotations

from types import SimpleNamespace

from services.autolabel.fusion import FusedObject, FusionEngine
from services.autolabel.ontology import get_ontology


def _fo(class_name, bbox, conf):
    onto = get_ontology()
    cid = onto.by_name(class_name).id
    obj = SimpleNamespace(class_id=cid, class_name=class_name, conf=conf,
                          bbox=SimpleNamespace(as_list=lambda b=bbox: list(b)),
                          provenance=SimpleNamespace())
    return FusedObject(obj=obj, mask=None)


def test_nested_same_object_collapses_to_one():
    eng = FusionEngine()
    # A white car boxed at three overlapping scales, plus a specific/fallback pair on the same pixels.
    objs = [
        _fo("sedan", (100, 100, 300, 260), 0.90),
        _fo("sedan", (95, 96, 305, 268), 0.70),           # near-duplicate, lower conf
        _fo("sedan", (140, 140, 250, 230), 0.60),          # nested inside the big one
        _fo("vehicle_fallback", (98, 99, 302, 262), 0.55), # fallback over the same car
    ]
    kept = eng._suppress_duplicates(objs)
    assert len(kept) == 1
    assert kept[0].obj.conf == 0.90                        # the highest-confidence box wins
    assert len(kept[0].obj.provenance.merged_duplicates) == 3


def test_rider_on_two_wheeler_kept_separate():
    eng = FusionEngine()
    # A rider box overlapping the motorcycle it sits on: different l1 superclasses, must both survive.
    objs = [
        _fo("motorcycle", (100, 200, 180, 340), 0.88),
        _fo("rider", (110, 150, 170, 300), 0.80),
    ]
    kept = eng._suppress_duplicates(objs)
    assert len(kept) == 2


def test_disjoint_vehicles_both_kept():
    eng = FusionEngine()
    objs = [
        _fo("sedan", (100, 100, 200, 200), 0.9),
        _fo("sedan", (400, 100, 520, 210), 0.9),   # a different car elsewhere in the frame
    ]
    assert len(eng._suppress_duplicates(objs)) == 2
