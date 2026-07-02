"""Cleanup sweep logic: which existing objects get removed (stuff/oversize/duplicate) and that a removed
object round-trips through snapshot -> restore unchanged."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from services.agent.cleanup_sweep import _dup_removals, _reason, _restore, _snapshot
from services.autolabel.ontology import get_ontology


def _o(name, bbox, conf=0.8, oid=None):
    onto = get_ontology()
    return SimpleNamespace(object_id=oid or uuid.uuid4(), class_id=onto.by_name(name).id, bbox=list(bbox), conf=conf)


def test_reason_flags_stuff_and_oversize():
    onto = get_ontology()
    assert _reason(_o("tree", [0, 0, 50, 50]), onto, 1000, 1000, None, 0.85) == "stuff"
    assert _reason(_o("sedan", [10, 10, 960, 960]), onto, 1000, 1000, None, 0.85) == "oversize"
    assert _reason(_o("sedan", [100, 100, 300, 300]), onto, 1000, 1000, None, 0.85) is None


def test_ego_reason_when_mask_present():
    from services.autolabel.ego_mask import EgoMask

    onto = get_ontology()
    grid = tuple(tuple(1 if gy >= 40 else 0 for gx in range(64)) for gy in range(48))  # bottom band = hood
    ego = EgoMask(grid=grid, area_frac=0.16)
    assert _reason(_o("sedan", [400, 900, 900, 990]), onto, 1000, 1000, ego, 0.85) == "ego_hood"
    assert _reason(_o("sedan", [400, 100, 900, 300]), onto, 1000, 1000, ego, 0.85) is None


def test_dedup_drops_nested_same_class_keeps_rider():
    onto = get_ontology()
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    objs = [_o("sedan", [100, 100, 300, 260], 0.9, a), _o("sedan", [96, 98, 304, 264], 0.7, b),
            _o("sedan", [140, 140, 250, 230], 0.6, c)]
    drop = _dup_removals(objs, onto)
    assert drop == {b, c}                                   # only the highest-conf sedan survives

    bike, rider = uuid.uuid4(), uuid.uuid4()
    objs2 = [_o("motorcycle", [100, 200, 180, 340], 0.88, bike), _o("rider", [110, 150, 170, 300], 0.8, rider)]
    assert _dup_removals(objs2, onto) == set()              # different superclass -> both kept


def test_snapshot_restore_roundtrip():
    from db.models import Object

    onto = get_ontology()
    oid, fid = uuid.uuid4(), uuid.uuid4()
    obj = Object(object_id=oid, frame_id=fid, class_id=onto.by_name("tree").id, bbox=[1.0, 2.0, 3.0, 4.0],
                 conf=0.5, source="fused", state="annotate", attrs={"a": 1}, provenance={"x": 1}, version=1)
    snap = _snapshot(obj)
    assert snap["object_id"] == str(oid) and snap["class_id"] == obj.class_id
    restored = _restore({k: v for k, v in snap.items() if not k.startswith("_")})
    assert restored.object_id == oid and restored.frame_id == fid
    assert restored.bbox == [1.0, 2.0, 3.0, 4.0] and restored.attrs == {"a": 1}
