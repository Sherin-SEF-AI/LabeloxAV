"""M0 unit tests: config, ontology, timebase, schemas. No infra required."""

from __future__ import annotations

import os

from core.schemas import BBox, GateState, UnifiedObject
from core.timebase import datetime_to_ns, frame_ts_ns, ns_to_datetime, now_ns
from services.autolabel.ontology import load_ontology


def test_config_loads_yaml_defaults():
    from core.config import Settings

    s = Settings()
    assert s.gpu.mode in ("sequential", "concurrent")
    assert s.gate.auto_accept == 0.95
    assert s.postgres.async_dsn.startswith("postgresql+asyncpg://")


def test_config_env_override(monkeypatch):
    from core.config import Settings

    monkeypatch.setenv("LBX_GPU__MODE", "concurrent")
    monkeypatch.setenv("LBX_GATE__AUTO_ACCEPT", "0.9")
    s = Settings()
    assert s.gpu.mode == "concurrent"
    assert s.gate.auto_accept == 0.9


def test_ontology_loads_governed_classes():
    # The ontology is additive + extensible (P1 added 166 governed classes; annotators can add more
    # customs), so assert the governed floor and that key classes resolve rather than a frozen count.
    onto = load_ontology()
    assert onto.version == "labelox-in-0.1.0"
    assert len(onto.classes) >= 166
    assert onto.hierarchy_levels == 4
    # core + P1 additions resolve
    for name in ("motorcycle", "autorickshaw", "drain", "lane_arrow", "sky", "pedestrian_signal", "bmtc_bus_shelter"):
        assert onto.has_name(name), f"missing class {name}"
    # ids/names are unique even after the additions
    assert len({c.id for c in onto.classes}) == len(onto.classes)
    assert len({c.name for c in onto.classes}) == len(onto.classes)


def test_ontology_has_moat_and_fallback_classes():
    onto = load_ontology()
    assert onto.has_name("autorickshaw")
    assert onto.has_name("water_tanker")
    assert onto.has_name("cattle")
    fb = onto.fallback_ids()
    assert onto.by_name("vehicle_fallback").id in fb
    assert onto.by_name("object_fallback").id in fb
    assert onto.is_fallback(onto.by_name("object_fallback").id)


def test_ontology_concept_phrases_put_india_first():
    onto = load_ontology()
    phrases = onto.concept_phrases(india_first=True)
    # An India-specific class should appear before a generic one.
    assert phrases.index("autorickshaw") < phrases.index("sedan")


def test_ontology_attr_validation():
    onto = load_ontology()
    assert onto.validate_attrs({"occlusion": 25, "overload": True}) == []
    errs = onto.validate_attrs({"occlusion": 33, "direction": "sideways", "bogus": 1})
    assert len(errs) == 3


def test_timebase_roundtrip_is_integer_ns():
    ts = now_ns()
    assert isinstance(ts, int)
    dt = ns_to_datetime(ts)
    assert abs(datetime_to_ns(dt) - ts) < 1_000_000  # within 1 ms
    assert frame_ts_ns(1000, 6, 3.0) == 1000 + 2_000_000_000


def test_unified_object_shape():
    obj = UnifiedObject(
        frame_id="00000000-0000-0000-0000-000000000001",
        class_id=6,
        class_name="autorickshaw",
        bbox=BBox(x1=10, y1=20, x2=110, y2=220),
        conf=0.81,
    )
    assert obj.state == GateState.review
    assert obj.bbox.area == 100 * 200
    assert obj.bbox.centroid == (60.0, 120.0)
