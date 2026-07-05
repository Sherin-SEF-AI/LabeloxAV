"""Platform registry invariants: the seven platforms, their gates, and route resolution."""

from platforms.registry import PLATFORMS, as_dicts, gates, platform_by_id, platform_for_route

EXPECTED = {"labelox", "sanyx", "calyx", "sievyx", "oraclyx", "verdyx", "forgyx"}


def test_seven_platforms_unique_ids():
    ids = [p.id for p in PLATFORMS]
    assert set(ids) == EXPECTED
    assert len(ids) == len(set(ids)) == 7


def test_gates_are_the_blocking_planes_in_order():
    assert [p.id for p in gates()] == ["sanyx", "calyx", "verdyx", "forgyx"]


def test_route_resolves_by_canonical_and_legacy_prefix():
    assert platform_for_route("/sanyx/quarantine").id == "sanyx"
    assert platform_for_route("/calibration/123").id == "calyx"   # legacy prefix
    assert platform_for_route("/training/runs").id == "verdyx"    # legacy prefix
    assert platform_for_route("/nope") is None


def test_as_dicts_ordered_and_complete():
    ds = as_dicts()
    assert [d["order"] for d in ds] == sorted(d["order"] for d in ds)
    for d in ds:
        assert {"id", "label", "role", "backend_prefix", "legacy_prefixes", "gate", "flywheel_stage"} <= set(d)


def test_by_id_lookup():
    assert platform_by_id("forgyx").gate == "benchmark"
    assert platform_by_id("missing") is None
