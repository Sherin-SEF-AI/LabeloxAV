"""M8 acceptance: the dual gate blocks a quantization that speeds up but drops a protected slice, and blocks a
latency regression; a genuinely better artifact passes. Plus capability gating and Pareto ranking."""

import pytest

from services.forgyx.capabilities import CapabilityError, available_targets, require
from services.forgyx.gate import dual_gate, latency_regressed, pareto_rank

PROTECTED = ["pedestrian_night"]


def _ev(agg, ped):
    return {"aggregate": {"map50": agg}, "per_slice": {"pedestrian_night": {"map": ped}, "sedan_day": {"map": 0.7}}}


def test_faster_but_drops_protected_slice_is_blocked():
    champ = _ev(0.50, 0.60)
    cand = _ev(0.52, 0.40)                    # a bit better aggregate, faster, but pedestrian-night collapses
    r = dual_gate({"p95": 20.0}, {"p95": 12.0}, champ, cand, PROTECTED)
    assert r["promote"] is False
    assert any("pedestrian_night" in reason for reason in r["reasons"])
    assert r["latency"]["faster"] is True     # it was faster, and still blocked


def test_latency_regression_is_blocked():
    champ = _ev(0.50, 0.60)
    cand = _ev(0.55, 0.62)                    # more accurate, no slice regression
    r = dual_gate({"p95": 20.0}, {"p95": 30.0}, champ, cand, PROTECTED)  # but 50% slower
    assert r["promote"] is False
    assert r["latency"]["regressed"] is True


def test_genuinely_better_passes():
    champ = _ev(0.50, 0.60)
    cand = _ev(0.55, 0.63)
    r = dual_gate({"p95": 20.0}, {"p95": 15.0}, champ, cand, PROTECTED)
    assert r["promote"] is True


def test_latency_regressed_tolerance():
    assert latency_regressed({"p95": 10.0}, {"p95": 10.5})["regressed"] is False   # 5% within 10% tol
    assert latency_regressed({"p95": 10.0}, {"p95": 12.0})["regressed"] is True    # 20% over tol


def test_capability_require_raises_when_absent():
    if not available_targets().get("agx_orin_trt"):
        with pytest.raises(CapabilityError):
            require("agx_orin_trt")
    with pytest.raises(CapabilityError):
        require("nonexistent_target")


def test_pareto_rank_front():
    b = [
        {"target": "a", "latency_ms": {"p95": 10}, "map50": 0.8},   # front
        {"target": "b", "latency_ms": {"p95": 20}, "map50": 0.7},   # dominated by a
        {"target": "c", "latency_ms": {"p95": 30}, "map50": 0.9},   # front (most accurate)
    ]
    ranked = {r["target"]: r["pareto_rank"] for r in pareto_rank(b)}
    assert ranked["a"] == 0 and ranked["c"] == 0 and ranked["b"] >= 1
