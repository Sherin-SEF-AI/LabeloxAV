"""FORGYX M16 tests: model-target co-optimization planning against a latency budget, thermal/power envelope from
a device-farm run (refused without readings), signed deployment packaging with tamper detection, and the
rollout/rollback state machine."""

import pytest

from services.forgyx.cooptimize import plan_cooptimization
from services.forgyx.packaging import build_manifest, plan_rollout, sign_manifest, verify_manifest
from services.forgyx.thermal import thermal_envelope


def test_cooptimization_meets_budget():
    # base fp16 at 50 ms cannot meet the 15 ms AGX Orin budget without pruning + int8
    plan = plan_cooptimization("agx_orin_trt", base_latency_ms=50.0, base_map50=0.6)
    assert plan["feasible"] is True
    assert plan["chosen"]["est_latency_ms"] <= plan["budget_ms"]
    # among feasible configs the chosen one keeps the most accuracy
    feas = [c for c in plan["ranked"] if c["est_latency_ms"] <= plan["budget_ms"]]
    assert plan["chosen"]["est_map50"] == max(c["est_map50"] for c in feas)


def test_thermal_envelope_requires_real_readings():
    good = thermal_envelope("orin_nano_trt", {"peak_temp_c": 78.0, "power_w": 12.0, "throttled_fps": 29.8},
                            cold_fps=30.0)
    assert good["passed"] is True and good["headroom_c"] > 0
    # a run that throttles fails
    hot = thermal_envelope("orin_nano_trt", {"peak_temp_c": 88.0, "power_w": 12.0, "throttled_fps": 20.0},
                           cold_fps=30.0)
    assert hot["passed"] is False and hot["throttled"] is True
    # no device readings -> refused, not assumed passing
    with pytest.raises(ValueError):
        thermal_envelope("orin_nano_trt", {}, cold_fps=30.0)


def test_signed_packaging_detects_tamper():
    key = "unit-test-key"
    m = build_manifest("d1", "yolo11n-v7", "agx_orin_trt", "tensorrt", "s3://a.engine",
                       release_commit="c1", verdict_ref="e1", benchmark_ref="b1",
                       thermal_envelope={"sustained_fps": 62.0})
    sig = sign_manifest(m, key)
    assert verify_manifest(m, sig, key) is True
    # a swapped artifact invalidates the signature
    tampered = {**m, "artifact_uri": "s3://evil.engine"}
    assert verify_manifest(tampered, sig, key) is False
    # a wrong key cannot verify
    assert verify_manifest(m, sig, "other-key") is False


def test_rollout_state_machine():
    assert plan_rollout("none", "canary")["allowed"] is True
    assert plan_rollout("none", "full")["allowed"] is False       # cannot skip canary to full fleet
    assert plan_rollout("canary", "full")["new_state"] == "full"
    assert plan_rollout("full", "rolled_back")["allowed"] is True
    assert plan_rollout("rolled_back", "canary")["allowed"] is False
