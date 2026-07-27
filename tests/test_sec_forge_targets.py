"""SEC-M9: CCTV forge targets for the Sec pack.

The Sec pack declares real security-camera / NVR edge targets, and FORGYX's per-target thermal and latency
checks resolve a target's profile from the pack (falling back to the legacy AV registry). AV is byte-identical
(the AV pack's forge targets mirror the registry); Sec targets, previously "unknown", now work end to end.
FORGYX's gate/packaging stay generic (opaque target strings).
"""

from __future__ import annotations

from packs.base import ForgeTarget
from packs.registry import get_pack
from services import domain
from services.forgyx.cooptimize import TARGET_BUDGET_MS, _budget_for, plan_cooptimization
from services.forgyx.thermal import TARGET_THERMAL, _thermal_for, thermal_envelope


def test_sec_pack_declares_cctv_forge_targets():
    targets = get_pack("sec").forge_targets
    assert len(targets) == 5
    names = {t.name for t in targets}
    assert {"ambarella_cv5", "axis_artpec8", "hailo8_m2", "openvino_myriadx", "x86_onnx_nvr"} == names
    for t in targets:
        assert isinstance(t, ForgeTarget)
        assert t.throttle_temp_c > 0 and t.power_ceiling_w > 0 and t.latency_budget_ms > 0
        assert t.export_format and t.backend and t.backend_modules


def test_av_pack_targets_are_disjoint_from_sec():
    av = {t.name for t in get_pack("av").forge_targets}
    sec = {t.name for t in get_pack("sec").forge_targets}
    assert av.isdisjoint(sec)


def test_resolver_finds_targets_across_packs():
    assert domain.resolve_forge_target("agx_orin_trt").name == "agx_orin_trt"     # AV
    assert domain.resolve_forge_target("hailo8_m2").name == "hailo8_m2"           # Sec
    assert domain.resolve_forge_target("does_not_exist") is None


def test_av_thermal_and_budget_are_byte_identical_to_the_registry():
    for name, env in TARGET_THERMAL.items():
        assert _thermal_for(name) == env
    for name, budget in TARGET_BUDGET_MS.items():
        assert _budget_for(name) == budget


def test_forgyx_thermal_and_cooptimize_work_for_a_sec_target():
    # Previously these raised "unknown target"; now the Sec pack supplies the envelope.
    env = thermal_envelope("hailo8_m2", {"peak_temp_c": 70.0, "power_w": 10.0, "throttled_fps": 30.0},
                           cold_fps=30.0)
    assert env["passed"] is True and env["target"] == "hailo8_m2"

    plan = plan_cooptimization("x86_onnx_nvr", base_latency_ms=20.0, base_map50=0.6)
    assert plan["budget_ms"] == 15.0 and plan["feasible"] is True


def test_forgyx_gate_stays_generic_over_opaque_target_strings():
    from services.forgyx.gate import pareto_rank

    benches = [
        {"target": "ambarella_cv5", "p95": 30.0, "map50": 0.55},
        {"target": "hailo8_m2", "p95": 9.0, "map50": 0.58},
    ]
    ranked = pareto_rank(benches)
    assert len(ranked) == 2 and all("target" in b for b in ranked)
