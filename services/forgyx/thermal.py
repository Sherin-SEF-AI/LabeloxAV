"""FORGYX thermal/power envelope (M16). An edge model that hits its latency budget cold can still fail in a
sealed automotive enclosure once the SoC heats and throttles. This checks a candidate against the target's
thermal and power envelope: whether sustained inference stays under the throttle temperature and power ceiling,
and what sustained frame rate survives after any throttling. Capability-honest: the readings come from a real
device farm run (hardware-in-the-loop); with no device readings present the envelope is refused rather than
assumed passing. Pure over the readings, so the throttle logic is testable."""

from __future__ import annotations

# per-target thermal ceilings: throttle onset temperature (C) and sustained power ceiling (W). Legacy AV
# silicon registry; the active-pack forge targets are consulted first (a pack's own hardware envelopes), and
# this is the fallback for targets a pack does not declare.
TARGET_THERMAL = {
    "sentrixai_litert": {"throttle_temp_c": 80.0, "power_ceiling_w": 6.0},
    "agx_orin_trt": {"throttle_temp_c": 90.0, "power_ceiling_w": 60.0},
    "orin_nano_trt": {"throttle_temp_c": 85.0, "power_ceiling_w": 15.0},
    "pi_hailo": {"throttle_temp_c": 80.0, "power_ceiling_w": 8.0},
}


def _thermal_for(target: str) -> dict | None:
    """The target's thermal envelope from any pack's forge targets, else the legacy AV registry. For an AV
    target the pack values mirror the registry, so the result is identical."""
    from services.domain import resolve_forge_target

    ft = resolve_forge_target(target)
    if ft is not None:
        return {"throttle_temp_c": ft.throttle_temp_c, "power_ceiling_w": ft.power_ceiling_w}
    return TARGET_THERMAL.get(target)


def thermal_envelope(target: str, readings: dict, cold_fps: float) -> dict:
    """Evaluate a device-farm sustained-load run against the target envelope.

    readings is a real hardware-in-the-loop measurement:
    {peak_temp_c, steady_temp_c, power_w, throttled_fps}. Returns whether the run stayed within the throttle
    temperature and power ceiling, the sustained fps after throttling, and the headroom to the throttle point.
    passed=False when the SoC throttled or exceeded the power ceiling. Raises when readings are absent (no
    fabricated envelope)."""
    env = _thermal_for(target)
    if env is None:
        raise ValueError(f"unknown target {target}")
    required = ("peak_temp_c", "power_w", "throttled_fps")
    if not readings or any(readings.get(k) is None for k in required):
        raise ValueError("thermal envelope needs a real device-farm run (peak_temp_c, power_w, throttled_fps)")

    peak = float(readings["peak_temp_c"])
    power = float(readings["power_w"])
    sustained_fps = float(readings["throttled_fps"])
    throttled = peak >= env["throttle_temp_c"] or sustained_fps < cold_fps * 0.98
    over_power = power > env["power_ceiling_w"]
    headroom_c = round(env["throttle_temp_c"] - peak, 2)
    return {"target": target, "passed": not (throttled or over_power),
            "throttled": throttled, "over_power": over_power,
            "sustained_fps": round(sustained_fps, 3), "cold_fps": round(cold_fps, 3),
            "headroom_c": headroom_c, "power_w": round(power, 3),
            "power_ceiling_w": env["power_ceiling_w"], "throttle_temp_c": env["throttle_temp_c"]}
