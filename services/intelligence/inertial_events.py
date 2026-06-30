"""M-IMU.4: inertial event tagging, anomaly pre-marking, and maneuver segmentation on the ego-state series
(derived today, a measured IMU once ingested). Detects hard braking, hard acceleration, swerve, and impact
(speed bump / pothole, surfaced as a jerk spike), pre-marks statistical anomalies for a human to confirm,
and segments the clip into driving maneuvers. Deterministic and testable on the ego-state series.

Honest caveat: with no measured vertical accelerometer, an 'impact' is a longitudinal-jerk spike, a proxy
for a road-defect hit rather than a measured vertical g. It points the labeler at the moment; the measured
IMU sharpens it later.
"""

from __future__ import annotations

import statistics

# thresholds: longitudinal/lateral accel in m/s^2, jerk in m/s^3
_THRESH = {"brake": 3.0, "accel": 3.0, "lat": 4.0, "jerk": 8.0}
# per-event normalizer for the 0..1 severity
_NORM = {"hard_brake": 8.0, "hard_accel": 8.0, "swerve": 9.0, "impact": 25.0}


def _event_runs(series: list[dict], kind: str, key: str, exceeds) -> list[dict]:
    """Contiguous windows where exceeds(value) holds for series[i][key], with the signed peak and severity."""
    events: list[dict] = []
    run: dict | None = None
    for s in series:
        v = s.get(key)
        if v is not None and exceeds(v):
            if run is None:
                run = {"t_in_ns": s["ts_ns"], "t_out_ns": s["ts_ns"], "peak": v}
            run["t_out_ns"] = s["ts_ns"]
            if abs(v) > abs(run["peak"]):
                run["peak"] = v
        elif run is not None:
            events.append({"kind": kind, **run, "severity": round(min(1.0, abs(run["peak"]) / _NORM[kind]), 2)})
            run = None
    if run is not None:
        events.append({"kind": kind, **run, "severity": round(min(1.0, abs(run["peak"]) / _NORM[kind]), 2)})
    return events


def detect_inertial_events(series: list[dict], thresholds: dict | None = None) -> list[dict]:
    """Tag hard-brake, hard-accel, swerve, and impact events on the ego-state series, ordered by start time."""
    t = {**_THRESH, **(thresholds or {})}
    ev: list[dict] = []
    ev += _event_runs(series, "hard_brake", "long_accel", lambda v: v <= -t["brake"])
    ev += _event_runs(series, "hard_accel", "long_accel", lambda v: v >= t["accel"])
    ev += _event_runs(series, "swerve", "lat_accel", lambda v: abs(v) >= t["lat"])
    ev += _event_runs(series, "impact", "jerk", lambda v: abs(v) >= t["jerk"])
    return sorted(ev, key=lambda e: e["t_in_ns"])


def inertial_anomalies(series: list[dict], key: str = "jerk", z: float = 3.5) -> list[dict]:
    """Robust spike pre-marking: samples whose |value| exceeds median + z * (1.4826 * MAD), pending human
    confirmation. MAD-based so a few large hits do not mask the rest the way a standard deviation would."""
    vals = [abs(s[key]) for s in series if s.get(key) is not None]
    if len(vals) < 8:
        return []
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals]) or 1e-6
    out = []
    for s in series:
        v = s.get(key)
        if v is None:
            continue
        score = (abs(v) - med) / (1.4826 * mad)
        if score >= z:
            out.append({"ts_ns": s["ts_ns"], "metric": key, "value": v, "z": round(score, 2), "status": "pending"})
    return out


def _maneuver(s: dict) -> str:
    if s.get("speed_mps") is not None and s["speed_mps"] < 0.5:
        return "stationary"
    if s.get("lat_accel") is not None and abs(s["lat_accel"]) >= 2.0:
        return "turn"
    if s.get("long_accel") is not None and s["long_accel"] <= -1.5:
        return "brake"
    if s.get("long_accel") is not None and s["long_accel"] >= 1.5:
        return "accelerate"
    return "cruise"


def segment_maneuvers(series: list[dict]) -> list[dict]:
    """Merge per-sample maneuver labels into contiguous segments (stationary | turn | brake | accelerate |
    cruise)."""
    segs: list[dict] = []
    cur: dict | None = None
    for s in series:
        lab = _maneuver(s)
        if cur is None or cur["kind"] != lab:
            if cur is not None:
                segs.append(cur)
            cur = {"kind": lab, "t_in_ns": s["ts_ns"], "t_out_ns": s["ts_ns"]}
        else:
            cur["t_out_ns"] = s["ts_ns"]
    if cur is not None:
        segs.append(cur)
    return segs


async def session_inertial_events(session_id) -> dict:
    """Ego-state -> events + anomaly pre-marks + maneuver segments for a session."""
    from services.intelligence.egostate import session_ego_state
    ego = await session_ego_state(session_id)
    series = ego["series"]
    return {"session_id": str(session_id), "source": ego["source"], "n_samples": ego["n_samples"],
            "events": detect_inertial_events(series),
            "anomalies": inertial_anomalies(series),
            "maneuvers": segment_maneuvers(series)}
