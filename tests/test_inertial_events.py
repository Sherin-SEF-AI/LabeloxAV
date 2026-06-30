"""M-IMU.4: inertial event tagging, anomaly pre-marking, and maneuver segmentation on the ego-state series."""

from __future__ import annotations

from services.intelligence.inertial_events import (
    detect_inertial_events,
    inertial_anomalies,
    segment_maneuvers,
)


def _s(ts, **kw):
    base = {"ts_ns": ts, "speed_mps": 10.0, "heading_deg": 0.0, "yaw_rate": 0.0,
            "long_accel": 0.0, "lat_accel": 0.0, "jerk": 0.0}
    base.update(kw)
    return base


def test_hard_brake_event_spans_the_window():
    series = [_s(0), _s(1, long_accel=-5.0), _s(2, long_accel=-6.0), _s(3, long_accel=0.0)]
    hb = [e for e in detect_inertial_events(series) if e["kind"] == "hard_brake"]
    assert len(hb) == 1
    assert hb[0]["t_in_ns"] == 1 and hb[0]["t_out_ns"] == 2 and hb[0]["peak"] == -6.0
    assert 0.0 < hb[0]["severity"] <= 1.0


def test_swerve_and_impact_detected():
    series = [_s(0), _s(1, lat_accel=5.5), _s(2, jerk=12.0)]
    kinds = {e["kind"] for e in detect_inertial_events(series)}
    assert "swerve" in kinds and "impact" in kinds


def test_hard_accel_distinct_from_brake():
    series = [_s(0), _s(1, long_accel=4.0)]
    kinds = {e["kind"] for e in detect_inertial_events(series)}
    assert kinds == {"hard_accel"}


def test_anomaly_premark_flags_the_spike_as_pending():
    series = [_s(i, jerk=0.4 + 0.05 * (i % 3)) for i in range(20)] + [_s(20, jerk=15.0)]
    an = inertial_anomalies(series, key="jerk")
    assert any(a["ts_ns"] == 20 for a in an)
    assert all(a["status"] == "pending" for a in an)
    assert not any(a["ts_ns"] < 20 for a in an)   # the small jerks are not anomalies


def test_maneuver_segmentation_merges_contiguous_labels():
    series = [_s(0), _s(1), _s(2, long_accel=-2.0), _s(3, long_accel=-2.0), _s(4, speed_mps=0.0)]
    kinds = [s["kind"] for s in segment_maneuvers(series)]
    assert kinds == ["cruise", "brake", "stationary"]
