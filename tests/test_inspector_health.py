"""Session health checks (M-I.2): rate deviation, dropout, missing topic, and verdict rollup over an index."""

from __future__ import annotations

from core.config import get_settings
from services.inspector.health import evaluate, verdict_of


def _healthy_index():
    return {
        "topics": {
            "/camera/cam_f": {"name": "/camera/cam_f", "schema": "foxglove.CompressedImage", "count": 60, "rate": 10.0, "first_ts": 0, "last_ts": 6_000_000_000},
            "/imu": {"name": "/imu", "schema": "sensor.Imu", "count": 1200, "rate": 200.0, "first_ts": 0, "last_ts": 6_000_000_000},
            "/gnss": {"name": "/gnss", "schema": "foxglove.LocationFix", "count": 60, "rate": 10.0, "first_ts": 0, "last_ts": 6_000_000_000},
            "/can/speed": {"name": "/can/speed", "schema": "can.Signal", "count": 600, "rate": 100.0, "first_ts": 0, "last_ts": 6_000_000_000},
        },
        "gaps": {},
        "time_range": [0, 6_000_000_000],
    }


def _gnss():
    return {"present": True, "total": 60, "valid": 60}


def test_healthy_is_pass():
    checks = evaluate(_healthy_index(), _gnss(), cfg=get_settings().inspector)
    assert verdict_of(checks) == "pass"
    assert all(c["status"] == "pass" for c in checks)


def test_wrong_imu_rate_fails():
    idx = _healthy_index()
    idx["topics"]["/imu"]["rate"] = 247.0     # the 247Hz-against-200 catch
    checks = evaluate(idx, _gnss(), cfg=get_settings().inspector)
    assert verdict_of(checks) == "fail"
    imu = next(c for c in checks if c["name"] == "rate_deviation" and c["evidence"].get("topic") == "/imu")
    assert imu["status"] == "fail" and imu["evidence"]["measured"] == 247.0


def test_dropout_fails_with_evidence_window():
    idx = _healthy_index()
    idx["gaps"]["/imu"] = [[1_000_000_000, 2_010_000_000]]   # a ~1s dropout at 200Hz nominal (5ms period)
    checks = evaluate(idx, _gnss(), cfg=get_settings().inspector)
    drop = next(c for c in checks if c["name"] == "dropout")
    assert drop["status"] == "fail" and drop["evidence"]["worst_window"] == [1_000_000_000, 2_010_000_000]
    assert verdict_of(checks) == "fail"


def test_missing_required_topic_fails():
    idx = _healthy_index()
    del idx["topics"]["/imu"]                  # IMU is a required rig topic
    checks = evaluate(idx, _gnss(), cfg=get_settings().inspector)
    miss = next(c for c in checks if c["name"] == "missing_topic")
    assert miss["status"] == "fail" and "imu" in miss["evidence"]["expected"]["match"]
    assert verdict_of(checks) == "fail"


def test_verdict_ordering():
    assert verdict_of([{"status": "pass"}, {"status": "warn"}]) == "warn"
    assert verdict_of([{"status": "warn"}, {"status": "fail"}]) == "fail"
    assert verdict_of([{"status": "pass"}]) == "pass"
