"""M-IMU.1: derived ego-state. From a GNSS track + CAN speed, derive heading, yaw rate, longitudinal and
lateral acceleration, and jerk. These drive synthetic tracks (straight acceleration, a turn, a hard brake,
and GNSS-only speed) and check the derived signal matches the motion."""

from __future__ import annotations

from services.intelligence.egostate import derive_ego_state

SEC = 1_000_000_000
DEG_E = 0.0001   # ~11.1 m of longitude at the equator


def test_straight_acceleration():
    # due east at the equator, accelerating 10 -> 12 -> 14 m/s
    s = [(0, 0.0, 0.0, 10.0), (SEC, 0.0, DEG_E, 12.0), (2 * SEC, 0.0, 2 * DEG_E, 14.0)]
    out = derive_ego_state(s)
    assert abs(out[1]["heading_deg"] - 90.0) < 1.0          # heading east
    assert abs(out[2]["long_accel"] - 2.0) < 0.05           # (14-12)/1s
    assert abs(out[2]["yaw_rate"]) < 1e-3                    # straight: no yaw
    assert abs(out[2]["lat_accel"]) < 0.05


def test_turn_produces_yaw_rate_and_lateral_accel():
    # east, then north: a sharp left turn
    s = [(0, 0.0, 0.0, 10.0), (SEC, 0.0, DEG_E, 10.0), (2 * SEC, DEG_E, DEG_E, 10.0)]
    out = derive_ego_state(s)
    assert out[2]["yaw_rate"] is not None and abs(out[2]["yaw_rate"]) > 0.1
    assert out[2]["lat_accel"] is not None and abs(out[2]["lat_accel"]) > 0.1


def test_speed_from_gnss_when_no_can_speed():
    s = [(0, 0.0, 0.0, None), (SEC, 0.0, DEG_E, None)]
    out = derive_ego_state(s)
    assert out[1]["speed_mps"] is not None and 9.0 < out[1]["speed_mps"] < 13.0   # ~11.1 m/s


def test_hard_brake_is_negative_accel_with_jerk():
    s = [(0, 0.0, 0.0, 20.0), (SEC, 0.0, DEG_E, 20.0), (2 * SEC, 0.0, 2 * DEG_E, 8.0)]
    out = derive_ego_state(s)
    assert out[2]["long_accel"] < -5.0                      # hard deceleration
    assert out[2]["jerk"] is not None


def test_first_sample_has_no_derivatives():
    out = derive_ego_state([(0, 0.0, 0.0, 10.0)])
    assert out[0]["yaw_rate"] is None and out[0]["long_accel"] is None and out[0]["jerk"] is None
