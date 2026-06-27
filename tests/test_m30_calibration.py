"""M3.0 calibration validation, the two acceptance proofs as deterministic units: the FOV check flags a
camera carrying the wrong lens's intrinsics, and the time-sync check flags the 247Hz-vs-200Hz IMU drift."""

from __future__ import annotations

from core.config import get_settings
from services.calibration.intrinsics import validate_intrinsics
from services.calibration.timesync import imu_rate_check


def test_fov_check_flags_mismatched_lens():
    cfg = get_settings()
    narrow, wide = cfg.rig.lenses["narrow"], cfg.rig.lenses["wide"]
    # a correctly configured narrow camera passes its FOV check
    assert validate_intrinsics(narrow, "narrow")["fov_check"]["ok"] is True
    # a camera CONFIGURED narrow but carrying WIDE intrinsics fails (the narrow-vs-wide lens-mix catch)
    bad = validate_intrinsics(wide, "narrow")["fov_check"]
    assert bad["ok"] is False
    assert bad["implied_fov_deg"] > 90 and bad["expected_fov_deg"] < 60  # ~120 implied vs ~37 configured


def test_timesync_flags_imu_rate_drift():
    # a clean 200 Hz IMU stream passes the rate check
    ts200 = [int(i * (1e9 / 200)) for i in range(200)]
    assert imu_rate_check(ts200)["ok"] is True
    # a 247 Hz stream (the drift that surfaced on the rig) fails
    ts247 = [int(i * (1e9 / 247)) for i in range(247)]
    bad = imu_rate_check(ts247)
    assert bad["ok"] is False and bad["hz"] > 240
