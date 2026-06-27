"""Time-sync validation (M3.0). The STM32 PPS time base aligns camera, IMU, and GNSS. Validate the IMU
sample rate against the target (this catches the 247Hz-vs-200Hz drift that surfaced on the rig) and the
inter-sensor time offset against the PPS-aligned expectation."""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("calib_timesync")


def imu_rate_check(imu_ts_ns: list[int]) -> dict:
    cfg = get_settings().spatial
    if len(imu_ts_ns) < 2:
        return {"hz": None, "target": cfg.imu_hz_target, "ok": False, "reason": "insufficient samples"}
    ts = sorted(imu_ts_ns)
    dur_s = (ts[-1] - ts[0]) / 1e9
    hz = (len(ts) - 1) / dur_s if dur_s > 0 else 0.0
    ok = abs(hz - cfg.imu_hz_target) <= cfg.imu_hz_tolerance
    return {"hz": round(hz, 1), "target": cfg.imu_hz_target, "tolerance": cfg.imu_hz_tolerance, "ok": bool(ok)}


def time_offset_check(a_ts_ns: list[int], b_ts_ns: list[int]) -> dict:
    """Median time offset of stream a relative to b (nearest-neighbour). PPS sync should keep it small."""
    cfg = get_settings().spatial
    if not a_ts_ns or not b_ts_ns:
        return {"offset_ns": None, "status": "warn", "ok": True}
    a, b = np.array(sorted(a_ts_ns)), np.array(sorted(b_ts_ns))
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    nearest = np.where(np.abs(a - b[idx - 1]) < np.abs(a - b[idx]), b[idx - 1], b[idx])
    offset = int(np.median(a - nearest))
    mag = abs(offset)
    status = "pass" if mag <= cfg.time_offset_ns_warn else ("warn" if mag <= cfg.time_offset_ns_fail else "fail")
    return {"offset_ns": offset, "warn_ns": cfg.time_offset_ns_warn, "fail_ns": cfg.time_offset_ns_fail,
            "status": status, "ok": status != "fail"}


def validate_timesync(imu_ts_ns: list[int], cam_ts_ns: list[int] | None = None,
                      gnss_ts_ns: list[int] | None = None) -> dict:
    rate = imu_rate_check(imu_ts_ns)
    offsets: dict = {}
    if cam_ts_ns:
        offsets["cam_imu"] = time_offset_check(cam_ts_ns, imu_ts_ns)
    if gnss_ts_ns:
        offsets["gnss_imu"] = time_offset_check(gnss_ts_ns, imu_ts_ns)
    ok = rate["ok"] and all(o["ok"] for o in offsets.values())
    return {"imu_rate": rate, "offsets": offsets,
            "time_offset_ns": offsets.get("cam_imu", {}).get("offset_ns"), "ok": ok}
