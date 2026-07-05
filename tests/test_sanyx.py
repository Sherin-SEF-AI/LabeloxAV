"""SANYX ingest-QA tests: each check family scores healthy input high and corrupted input low, and the
aggregate quarantines a session on either a low overall score or a single hard-failing check. This is the
M1 acceptance at the unit level (the corrupted session is quarantined and would never produce a Sample)."""

import numpy as np

from core.config import SanyxSettings
from services.sanyx import checks as C
from services.sanyx.score import aggregate

CFG = SanyxSettings()


def _imu_ts(hz=200.0, n=2000, start=0):
    step = int(1e9 / hz)
    return [start + i * step for i in range(n)]


def test_time_sync_healthy_vs_pps_loss():
    imu = _imu_ts()
    cams = {"cam_ft": _imu_ts(hz=30, n=300), "cam_rt": _imu_ts(hz=30, n=300)}
    ok = C.check_time_sync(CFG, imu, cams, gnss_ts_ns=_imu_ts(hz=10, n=100), pps_ts_ns=_imu_ts(hz=1, n=10))
    assert ok["status"] == "pass" and ok["score"] > 0.9

    strict = SanyxSettings(pps_required=True)
    lost = C.check_time_sync(strict, imu, cams, pps_ts_ns=[])
    assert lost["status"] == "quarantine" and lost["score"] == 0.0


def test_dropped_frames_healthy_vs_gap():
    healthy = C.check_dropped_frames(CFG, {"cam_ft": list(range(1000))})
    assert healthy["status"] == "pass" and healthy["score"] == 1.0

    seq = list(range(0, 400)) + list(range(460, 1000))  # a 60-frame gap
    bad = C.check_dropped_frames(CFG, {"cam_ft": seq})
    assert bad["status"] == "quarantine"
    assert bad["evidence"]["worst_gap"] >= CFG.max_gap_frames_fail


def test_exposure_healthy_vs_mostly_clipped():
    good = [{"well_exposed": True} for _ in range(100)]
    assert C.check_exposure(CFG, good)["status"] == "pass"

    bad = [{"well_exposed": i < 30} for i in range(100)]  # 30% well exposed
    r = C.check_exposure(CFG, bad)
    assert r["status"] == "quarantine"


def test_gps_healthy_vs_no_fix():
    ts = _imu_ts(hz=10, n=600)
    good = C.check_gps(CFG, fix_types=[3] * 600, hdop=[0.8] * 600, ts_ns=ts, rtk_flags=[1] * 600)
    assert good["status"] == "pass" and good["score"] > 0.9

    bad = C.check_gps(CFG, fix_types=[0] * 600, hdop=[20.0] * 600, ts_ns=ts)
    assert bad["status"] == "quarantine"


def test_imu_healthy_vs_saturated():
    n = 2000
    rng = np.random.default_rng(0)
    accel = rng.normal(0, 1.0, size=(n, 3)) + np.array([0, 0, 9.81])
    gyro = rng.normal(0, 0.05, size=(n, 3))
    good = C.check_imu(CFG, accel, gyro, _imu_ts(n=n))
    assert good["status"] == "pass"

    accel_sat = accel.copy()
    accel_sat[:400] = 160.0  # 20% saturated, well over imu_sat_frac_max
    bad = C.check_imu(CFG, accel_sat, gyro, _imu_ts(n=n))
    assert bad["status"] in ("degraded", "quarantine")
    assert bad["evidence"]["saturation_frac"] > CFG.imu_sat_frac_max


def test_lens_healthy_vs_occluded():
    rng = np.random.default_rng(1)
    textured = [rng.integers(0, 255, size=(120, 160), dtype=np.uint8) for _ in range(12)]
    assert C.check_lens_contamination(CFG, textured)["status"] == "pass"

    occluded = []
    for _ in range(12):
        f = rng.integers(60, 255, size=(120, 160), dtype=np.uint8)
        f[:, :80] = 0  # left half persistently blacked out (dirt / obstruction)
        occluded.append(f)
    r = C.check_lens_contamination(CFG, occluded)
    assert r["status"] == "quarantine"
    assert r["evidence"]["occlusion_area_frac"] > CFG.occlusion_area_frac_max


def test_aggregate_pass_and_hard_fail_quarantine():
    healthy = [
        {"name": "time_sync", "score": 1.0, "status": "pass"},
        {"name": "dropped_frames", "score": 1.0, "status": "pass"},
        {"name": "exposure", "score": 0.95, "status": "pass"},
        {"name": "gps", "score": 0.95, "status": "pass"},
        {"name": "imu", "score": 0.95, "status": "pass"},
        {"name": "lens", "score": 1.0, "status": "pass"},
    ]
    agg = aggregate(healthy, CFG)
    assert agg["decision"] == "pass" and agg["score"] >= CFG.pass_min

    hard = list(healthy)
    hard[0] = {"name": "time_sync", "score": 0.0, "status": "quarantine"}  # one hard fail vetoes
    agg2 = aggregate(hard, CFG)
    assert agg2["decision"] == "quarantine" and "time_sync" in agg2["hard_failed"]

    degraded = [{**c, "score": 0.7, "status": "degraded"} for c in healthy]
    assert aggregate(degraded, CFG)["decision"] in ("degraded", "quarantine")
