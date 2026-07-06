"""SANYX M10 tests: root-cause classification on known-fault signatures, rig predictive-maintenance trend
detection, and mid-stream flagging by the streaming accumulator."""

import numpy as np

from core.config import SanyxSettings
from services.sanyx.rootcause import classify, feature_vector
from services.sanyx.stream import StreamingHealth
from services.sanyx.trends import detect_trends

CFG = SanyxSettings()


def _check(name, status, **ev):
    return {"name": name, "status": status, "score": 0.2, "detail": "", "evidence": ev}


def test_rootcause_time_sync_loss_dominates():
    r = classify([_check("time_sync", "quarantine", pps_present=False)])
    assert r["fault"] == "time_sync_loss" and r["confidence"] == 1.0 and r["remediation"]


def test_rootcause_lens_fouling():
    r = classify([_check("lens", "quarantine", occlusion_area_frac=0.5, highfreq_ratio=0.001)])
    assert r["fault"] == "lens_fouling"


def test_rootcause_gps_multipath():
    r = classify([_check("gps", "degraded", median_hdop=20.0, longest_dropout_s=10.0)])
    assert r["fault"] == "gps_urban_canyon_multipath"


def test_rootcause_imu_thermal_drift():
    r = classify([_check("imu", "degraded", bias_jump=3.0, saturation_frac=0.05)])
    assert r["fault"] == "imu_thermal_drift"


def test_rootcause_loose_connector():
    r = classify([_check("dropped_frames", "quarantine", worst_gap=60, worst_drop_rate=0.1)])
    assert r["fault"] == "loose_gmsl2_connector"


def test_rootcause_none_when_healthy():
    assert classify([_check("exposure", "pass", well_exposed_frac=0.98)])["fault"] is None


def test_feature_vector_length():
    assert len(feature_vector([_check("gps", "degraded", median_hdop=5.0)])) == 10


def test_trends_flags_a_falling_module():
    alerts = detect_trends({"lens": [0.9, 0.8, 0.6, 0.4], "exposure": [0.9, 0.9, 0.9, 0.9]})
    by_comp = {a["component"]: a for a in alerts}
    assert "camera_optics" in by_comp          # lens is falling
    assert "camera_exposure" not in by_comp     # exposure is stable
    assert by_comp["camera_optics"]["trend"] == "falling"


def test_trends_severity_escalates_near_failure():
    alerts = detect_trends({"lens": [0.5, 0.42, 0.35, 0.31]})   # already near the 0.3 fail floor
    assert alerts and alerts[0]["severity"] in ("warn", "critical")


def test_streaming_flags_mid_stream():
    sh = StreamingHealth(CFG)
    step = int(1e8)
    r1 = sh.feed({"cam_ft": [i * step for i in range(30)]})     # contiguous, healthy
    assert not r1["flagged"]
    # a big gap in the next window (jump past ~40 frames) crosses the dropped-frames quarantine threshold
    r2 = sh.feed({"cam_ft": [(72 + i) * step for i in range(10)]})
    assert r2["flagged"] and sh.first_flag_window == 2
