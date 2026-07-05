"""CALYX calibration-drift tests: a synthetic extrinsic perturbation is recovered as an SE(3) delta within
tolerance, and its magnitude gates the session by severity (M2 acceptance)."""

import numpy as np
from scipy.spatial.transform import Rotation

from core.config import CalyxSettings
from core.geometry import se3, se3_magnitude, transform_points
from services.calyx.drift import (
    estimate_extrinsic_drift,
    rigid_align,
    severity,
    temporal_consistency,
)

CFG = CalyxSettings()


def _points(n=200, seed=0):
    return np.random.default_rng(seed).uniform(-5, 5, size=(n, 3))


def test_rigid_align_recovers_known_transform():
    src = _points()
    T = se3(Rotation.from_euler("xyz", [2, -3, 1], degrees=True).as_matrix(), [0.1, -0.2, 0.05])
    dst = transform_points(T, src)
    T_est, resid = rigid_align(src, dst)
    np.testing.assert_allclose(T_est, T, atol=1e-6)
    assert resid < 1e-6


def test_drift_detected_severity_for_small_perturbation():
    src = _points()
    perturb = se3(Rotation.from_euler("y", 1.5, degrees=True).as_matrix(), [0.10, 0.0, 0.0])
    dst = transform_points(perturb, src)
    out = estimate_extrinsic_drift(src, dst, CFG)
    # recovered magnitude matches the injected perturbation
    assert abs(out["magnitude"]["rotation_deg"] - 1.5) < 0.05
    assert abs(out["magnitude"]["translation_m"] - 0.10) < 1e-3
    assert out["severity"] == "drift_detected"   # 1.5 deg / 0.10 m is between flag and block


def test_block_severity_for_large_perturbation():
    src = _points()
    perturb = se3(Rotation.from_euler("z", 3.0, degrees=True).as_matrix(), [0.30, 0.0, 0.0])
    dst = transform_points(perturb, src)
    out = estimate_extrinsic_drift(src, dst, CFG)
    assert out["severity"] == "block"


def test_ok_severity_when_no_drift():
    src = _points()
    out = estimate_extrinsic_drift(src, src.copy(), CFG)
    assert out["severity"] == "ok"
    assert out["magnitude"]["rotation_deg"] < 1e-6


def test_low_confidence_below_min_correspondences():
    src = _points(n=4)
    out = estimate_extrinsic_drift(src, src.copy(), CFG)
    assert out["confidence"] == "low" and out["severity"] == "ok"


def test_severity_thresholds_direct():
    assert severity(se3_magnitude(se3(np.eye(3), [0, 0, 0])), CFG) == "ok"
    big = se3(Rotation.from_euler("x", 5, degrees=True).as_matrix(), [0, 0, 0])
    assert severity(se3_magnitude(big), CFG) == "block"


def test_temporal_consistency_flags_discontinuity():
    smooth = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert temporal_consistency(smooth, CFG)["ok"]
    jump = [0.0, 1.0, 2.0, 20.0, 3.0, 4.0]  # a spike: 18 deg up then 17 deg back, both over the 5 deg limit
    r = temporal_consistency(jump, CFG)
    assert not r["ok"] and r["discontinuities"] == 2  # into and out of the spike
