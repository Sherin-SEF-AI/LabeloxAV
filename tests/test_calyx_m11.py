"""CALYX M11 tests: self-cal recovery undoes measured drift, calibration confidence is monotonic, targetless
calibration recovers known intrinsics from a synthetic scene, and cross-session consensus fuses toward truth."""

import numpy as np
from scipy.spatial.transform import Rotation

from core.geometry import se3, se3_compose, se3_magnitude
from services.calyx.consensus import fuse_calibrations
from services.calyx.recover import correct_extrinsic, is_recoverable, recover
from services.calyx.targetless import calibrate_targetless, focal_from_vanishing_points, pitch_from_horizon
from services.calyx.uncertainty import calibration_confidence, sample_confidence


def test_correct_extrinsic_undoes_drift():
    nominal = se3(Rotation.from_euler("xyz", [3, -2, 1], degrees=True).as_matrix(), [0.2, 0.0, 0.5])
    drift = se3(Rotation.from_euler("y", 1.2, degrees=True).as_matrix(), [0.03, 0, 0])
    corrected = correct_extrinsic(nominal, drift)
    # applying the drift to the corrected extrinsic returns the nominal (the drift is undone)
    np.testing.assert_allclose(se3_compose(drift, corrected), nominal, atol=1e-9)


def test_recover_applies_only_when_confident_and_not_blocking():
    nominal = np.eye(4)
    drift = se3(Rotation.from_euler("y", 1.2, degrees=True).as_matrix(), [0.03, 0, 0])
    good = recover(nominal, drift, residual_px=1.0, n_corr=100, severity="drift_detected")
    assert good["apply"] is True and good["confidence"] > 0.4
    blocked = recover(nominal, drift, residual_px=1.0, n_corr=100, severity="block")
    assert blocked["apply"] is False           # a blocking drift is not silently recovered
    thin = recover(nominal, drift, residual_px=12.0, n_corr=3, severity="drift_detected")
    assert thin["apply"] is False              # too weak to trust


def test_is_recoverable():
    assert is_recoverable("drift_detected", 0.8)
    assert not is_recoverable("block", 0.9)
    assert not is_recoverable("drift_detected", 0.1)


def test_calibration_confidence_monotonic():
    assert calibration_confidence(1.0, 100) > calibration_confidence(6.0, 100)   # worse residual -> lower
    assert calibration_confidence(2.0, 100) > calibration_confidence(2.0, 5)     # more corr -> higher
    assert calibration_confidence(None, 100) == 0.0
    assert 0.0 <= calibration_confidence(2.0, 50) <= 1.0
    assert sample_confidence(0.8, frame_quality=0.2) < sample_confidence(0.8, frame_quality=1.0)


def test_targetless_recovers_known_focal_and_pitch():
    f, cx, cy = 900.0, 640.0, 360.0
    # two orthogonal camera-frame directions d1.d2 = 0, both in front of the camera
    d1, d2 = np.array([1.0, 0.0, 1.0]), np.array([-1.0, 2.0, 1.0])
    assert abs(d1 @ d2) < 1e-9
    vp1 = (f * d1[0] / d1[2] + cx, f * d1[1] / d1[2] + cy)
    vp2 = (f * d2[0] / d2[2] + cx, f * d2[1] / d2[2] + cy)
    assert abs(focal_from_vanishing_points(vp1, vp2, cx, cy) - f) < 1e-6

    horizon_y = cy - f * np.tan(np.radians(5.0))
    assert abs(pitch_from_horizon(horizon_y, cy, f) - 5.0) < 1e-6

    out = calibrate_targetless(vp1, vp2, horizon_y, cx, cy)
    assert out["ok"] and abs(out["focal"] - f) < 1e-3 and abs(out["pitch_deg"] - 5.0) < 1e-2


def test_targetless_rejects_non_orthogonal():
    # two vanishing points on the same side of the principal point are not orthogonal in view
    assert focal_from_vanishing_points((900, 360), (1000, 360), 640, 360) is None


def test_consensus_fuses_toward_truth():
    truth_rpy, truth_xyz = np.array([1.0, 2.0, 3.0]), np.array([0.10, 0.0, 0.50])
    rng = np.random.default_rng(0)
    calibs = []
    for _ in range(8):
        calibs.append({"rpy_deg": (truth_rpy + rng.normal(0, 0.3, 3)).tolist(),
                       "xyz_m": (truth_xyz + rng.normal(0, 0.01, 3)).tolist(),
                       "confidence": float(rng.uniform(0.5, 0.9))})
    fused = fuse_calibrations(calibs)
    assert fused["n"] == 8
    assert np.allclose(fused["rpy_deg"], truth_rpy, atol=0.3)   # fused prior near truth
    assert fused["confidence"] > 0.5


def test_consensus_empty():
    assert fuse_calibrations([])["n"] == 0
