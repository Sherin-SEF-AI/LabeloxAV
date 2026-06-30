"""M-CAL.4: cross-camera extrinsic consistency. The relative pose between two resolved calibrations, fed to
the epipolar (Sampson) residual, must read near zero for points projected consistently into both cameras and
must inflate when one camera's pose is wrong. This proves the relative-pose convention matches the projection
and that the dormant epipolar check now produces a real signal."""

from __future__ import annotations

import numpy as np

from services.calibration.extrinsics_check import epipolar_consistency, relative_pose
from services.calibration.resolve import Calibration
from services.lidar.project import project_to_camera

W, H = 1280, 720


def _cam(name, yaw, y):
    return Calibration(name, "pinhole", 1000.0, 1000.0, 640.0, 360.0, [],
                       rpy_deg=(0.0, 0.0, yaw), xyz_m=(0.0, y, 1.5), source="measured", quality=1.0)


def _seen_by_both(a, b, n, seed):
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.uniform(7, 18, n), rng.uniform(-3, 3, n), rng.uniform(0.0, 3.0, n)], 1).astype(np.float32)
    pa, pb = project_to_camera(pts, a.cam_id, W, H, a), project_to_camera(pts, b.cam_id, W, H, b)
    both = pa["in_front"] & pb["in_front"] & pa["in_image"] & pb["in_image"]
    return pa["uv"][both], pb["uv"][both]


def test_relative_pose_is_a_rotation():
    r, _ = relative_pose(_cam("cam_a", -12, 0.5), _cam("cam_b", 12, -0.5))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)   # orthonormal


def test_consistent_rig_has_near_zero_residual():
    a, b = _cam("cam_a", -12, 0.6), _cam("cam_b", 12, -0.6)
    ua, ub = _seen_by_both(a, b, 24, 0)
    assert len(ua) >= 8
    res = epipolar_consistency(ua, ub, a, b)
    assert res["mean_sampson_px"] is not None and res["mean_sampson_px"] < 1.0   # exact projections fit


def test_wrong_extrinsics_inflate_the_residual():
    a, b = _cam("cam_a", -12, 0.6), _cam("cam_b", 12, -0.6)
    ua, ub = _seen_by_both(a, b, 32, 1)
    good = epipolar_consistency(ua, ub, a, b)["mean_sampson_px"]
    bad_b = _cam("cam_b", 30, -0.6)                       # a corrupted yaw breaks the epipolar geometry
    bad = epipolar_consistency(ua, ub, a, bad_b)["mean_sampson_px"]
    assert good < 0.1 and bad > 1.0 and bad > good + 1.0   # consistent ~0, a wrong pose breaks sub-pixel fit
