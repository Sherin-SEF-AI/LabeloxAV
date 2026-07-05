"""Kernel geometry tests: SE(3) algebra, SLERP, projection round-trips, and epipolar consistency."""

import numpy as np
from scipy.spatial.transform import Rotation

from core.geometry import (
    backproject,
    epipolar_residual,
    project_points,
    reprojection_error,
    rotation_angle_deg,
    se3,
    se3_compose,
    se3_delta,
    se3_inverse,
    se3_magnitude,
    slerp,
    slerp_series,
    transform_points,
)


def _K():
    return np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])


def test_se3_inverse_roundtrip():
    R = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    T = se3(R, [1.0, -2.0, 3.0])
    np.testing.assert_allclose(se3_compose(T, se3_inverse(T)), np.eye(4), atol=1e-9)
    pts = np.random.default_rng(0).normal(size=(50, 3))
    back = transform_points(se3_inverse(T), transform_points(T, pts))
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_se3_compose_associativity():
    rng = np.random.default_rng(1)
    A = se3(Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), rng.normal(size=3))
    B = se3(Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), rng.normal(size=3))
    C = se3(Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), rng.normal(size=3))
    np.testing.assert_allclose(se3_compose(A, B, C), (A @ B) @ C, atol=1e-12)


def test_se3_delta_and_magnitude():
    a = se3(Rotation.from_euler("z", 5, degrees=True).as_matrix(), [0.0, 0.0, 0.0])
    b = se3(Rotation.from_euler("z", 8, degrees=True).as_matrix(), [0.10, 0.0, 0.0])
    mag = se3_magnitude(se3_delta(a, b))
    assert abs(mag["rotation_deg"] - 3.0) < 1e-6
    assert abs(mag["translation_m"] - 0.10) < 1e-6


def test_slerp_endpoints_and_midpoint():
    q0 = Rotation.identity().as_quat()
    q1 = Rotation.from_euler("y", 90, degrees=True).as_quat()
    np.testing.assert_allclose(rotation_angle_deg(slerp(q0, q1, 0.0), q0), 0.0, atol=1e-9)
    np.testing.assert_allclose(rotation_angle_deg(slerp(q0, q1, 1.0), q1), 0.0, atol=1e-6)
    mid = slerp(q0, q1, 0.5)
    assert abs(rotation_angle_deg(q0, mid) - 45.0) < 1e-6


def test_slerp_series_clamps_and_interpolates():
    times = np.array([0.0, 1.0, 2.0])
    quats = np.vstack([Rotation.from_euler("z", a, degrees=True).as_quat() for a in (0, 30, 90)])
    out = slerp_series(times, quats, np.array([-1.0, 0.5, 3.0]))
    # clamped ends match the first/last key, the interior is halfway to the 30 deg key
    assert abs(rotation_angle_deg(out[0], quats[0])) < 1e-6
    assert abs(rotation_angle_deg(out[2], quats[2])) < 1e-6
    assert abs(rotation_angle_deg(quats[0], out[1]) - 15.0) < 1e-6


def test_project_backproject_roundtrip():
    K = _K()
    rng = np.random.default_rng(2)
    pts = np.hstack([rng.uniform(-3, 3, size=(40, 2)), rng.uniform(2, 30, size=(40, 1))])
    uv, valid = project_points(pts, K)
    assert valid.all()
    back = backproject(uv, pts[:, 2], K)
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_project_marks_behind_camera_invalid():
    K = _K()
    pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
    _, valid = project_points(pts, K)
    assert valid.tolist() == [True, False]


def test_reprojection_error_zero_for_consistent():
    K = _K()
    T_cam_world = se3(Rotation.from_euler("xyz", [3, -4, 2], degrees=True).as_matrix(), [0.2, -0.1, 0.5])
    rng = np.random.default_rng(3)
    world = np.hstack([rng.uniform(-4, 4, size=(30, 2)), rng.uniform(3, 25, size=(30, 1))])
    uv, valid = project_points(transform_points(T_cam_world, world), K)
    rep = reprojection_error(world[valid], uv[valid], T_cam_world, K)
    assert rep["rms_px"] < 1e-6


def test_epipolar_residual_zero_for_consistent_stereo():
    K = _K()
    R = Rotation.from_euler("y", 5, degrees=True).as_matrix()
    t = np.array([-0.30, 0.0, 0.0])  # baseline
    rng = np.random.default_rng(4)
    world = np.hstack([rng.uniform(-2, 2, size=(60, 2)), rng.uniform(4, 20, size=(60, 1))])
    uv1, v1 = project_points(world, K)                      # camera 1 at origin
    uv2, v2 = project_points(transform_points(se3(R, t), world), K)  # camera 2
    m = v1 & v2
    res = epipolar_residual(uv1[m], uv2[m], R, t, K, K)
    assert res["rms_px"] < 1e-6
