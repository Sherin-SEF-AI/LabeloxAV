"""Kernel geometry: the shared SE(3) / SLERP / projection / epipolar substrate.

This is the canonical math the platforms share instead of re-deriving it: LABELOX core uses it for
cross-camera label propagation, CALYX for reprojection residuals and extrinsic drift, ORACLYX for
multi-view triangulation. The scattered helpers under services/lidar and services/calibration should be
migrated to call this module rather than keep private copies.

Conventions:
- Rotations are right-handed. SE(3) poses are 4x4 homogeneous matrices ``T`` such that a point in frame B
  maps to frame A by ``p_a = T_a_b @ [p_b; 1]``. Read ``T_a_b`` as "B expressed in A".
- Quaternions are (x, y, z, w) to match scipy.spatial.transform.Rotation.
- Camera intrinsics ``K`` are 3x3 pinhole matrices; the projection helpers take an optional OpenCV-style
  distortion vector.
- Points are (N, 3) or (N, 2) float arrays; single points may be passed as (3,) / (2,) and are handled.

Everything is numpy + scipy (both already in the stack); no heavy dependency is pulled in.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = [
    "se3",
    "se3_inverse",
    "se3_compose",
    "transform_points",
    "se3_delta",
    "se3_magnitude",
    "slerp",
    "slerp_series",
    "rotation_angle_deg",
    "project_points",
    "backproject",
    "reprojection_error",
    "essential_matrix",
    "fundamental_matrix",
    "sampson_distance",
    "epipolar_residual",
]

_EPS = 1e-9


def _as_points(p: np.ndarray, dim: int) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] != dim:
        raise ValueError(f"expected (N, {dim}) points, got shape {a.shape}")
    return a


# --------------------------------------------------------------------------- SE(3)

def se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) pose from a rotation and a translation.

    ``rotation`` may be a 3x3 rotation matrix, a (x,y,z,w) quaternion, or a (3,) rotation vector (axis-angle).
    ``translation`` is a length-3 vector.
    """
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape == (3, 3):
        rot = r
    elif r.shape == (4,):
        rot = Rotation.from_quat(r).as_matrix()
    elif r.shape == (3,):
        rot = Rotation.from_rotvec(r).as_matrix()
    else:
        raise ValueError(f"rotation must be 3x3, quat(4,), or rotvec(3,); got {r.shape}")
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) pose, computed in closed form (transpose of R, rotated negative t)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_compose(*poses: np.ndarray) -> np.ndarray:
    """Left-to-right composition: ``se3_compose(A, B, C) == A @ B @ C``. Empty returns identity."""
    out = np.eye(4)
    for p in poses:
        out = out @ np.asarray(p, dtype=np.float64)
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply an SE(3) pose to (N, 3) points: ``p_out = R @ p + t``."""
    pts = _as_points(points, 3)
    T = np.asarray(T, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def se3_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The relative pose that carries ``a`` onto ``b``: ``inv(a) @ b``. Used for calibration drift deltas."""
    return se3_inverse(a) @ np.asarray(b, dtype=np.float64)


def se3_magnitude(T: np.ndarray) -> dict:
    """Scalar magnitude of an SE(3) delta: translation norm (metres) and rotation angle (degrees)."""
    T = np.asarray(T, dtype=np.float64)
    trans = float(np.linalg.norm(T[:3, 3]))
    angle = float(np.degrees(Rotation.from_matrix(T[:3, :3]).magnitude()))
    return {"translation_m": trans, "rotation_deg": angle}


# --------------------------------------------------------------------------- rotation interpolation

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two (x,y,z,w) quaternions at fraction ``t`` in [0, 1]."""
    key = Rotation.from_quat(np.vstack([np.asarray(q0, float), np.asarray(q1, float)]))
    return Slerp([0.0, 1.0], key)(float(np.clip(t, 0.0, 1.0))).as_quat()


def slerp_series(key_times: np.ndarray, key_quats: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Interpolate a rotation trajectory: fit SLERP over (key_times, key_quats (N,4)) and sample at
    query_times (clamped to the key range). Returns (M, 4) quaternions. Temporal-consistency checks in
    CALYX flag query points whose interpolated rotation disagrees with the observed one."""
    kt = np.asarray(key_times, dtype=np.float64)
    if kt.ndim != 1 or len(kt) < 2:
        raise ValueError("need at least two key times")
    interp = Slerp(kt, Rotation.from_quat(np.asarray(key_quats, dtype=np.float64)))
    qt = np.clip(np.asarray(query_times, dtype=np.float64), kt[0], kt[-1])
    return interp(qt).as_quat()


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotations (each a 3x3 matrix or (x,y,z,w) quaternion)."""
    def _rot(x):
        x = np.asarray(x, dtype=np.float64)
        return Rotation.from_matrix(x) if x.shape == (3, 3) else Rotation.from_quat(x)
    rel = _rot(r_a).inv() * _rot(r_b)
    return float(np.degrees(rel.magnitude()))


# --------------------------------------------------------------------------- projection / back-projection

def project_points(points_cam: np.ndarray, K: np.ndarray, dist: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) camera-frame points to (N, 2) pixels via a pinhole ``K`` with optional OpenCV distortion
    (k1, k2, p1, p2[, k3]). Returns (uv, valid) where ``valid`` marks points in front of the camera (z > 0)."""
    pts = _as_points(points_cam, 3)
    K = np.asarray(K, dtype=np.float64)
    z = pts[:, 2]
    valid = z > _EPS
    zc = np.where(valid, z, 1.0)
    x = pts[:, 0] / zc
    y = pts[:, 1] / zc
    if dist is not None:
        d = np.asarray(dist, dtype=np.float64).reshape(-1)
        k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
        k3 = d[4] if d.size > 4 else 0.0
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = x_d, y_d
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1), valid


def backproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project (N, 2) pixels at per-pixel ``depth`` (metres along +z) to (N, 3) camera-frame points."""
    px = _as_points(uv, 2)
    K = np.asarray(K, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    if z.shape[0] != px.shape[0]:
        raise ValueError("depth length must match number of pixels")
    x = (px[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (px[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def reprojection_error(points_world: np.ndarray, uv_obs: np.ndarray, T_cam_world: np.ndarray,
                       K: np.ndarray, dist: np.ndarray | None = None) -> dict:
    """Reproject world points through ``T_cam_world`` (world expressed in the camera) and compare to observed
    pixels. Returns per-point residual pixels and summary stats. This is the core CALYX residual."""
    pts_cam = transform_points(T_cam_world, points_world)
    uv_proj, valid = project_points(pts_cam, K, dist)
    obs = _as_points(uv_obs, 2)
    resid = np.linalg.norm(uv_proj - obs, axis=1)
    resid_valid = resid[valid]
    return {
        "residuals_px": resid.tolist(),
        "valid": valid.tolist(),
        "rms_px": float(np.sqrt(np.mean(resid_valid ** 2))) if resid_valid.size else None,
        "max_px": float(resid_valid.max()) if resid_valid.size else None,
        "n_valid": int(valid.sum()),
    }


# --------------------------------------------------------------------------- epipolar

def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Essential matrix ``E = [t]_x R`` for the relative pose (R, t) from camera 1 to camera 2."""
    return _skew(np.asarray(t, dtype=np.float64).reshape(3)) @ np.asarray(R, dtype=np.float64)


def fundamental_matrix(R: np.ndarray, t: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Fundamental matrix ``F = K2^-T E K1^-1`` relating pixels in image 1 to epipolar lines in image 2."""
    E = essential_matrix(R, t)
    return np.linalg.inv(np.asarray(K2, dtype=np.float64)).T @ E @ np.linalg.inv(np.asarray(K1, dtype=np.float64))


def sampson_distance(pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """First-order geometric (Sampson) epipolar distance per correspondence, in pixels. Near zero when the two
    pixels are consistent with the fundamental matrix; the camera-only CALYX consistency check."""
    p1 = _as_points(pts1, 2)
    p2 = _as_points(pts2, 2)
    x1 = np.hstack([p1, np.ones((p1.shape[0], 1))])
    x2 = np.hstack([p2, np.ones((p2.shape[0], 1))])
    F = np.asarray(F, dtype=np.float64)
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.abs(x2Fx1) / np.sqrt(np.maximum(denom, _EPS))


def epipolar_residual(pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, t: np.ndarray,
                      K1: np.ndarray, K2: np.ndarray) -> dict:
    """Sampson epipolar residual summary for feature correspondences across two calibrated cameras."""
    F = fundamental_matrix(R, t, K1, K2)
    d = sampson_distance(pts1, pts2, F)
    return {
        "sampson_px": d.tolist(),
        "rms_px": float(np.sqrt(np.mean(d ** 2))) if d.size else None,
        "median_px": float(np.median(d)) if d.size else None,
        "n": int(d.size),
    }
