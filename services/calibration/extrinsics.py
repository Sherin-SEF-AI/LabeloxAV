"""Extrinsics validation (M3.0): camera-to-camera consistency by epipolar (Sampson) geometry on
overlapping views, and camera-to-IMU / camera-to-GNSS extrinsics by motion residuals. Pure geometry, so
it is unit-testable with constructed correspondences and motions (the rig/IMU streams are not yet ingested).
"""

from __future__ import annotations

import numpy as np


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)


def epipolar_residual(pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, t: np.ndarray,
                      K1: np.ndarray, K2: np.ndarray) -> dict:
    """Mean Sampson distance (px) for point correspondences under the relative pose (R, t). Low means the
    two cameras' extrinsics are consistent with the observed geometry."""
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    F = np.linalg.inv(K2).T @ (_skew(np.asarray(t, np.float64).ravel()) @ np.asarray(R, np.float64)) @ np.linalg.inv(K1)
    res = []
    for (x1, x2) in zip(pts1, pts2):
        p1 = np.array([x1[0], x1[1], 1.0])
        p2 = np.array([x2[0], x2[1], 1.0])
        Fp1, Ftp2 = F @ p1, F.T @ p2
        den = Fp1[0] ** 2 + Fp1[1] ** 2 + Ftp2[0] ** 2 + Ftp2[1] ** 2
        res.append((p2 @ F @ p1) ** 2 / den if den > 0 else 0.0)
    return {"mean_sampson_px": float(np.sqrt(np.mean(res))) if res else None, "n": len(pts1)}


def imu_motion_residual(cam_rot_deg: float, imu_rot_deg: float) -> dict:
    """Residual between the rotation a camera observes over a window and the IMU-integrated rotation. A
    large residual means a bad camera-to-IMU extrinsic or time offset."""
    return {"cam_rot_deg": round(cam_rot_deg, 2), "imu_rot_deg": round(imu_rot_deg, 2),
            "residual_deg": round(abs(cam_rot_deg - imu_rot_deg), 2)}


def gnss_lever_arm_residual(cam_translation_m: float, gnss_translation_m: float) -> dict:
    """Residual between camera-derived and GNSS-derived translation over a window (lever-arm check)."""
    return {"cam_m": round(cam_translation_m, 3), "gnss_m": round(gnss_translation_m, 3),
            "residual_m": round(abs(cam_translation_m - gnss_translation_m), 3)}
