"""CALYX extrinsic-drift estimation (M2). An uncalibrated rig silently corrupts every label and every model,
and nobody watches for it. CALYX watches: it estimates the calibration drift as an SE(3) delta between what
the cameras see (visual odometry, or LiDAR-camera reprojection when LiDAR is present) and what the IMU and
GNSS report, and turns the magnitude of that delta into a severity that flags or blocks the session.

The estimator is a rigid (Kabsch) alignment: given corresponding points or trajectory positions in the two
frames, recover the SE(3) that best maps one onto the other. When the two streams are consistent that SE(3)
is identity; a drifted mount shows up as a non-trivial rotation or translation. It reuses core/geometry for
the SE(3) magnitude and composition. Pure, so a synthetic perturbation is recovered deterministically in the
test (the M2 acceptance)."""

from __future__ import annotations

import numpy as np

from core.config import CalyxSettings
from core.geometry import se3, se3_magnitude


def rigid_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch: the SE(3) pose T minimizing ||T . src - dst|| over corresponding (N,3) point sets. Returns
    (T 4x4, rms residual). Reflection is corrected so T is a proper rotation."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3 or src.shape != dst.shape:
        raise ValueError("need matching point sets with at least 3 correspondences")
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dc - R @ sc
    T = se3(R, t)
    resid = float(np.sqrt(np.mean(np.linalg.norm((src @ R.T + t) - dst, axis=1) ** 2)))
    return T, resid


def severity(mag: dict, cfg: CalyxSettings) -> str:
    """Map an SE(3) drift magnitude to {ok, drift_detected, block}."""
    if mag["rotation_deg"] >= cfg.rot_deg_block or mag["translation_m"] >= cfg.trans_m_block:
        return "block"
    if mag["rotation_deg"] >= cfg.rot_deg_flag or mag["translation_m"] >= cfg.trans_m_flag:
        return "drift_detected"
    return "ok"


def estimate_extrinsic_drift(cam_points: np.ndarray, inertial_points: np.ndarray, cfg: CalyxSettings) -> dict:
    """Estimate the SE(3) drift delta between the camera-observed geometry and the IMU/GNSS-reported geometry.
    cam_points and inertial_points are corresponding 3D positions (e.g. tracked-feature triangulations vs
    dead-reckoned positions) over a window. Returns the delta as a 4x4 matrix, its magnitude, a severity, and
    a confidence gated by the correspondence count."""
    cam = np.asarray(cam_points, dtype=np.float64).reshape(-1, 3)
    n = cam.shape[0]
    if n < cfg.min_correspondences:
        return {"delta": np.eye(4).tolist(), "magnitude": {"rotation_deg": 0.0, "translation_m": 0.0},
                "severity": "ok", "residual": None, "n": n, "confidence": "low",
                "detail": f"only {n} correspondences (< {cfg.min_correspondences}); drift not estimated"}
    T, resid = rigid_align(cam, inertial_points)
    mag = se3_magnitude(T)
    sev = severity(mag, cfg)
    return {"delta": T.tolist(), "magnitude": {k: round(v, 4) for k, v in mag.items()},
            "severity": sev, "residual": round(resid, 5), "n": n, "confidence": "ok",
            "detail": f"drift {mag['rotation_deg']:.2f} deg, {mag['translation_m']:.3f} m over {n} points"}


def temporal_consistency(rotations_deg: list[float], cfg: CalyxSettings) -> dict:
    """Flag rotation steps between adjacent frames that exceed what vehicle dynamics allow (a SLERP-interpolated
    rotation should be smooth; a discontinuity indicates a calibration or time-sync fault)."""
    steps = np.abs(np.diff(np.asarray(rotations_deg, dtype=np.float64))) if len(rotations_deg) > 1 else np.array([])
    bad = int(np.sum(steps > cfg.slerp_discontinuity_deg))
    return {"max_step_deg": float(steps.max()) if steps.size else 0.0, "discontinuities": bad,
            "ok": bad == 0}
