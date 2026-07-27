"""Calibration residual field (Tier 3). Given a calibration estimate, compute the per-feature reprojection
residual across the whole rig in parallel: project each world point through each camera's pose + intrinsics +
distortion and compare to the observed pixel. A dense residual field is what actually QAs a calibration, far
more than a single scalar RMS, and it is embarrassingly parallel, so it reuses the fused projection kernel over
all points x all cameras in one pass.

reprojection_residuals returns the per-(camera, point) residual in pixels plus a per-camera summary (RMS, max,
p95, median). It matches core.geometry.reprojection_error per camera and runs on the GPU through the projection
kernel when a CUDA device is present, else the identical NumPy math."""

from __future__ import annotations

import numpy as np

from core.accel.projection import (  # noqa: F401 (re-export gpu_available)
    gpu_available,
    project_world_batch,
)


def reprojection_residuals(points_world, uv_obs, T_cam_world, K, dist=None, model="pinhole", z_min=1e-9,
                           device=None) -> dict:
    """Per-(camera, point) reprojection residual across a rig.

    points_world (M, 3); uv_obs (C, M, 2) observed pixels per camera; T_cam_world (C, 4, 4) world->camera;
    K (C, 3, 3); dist (C, D) or None; model "pinhole"|"fisheye". Returns {residuals (C, M) pixels, valid (C, M)
    in-front mask, per_camera [{rms_px, max_px, p95_px, median_px, n_valid}]}. Only in-front points count toward
    the summaries (a point behind a camera has no meaningful reprojection)."""
    uv_obs = np.asarray(uv_obs, dtype=np.float64)
    if uv_obs.ndim == 2:                                  # (M, 2) single camera -> (1, M, 2)
        uv_obs = uv_obs[None]
    uv_proj, valid, _ = project_world_batch(points_world, T_cam_world, K, dist, model=model, z_min=z_min,
                                            device=device)
    resid = np.linalg.norm(uv_proj - uv_obs, axis=2)      # (C, M)

    per_camera = []
    for c in range(resid.shape[0]):
        rv = resid[c][valid[c]]
        if rv.size:
            per_camera.append({
                "rms_px": float(np.sqrt(np.mean(rv ** 2))),
                "max_px": float(rv.max()),
                "p95_px": float(np.percentile(rv, 95)),
                "median_px": float(np.median(rv)),
                "n_valid": int(rv.size),
            })
        else:
            per_camera.append({"rms_px": None, "max_px": None, "p95_px": None, "median_px": None, "n_valid": 0})
    return {"residuals": resid, "valid": valid, "per_camera": per_camera}
