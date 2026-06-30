"""LiDAR-to-camera calibration validation, extending the Phase 3 calibration service to the 3D sensor.

The reprojection residual (matched 3D points versus their observed pixels, from a target or tracked features)
measures the extrinsic and intrinsic quality. The coverage consistency is the always-available sanity signal
that the cloud and the camera see the same scene. A residual above the threshold, or one that has grown
versus the baseline, points to a mismatched or drifted calibration.
"""

from __future__ import annotations

import numpy as np

from services.lidar.ingest.normalize import Cloud
from services.lidar.project import project_to_camera


def reprojection_error(points_ego, observed_uv, cam_id: str, img_w: int, img_h: int) -> dict:
    """RMS and max pixel error between projected 3D points and their observed pixels (only points in front
    of the camera count). The core LiDAR-camera calibration residual."""
    pts = np.asarray(points_ego, dtype=np.float32).reshape(-1, 3)
    obs = np.asarray(observed_uv, dtype=np.float32).reshape(-1, 2)
    proj = project_to_camera(pts, cam_id, img_w, img_h)
    vis = proj["in_front"]
    if not np.any(vis):
        return {"rms": None, "max": None, "n": 0}
    err = np.linalg.norm(proj["uv"][vis] - obs[vis], axis=1)
    return {"rms": float(np.sqrt(np.mean(err ** 2))), "max": float(err.max()), "n": int(err.size)}


def coverage_consistency(cloud: Cloud, cam_id: str, img_w: int, img_h: int) -> dict:
    """How much of the cloud the camera sees: the in-front and in-image fractions. A cloud and camera that
    are consistently calibrated share a large in-image overlap for the forward field of view."""
    if cloud.n == 0:
        return {"in_front_frac": 0.0, "in_image_frac": 0.0}
    proj = project_to_camera(cloud.xyz, cam_id, img_w, img_h)
    return {"in_front_frac": round(float(proj["in_front"].mean()), 4),
            "in_image_frac": round(float(proj["in_image"].mean()), 4)}
