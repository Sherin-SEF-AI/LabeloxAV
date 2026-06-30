"""Scan alignment by ICP, NDT, or GICP (open3d), using the GNSS and IMU as the initial pose prior. Pseudo-
LiDAR registration is noisier than real LiDAR, so a low registration fitness is flagged rather than trusted.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("lidar_register")


def gnss_imu_prior(d_forward: float, d_left: float, d_yaw: float = 0.0, d_up: float = 0.0) -> np.ndarray:
    """A 4x4 ego-frame relative transform (x forward, y left, z up) from the forward/left displacement and the
    yaw change between two scans. The caller rotates the GNSS ENU delta into the ego frame by the heading
    first, so the translation is expressed in the cloud's own frame, not ENU."""
    c, s = np.cos(d_yaw), np.sin(d_yaw)
    t = np.eye(4)
    t[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t[:3, 3] = [d_forward, d_left, d_up]
    return t


def register_pair(source_xyz: np.ndarray, target_xyz: np.ndarray, init: np.ndarray | None = None,
                  method: str | None = None, voxel: float | None = None) -> dict:
    """Align source to target. Returns the 4x4 transform, the fitness and inlier RMSE, and a low-confidence
    flag when the fitness is below the configured threshold."""
    import open3d as o3d

    cfg = get_settings().lidar
    method = (method or cfg.register_method).lower()
    voxel = voxel if voxel is not None else cfg.register_voxel_m

    def _prep(xyz):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
        return pc

    src, tgt = _prep(source_xyz), _prep(target_xyz)
    init = np.asarray(init, dtype=np.float64) if init is not None else np.eye(4)
    max_corr = voxel * 2.0
    reg_mod = o3d.pipelines.registration
    if method == "gicp":
        reg = reg_mod.registration_generalized_icp(src, tgt, max_corr, init)
    elif method == "ndt":
        # open3d has no NDT; a coarse-to-fine point-to-plane ICP is the equivalent voxel-grid alignment
        reg = reg_mod.registration_icp(src, tgt, max_corr * 2, init,
                                       reg_mod.TransformationEstimationPointToPlane())
        reg = reg_mod.registration_icp(src, tgt, max_corr, reg.transformation,
                                       reg_mod.TransformationEstimationPointToPlane())
    else:  # icp point-to-plane
        reg = reg_mod.registration_icp(src, tgt, max_corr, init,
                                       reg_mod.TransformationEstimationPointToPlane())
    low_conf = reg.fitness < cfg.register_min_fitness
    out = {"transformation": np.asarray(reg.transformation).tolist(), "fitness": round(float(reg.fitness), 4),
           "rmse": round(float(reg.inlier_rmse), 4), "method": method, "low_confidence": bool(low_conf)}
    if low_conf:
        log.info("lidar.register_low_confidence", fitness=out["fitness"], method=method)
    return out


def accumulate_poses(transforms: list[np.ndarray]) -> list[np.ndarray]:
    """Chain relative scan-to-scan transforms into absolute poses in the first scan's frame."""
    poses = [np.eye(4)]
    for t in transforms:
        poses.append(poses[-1] @ np.asarray(t, dtype=np.float64))
    return poses
