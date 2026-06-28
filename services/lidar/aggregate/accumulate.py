"""Combine aligned scans across drives and vehicles into a dense aggregated map: transform each scan by its
optimized pose into a common frame, concatenate, and voxel-downsample so the map stays manageable. This is
the dense cloud the Swarm Cartography HD map fusion consumes.
"""

from __future__ import annotations

import numpy as np

from services.lidar.ingest.normalize import Cloud


def _voxel_downsample(xyz: np.ndarray, intensity: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    colors = np.repeat(intensity.reshape(-1, 1), 3, axis=1).astype(np.float64)
    pc.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    down = pc.voxel_down_sample(voxel)
    return np.asarray(down.points, dtype=np.float32), np.asarray(down.colors, dtype=np.float32)[:, 0]


def transform_cloud(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4x4 pose to points."""
    h = np.c_[xyz, np.ones(len(xyz))]
    return (np.asarray(pose, dtype=np.float64) @ h.T).T[:, :3]


def accumulate_scans(clouds: list[Cloud], poses: list[np.ndarray], voxel: float = 0.2) -> Cloud:
    """Transform every scan into the map frame and merge into one dense cloud, voxel-downsampled."""
    parts_xyz, parts_int = [], []
    for cloud, pose in zip(clouds, poses, strict=False):
        parts_xyz.append(transform_cloud(cloud.xyz, pose))
        parts_int.append(cloud.intensity)
    xyz = np.vstack(parts_xyz).astype(np.float32)
    inten = np.concatenate(parts_int).astype(np.float32)
    if voxel and len(xyz):
        xyz, inten = _voxel_downsample(xyz, inten, voxel)
    return Cloud(xyz=xyz, intensity=inten, ts_ns=clouds[0].ts_ns if clouds else 0,
                 source=clouds[0].source if clouds else "lidar", frame="map")
