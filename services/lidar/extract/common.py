"""Shared helpers for static scene element extraction: load a cloud with its Phase 2 segmentation and the
Phase 1 ground plane, split ground from non-ground, cluster non-ground points (open3d DBSCAN), and summarize
a cluster's geometry. The extractors (poles, road edges, buildings, vegetation, markings) build on these.
"""

from __future__ import annotations

import uuid

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from db.models import PointCloud, PointSegmentation
from db.session import get_sessionmaker
from services.lidar.clean.ground import segment_ground
from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.store import load_cloud
from services.lidar.segment3d.run import load_segmentation

log = get_logger("lidar_extract")


def height_above_plane(xyz: np.ndarray, plane: list[float]) -> np.ndarray:
    a, b, c, d = plane
    if abs(c) < 1e-6:
        return xyz[:, 2]
    return xyz[:, 2] - (-(a * xyz[:, 0] + b * xyz[:, 1] + d) / c)


def nonground_mask(cloud: Cloud, plane: list[float], thresh: float = 0.2) -> np.ndarray:
    """Points more than `thresh` metres above the ground plane."""
    return height_above_plane(cloud.xyz, plane) > thresh


def cluster_dbscan(xyz: np.ndarray, eps: float | None = None, min_points: int | None = None) -> np.ndarray:
    """Per-point cluster label via open3d DBSCAN; -1 is noise. Empty input returns an empty array."""
    cfg = get_settings().lidar
    eps = eps if eps is not None else cfg.extract_cluster_eps
    min_points = min_points if min_points is not None else cfg.extract_cluster_min_points
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int32)
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_points), dtype=np.int32)


def cluster_stats(xyz: np.ndarray, plane: list[float] | None = None) -> dict:
    """Geometry summary of a cluster: centroid, footprint extent in BEV, height, base z, and verticality."""
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    footprint = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    footprint_min = float(min(hi[0] - lo[0], hi[1] - lo[1]))
    if plane is not None:
        h = height_above_plane(xyz, plane)
        height = float(h.max() - h.min())
        base_z = float(xyz[h.argmin(), 2])
    else:
        height = float(hi[2] - lo[2])
        base_z = float(lo[2])
    return {"centroid": [float(x) for x in xyz.mean(axis=0)], "footprint": footprint,
            "footprint_min": footprint_min, "height": height, "base_z": base_z, "n": int(len(xyz)),
            "bounds_min": [float(x) for x in lo], "bounds_max": [float(x) for x in hi]}


async def load_for_extraction(cloud_id: uuid.UUID) -> dict | None:
    """Load a cloud, its latest segmentation (semantic + instance arrays), the ground plane, and the
    provenance (calibration version and the synchronized camera frames at the cloud's ts_ns)."""
    from sqlalchemy import select

    from db.models import Frame
    async with get_sessionmaker()() as db:
        pc = await db.get(PointCloud, cloud_id)
        if pc is None:
            return None
        seg = (await db.execute(select(PointSegmentation).where(PointSegmentation.cloud_id == cloud_id)
               .order_by(PointSegmentation.created_at.desc()).limit(1))).scalar_one_or_none()
        frame_ids = [str(f) for f in (await db.execute(select(Frame.frame_id).where(
            Frame.session_id == pc.session_id, Frame.ts_ns == pc.ts_ns))).scalars().all()]
        session_id, cloud_uri, calib = pc.session_id, pc.cloud_uri, pc.calibration_version
    cloud = load_cloud(cloud_uri)
    _, plane, _ = segment_ground(cloud)
    semantic = None
    if seg is not None:
        labels = load_segmentation(seg.labels_uri)
        if len(labels.get("semantic", [])) == cloud.n:
            semantic = np.asarray(labels["semantic"])
    return {"cloud": cloud, "plane": plane, "semantic": semantic, "session_id": session_id,
            "calibration_version": calib, "frame_ids": frame_ids}
