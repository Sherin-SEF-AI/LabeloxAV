"""The 3D quality checker, the 3D equivalent of the 2D quality reviewer. 3D detection on pseudo-LiDAR
produces floating boxes, below-ground boxes, and impossible dimensions; this layer catches them before they
pollute the dataset. Detects: floating (box bottom above the ground), below-ground (bottom below it),
impossible dimensions (a 50 m car), duplicate (overlapping cuboids on the same points), misaligned (box not
enclosing its points), and missing neighbour (dense points with no box). Flags route to the same review and
active-learning loop as 2D.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from db.models import Object3D, QualityFlag3D
from db.session import get_sessionmaker
from services.lidar.boxes import ground_z, iou_3d
from services.lidar.clean.ground import segment_ground
from services.lidar.extract.common import cluster_dbscan, height_above_plane
from services.lidar.ingest.store import load_cloud
from services.lidar.segment3d.semantic import points_in_cuboid

log = get_logger("lidar_quality3d")


def check_cuboid(cuboid: dict, cloud_xyz: np.ndarray, plane: list[float],
                 neighbors: list[dict] | None = None) -> list[dict]:
    """Return the quality flags for one cuboid. cuboid is {center, dims, yaw}."""
    cfg = get_settings().lidar
    cx, cy, cz = cuboid["center"]
    length, width, height = cuboid["dims"]
    flags: list[dict] = []

    if max(length, width, height) > cfg.quality_max_dim_m or min(length, width, height) <= 0.0:
        flags.append({"kind": "impossible_dims", "score": round(max(length, width, height) / cfg.quality_max_dim_m, 2),
                      "detail": {"dims": [length, width, height]}})

    gz = ground_z(plane, cx, cy)
    bottom = cz - height / 2.0
    gap = bottom - gz
    if gap > cfg.quality_float_gap_m:
        flags.append({"kind": "floating", "score": round(min(1.0, gap / 2.0), 2), "detail": {"gap_m": round(gap, 2)}})
    elif gap < -cfg.quality_below_ground_m:
        flags.append({"kind": "below_ground", "score": round(min(1.0, -gap / 2.0), 2),
                      "detail": {"below_m": round(-gap, 2)}})

    n_in = int(points_in_cuboid(cloud_xyz, cuboid).sum())
    if n_in < 8:
        flags.append({"kind": "misaligned", "score": round(1.0 - n_in / 8.0, 2), "detail": {"points_in_box": n_in}})

    for nb in neighbors or []:
        if iou_3d(cuboid, nb) > cfg.quality_duplicate_iou:
            flags.append({"kind": "duplicate", "score": round(iou_3d(cuboid, nb), 2),
                          "detail": {"other_object_3d_id": nb.get("object_3d_id")}})
            break
    return flags


def find_missing_neighbors(cloud_xyz: np.ndarray, plane: list[float], cuboids: list[dict],
                           min_cluster: int = 60) -> list[dict]:
    """Dense non-ground clusters with no cuboid over them: a likely missed object."""
    pts = cloud_xyz[height_above_plane(cloud_xyz, plane) > 0.3]
    if len(pts) < min_cluster:
        return []
    labels = cluster_dbscan(pts, eps=0.8)
    missing = []
    for cl in sorted(set(labels.tolist()) - {-1}):
        cxyz = pts[labels == cl]
        if len(cxyz) < min_cluster:
            continue
        centroid = cxyz.mean(axis=0)
        covered = any(points_in_cuboid(centroid[None, :], c)[0] for c in cuboids)
        if not covered:
            missing.append({"kind": "missing_neighbor", "score": round(min(1.0, len(cxyz) / 300.0), 2),
                            "detail": {"centroid": [round(float(x), 2) for x in centroid], "n_points": int(len(cxyz))}})
    return missing


async def check_cloud(cloud_id: uuid.UUID) -> dict:
    """Run the 3D quality checker over a cloud's objects and write quality_flag_3d rows."""
    async with get_sessionmaker()() as db:
        from db.models import PointCloud
        pc = await db.get(PointCloud, cloud_id)
        if pc is None:
            return {"error": "cloud not found"}
        objs = (await db.execute(select(Object3D).where(Object3D.cloud_id == cloud_id))).scalars().all()
        cloud_uri = pc.cloud_uri
    cloud = load_cloud(cloud_uri)
    _, plane, _ = segment_ground(cloud)
    cubs = [{"object_3d_id": str(o.object_3d_id), "center": o.center, "dims": o.dims, "yaw": o.yaw}
            for o in objs]

    written = 0
    async with get_sessionmaker()() as db:
        for i, o in enumerate(objs):
            neighbors = [c for j, c in enumerate(cubs) if j != i]
            for f in check_cuboid(cubs[i], cloud.xyz, plane, neighbors):
                db.add(QualityFlag3D(object_3d_id=o.object_3d_id, cloud_id=cloud_id, kind=f["kind"],
                                     score=f["score"], detail=f["detail"]))
                written += 1
        for f in find_missing_neighbors(cloud.xyz, plane, cubs):
            db.add(QualityFlag3D(object_3d_id=None, cloud_id=cloud_id, kind=f["kind"], score=f["score"],
                                 detail=f["detail"]))
            written += 1
        await db.commit()
    log.info("lidar.quality3d", cloud=str(cloud_id), objects=len(objs), flags=written)
    return {"cloud_id": str(cloud_id), "objects": len(objs), "flags": written}


async def confirm_flag(flag_id: uuid.UUID) -> dict:
    """Confirm a 3D quality flag: demote the flagged object back to review (the same loop as 2D)."""
    async with get_sessionmaker()() as db:
        flag = await db.get(QualityFlag3D, flag_id)
        if flag is None:
            return {"error": "flag not found"}
        flag.status = "confirmed"
        if flag.object_3d_id:
            o = await db.get(Object3D, flag.object_3d_id)
            if o is not None:
                o.state = "review"
                o.version += 1
        await db.commit()
    log.info("lidar.quality3d_confirm", flag=str(flag_id), kind=flag.kind)
    return {"flag_id": str(flag_id), "status": "confirmed", "kind": flag.kind}
