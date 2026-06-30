"""Extend the Phase 1 scene classification with 3D structure: tunnel, flyover, underpass, and dense-traffic
from the 3D occupancy. Writes a 3d_structure and 3d_density axis into the existing Frame.scene, so the
analytics and search reuse the same record.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select

from core.logging import get_logger
from db.models import Frame, Object3D, PointCloud
from db.session import get_sessionmaker
from services.lidar.clean.ground import segment_ground
from services.lidar.extract.common import height_above_plane
from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.store import load_cloud

log = get_logger("lidar_scene3d")


def classify_3d_structure(cloud: Cloud, plane: list[float], n_objects: int) -> dict:
    """3D structure from overhead returns and side walls over the ego corridor, plus density from objects."""
    h = height_above_plane(cloud.xyz, plane)
    fwd, lat = cloud.xyz[:, 0], cloud.xyz[:, 1]
    corridor = (fwd > 3) & (fwd < 25) & (np.abs(lat) < 3)
    n_corridor = int(corridor.sum())
    overhead_frac = float(((h > 4.0) & corridor).sum()) / max(n_corridor, 1)
    left_wall = int(((h > 1.0) & (lat > 3.5) & (fwd > 3) & (fwd < 25)).sum())
    right_wall = int(((h > 1.0) & (lat < -3.5) & (fwd > 3) & (fwd < 25)).sum())
    enclosed = left_wall > 50 and right_wall > 50

    if overhead_frac > 0.05:
        structure = "tunnel" if enclosed else "flyover"   # covered + walls = tunnel; covered + open = flyover/underpass
    else:
        structure = "open"
    density = "dense" if n_objects >= 8 else ("moderate" if n_objects >= 3 else "sparse")
    return {"3d_structure": structure, "3d_density": density, "overhead_frac": round(overhead_frac, 3),
            "n_objects": n_objects}


async def classify_session_3d(session_id: uuid.UUID) -> dict:
    """Classify the 3D structure of every cloud in a session and merge it into the synchronized Frame.scene."""
    async with get_sessionmaker()() as db:
        clouds = (await db.execute(select(PointCloud).where(PointCloud.session_id == session_id)
                  .order_by(PointCloud.ts_ns))).scalars().all()
    updated, by_structure = 0, {}
    for pc in clouds:
        cloud = load_cloud(pc.cloud_uri)
        _, plane, _ = segment_ground(cloud)
        async with get_sessionmaker()() as db:
            n_obj = (await db.execute(select(Object3D).where(Object3D.cloud_id == pc.cloud_id))).scalars().all()
            scene3d = classify_3d_structure(cloud, plane, len(n_obj))
            frame = (await db.execute(select(Frame).where(Frame.session_id == session_id,
                     Frame.ts_ns == pc.ts_ns).order_by(Frame.cam_id).limit(1))).scalar_one_or_none()
            if frame is not None:
                frame.scene = {**(frame.scene or {}), **scene3d}
                updated += 1
            await db.commit()
        by_structure[scene3d["3d_structure"]] = by_structure.get(scene3d["3d_structure"], 0) + 1
    log.info("lidar.scene3d", session=str(session_id), clouds=len(clouds), updated=updated, by=by_structure)
    return {"session_id": str(session_id), "clouds": len(clouds), "frames_updated": updated,
            "by_structure": by_structure}
