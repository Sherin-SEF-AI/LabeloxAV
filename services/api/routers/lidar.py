"""LiDAR BEV annotation endpoints. Lift the oriented boxes a human drew on a BEV frame to metric 3D
cuboids (stored on object.cuboid_3d), using the frame's stored BEV projection and its point cloud."""

from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Object
from services.api.deps import db_session
from services.lidar.bev import bev_box_to_cuboid

log = get_logger("api_lidar")
router = APIRouter()


@router.post("/lidar/cuboids/{frame_id}")
async def compute_cuboids(frame_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Convert every oriented box on a LiDAR BEV frame into a 3D cuboid (object.cuboid_3d). The z extent
    is taken from the points each box encloses, so the cuboids are data-driven, not assumed."""
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    if not frame.lidar or not frame.lidar.get("pcd_uri") or not frame.lidar.get("bev"):
        raise HTTPException(400, "not a LiDAR BEV frame")

    pts = np.frombuffer(get_object_store().get_bytes(frame.lidar["pcd_uri"]), dtype=np.float32).reshape(-1, 4)
    bev = frame.lidar["bev"]
    objects = (await db.execute(select(Object).where(Object.frame_id == frame_id))).scalars().all()
    out = []
    for o in objects:
        cub = bev_box_to_cuboid(list(o.bbox), float(o.rot_deg or 0.0), pts, bev)
        o.cuboid_3d = cub
        out.append({"object_id": str(o.object_id), "cuboid_3d": cub})
    await db.commit()
    log.info("lidar.cuboids", frame=str(frame_id), n=len(out))
    return {"frame_id": str(frame_id), "cuboids": len(out), "objects": out}
