"""LiDAR endpoints: the BEV cuboid lift (oriented boxes drawn on a BEV frame to metric 3D cuboids) plus the
3D viewer data plane (list a session's clouds, cloud metadata, and a packed binary point stream the browser
renders with three.js)."""

from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compute.worker.jobs.pointcloud_build import (
    PointCloudBuildSpec,
    _frame_groups,
    build_session_clouds,
)
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Object, PointCloud, PointCloudDerived
from services.api.deps import db_session
from services.lidar.bev import bev_box_to_cuboid
from services.lidar.ingest.store import load_cloud

log = get_logger("api_lidar")
router = APIRouter()


class BuildIn(BaseModel):
    limit: int = 1                 # synchronized frame groups to build in this request
    stride: int | None = None
    max_points: int | None = None


@router.post("/lidar/sessions/{session_id}/build")
async def build_clouds(session_id: uuid.UUID, body: BuildIn | None = None,
                       db: AsyncSession = Depends(db_session)):
    """Build pseudo-LiDAR clouds from a session's camera frames on the local 5080. Bounded to `limit` frame
    groups so the request stays interactive; bulk volume runs through the pointcloud_build burst job."""
    body = body or BuildIn()
    groups = await _frame_groups(session_id, None)
    if not groups:
        raise HTTPException(400, "no camera frame groups in this session")
    ts = [g[0] for g in groups[: max(1, body.limit)]]
    spec = PointCloudBuildSpec(session_id=session_id, ts_ns=ts, stride=body.stride, max_points=body.max_points)
    res = await build_session_clouds(spec)
    res["groups_total"] = len(groups)
    return res


@router.get("/lidar/sessions/{session_id}/clouds")
async def list_clouds(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Every point cloud in a session with its derived variants, for the viewer's cloud picker."""
    rows = (await db.execute(select(PointCloud).where(PointCloud.session_id == session_id)
                             .order_by(PointCloud.ts_ns))).scalars().all()
    out = []
    for r in rows:
        variants = (await db.execute(select(PointCloudDerived.kind)
                                     .where(PointCloudDerived.cloud_id == r.cloud_id))).scalars().all()
        out.append({"cloud_id": str(r.cloud_id), "ts_ns": r.ts_ns, "source": r.source,
                    "point_count": r.point_count, "depth_model": r.depth_model, "bounds": r.bounds,
                    "variants": ["raw", *sorted(set(variants))]})
    return {"session_id": str(session_id), "clouds": out}


async def _resolve_cloud(db: AsyncSession, cloud_id: uuid.UUID, variant: str | None):
    pc = await db.get(PointCloud, cloud_id)
    if pc is None:
        raise HTTPException(404, "cloud not found")
    if variant in (None, "", "raw"):
        return pc.cloud_uri, pc
    d = (await db.execute(select(PointCloudDerived).where(PointCloudDerived.cloud_id == cloud_id,
         PointCloudDerived.kind == variant).order_by(PointCloudDerived.created_at.desc()).limit(1))
         ).scalar_one_or_none()
    if d is None:
        raise HTTPException(404, f"variant '{variant}' not found for this cloud")
    return d.uri, pc


@router.get("/lidar/clouds/{cloud_id}")
async def cloud_meta(cloud_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    pc = await db.get(PointCloud, cloud_id)
    if pc is None:
        raise HTTPException(404, "cloud not found")
    variants = (await db.execute(select(PointCloudDerived.kind)
                                 .where(PointCloudDerived.cloud_id == cloud_id))).scalars().all()
    return {"cloud_id": str(pc.cloud_id), "session_id": str(pc.session_id), "ts_ns": pc.ts_ns,
            "source": pc.source, "point_count": pc.point_count, "depth_model": pc.depth_model,
            "calibration_version": pc.calibration_version, "bounds": pc.bounds,
            "variants": ["raw", *sorted(set(variants))]}


@router.get("/lidar/clouds/{cloud_id}/points")
async def cloud_points(cloud_id: uuid.UUID, variant: str | None = Query(None),
                       max_points: int = Query(400000, alias="max", ge=1000, le=5_000_000),
                       full: bool = Query(False), db: AsyncSession = Depends(db_session)):
    """Packed binary point stream for the browser: Float32 [x, y, z, intensity] interleaved, decimated to
    `max` unless `full`. The viewer reads it as one ArrayBuffer (no JSON parse of millions of numbers)."""
    uri, pc = await _resolve_cloud(db, cloud_id, variant)
    cloud = load_cloud(uri)
    decimated = not full and cloud.n > max_points
    if decimated:
        cloud = cloud.decimate(max_points, seed=pc.ts_ns % (2**31))
    packed = np.empty((cloud.n, 4), dtype=np.float32)
    packed[:, :3] = cloud.xyz
    packed[:, 3] = cloud.intensity
    imin = float(cloud.intensity.min()) if cloud.n else 0.0
    imax = float(cloud.intensity.max()) if cloud.n else 1.0
    return Response(content=packed.tobytes(), media_type="application/octet-stream",
                    headers={"X-Point-Count": str(cloud.n), "X-Source": cloud.source, "X-Frame": cloud.frame,
                             "X-Decimated": str(decimated), "X-Intensity-Min": f"{imin:.5f}",
                             "X-Intensity-Max": f"{imax:.5f}",
                             "Access-Control-Expose-Headers": "X-Point-Count,X-Source,X-Frame,X-Decimated,"
                                                              "X-Intensity-Min,X-Intensity-Max"})


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
