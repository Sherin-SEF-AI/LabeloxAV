"""LiDAR Phase 3 endpoints: static scene element extraction (feeding the HD map), 3D traversability,
multi-scan aggregation, the 3D quality checker, and the 3D data product export. Interactive review and
configuration are local; heavy work dispatches to the lidar_aggregate burst seam.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import StaticElement
from services.api.deps import db_session

log = get_logger("api_lidar_scene")
router = APIRouter()


class AggregateIn(BaseModel):
    session_ids: list[uuid.UUID]
    region: str | None = None
    compute_target: str = "local"


@router.post("/lidar/aggregate")
async def aggregate(body: AggregateIn):
    """Register, loop-close, and accumulate the clouds across sessions into a dense map (M-L3.2). Local on the
    box; the A100 burst is the seam for large volumes."""
    from compute.worker.jobs.lidar_aggregate import LidarAggregateSpec, run_lidar_aggregate
    return await run_lidar_aggregate(LidarAggregateSpec(session_ids=body.session_ids, region=body.region,
                                                        compute_target=body.compute_target))


@router.post("/lidar/clouds/{cloud_id}/extract")
async def extract_static(cloud_id: uuid.UUID):
    """Extract static scene elements (poles, road edges, buildings, vegetation, markings) from a cloud and
    feed them to the HD map (M-L3.0)."""
    from services.lidar.extract import extract_cloud
    res = await extract_cloud(cloud_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.post("/lidar/clouds/{cloud_id}/traverse")
async def traverse(cloud_id: uuid.UUID):
    """Produce the 3D free-space grid, metric drivable surface, road-surface class, and elevation profile
    for a cloud (M-L3.1)."""
    from services.lidar.traverse import traverse_cloud
    res = await traverse_cloud(cloud_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.post("/lidar/clouds/{cloud_id}/quality3d")
async def quality3d(cloud_id: uuid.UUID):
    """Run the mandatory 3D quality checker over a cloud's objects (floating, below-ground, impossible dims,
    duplicate, misaligned, missing neighbour) and write quality_flag_3d rows (M-L3.3)."""
    from services.lidar.quality3d import check_cloud
    res = await check_cloud(cloud_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.post("/lidar/quality3d/{flag_id}/confirm")
async def confirm_quality3d(flag_id: uuid.UUID):
    """Confirm a 3D quality flag: demote the flagged object back to review (the same loop as 2D)."""
    from services.lidar.quality3d import confirm_flag
    res = await confirm_flag(flag_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.get("/lidar/clouds/{cloud_id}/quality3d")
async def list_quality3d(cloud_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from db.models import QualityFlag3D
    rows = (await db.execute(select(QualityFlag3D).where(QualityFlag3D.cloud_id == cloud_id)
            .order_by(QualityFlag3D.score.desc()))).scalars().all()
    return {"cloud_id": str(cloud_id), "flags": [{"flag_id": str(f.flag_id), "kind": f.kind, "score": f.score,
            "status": f.status, "object_3d_id": str(f.object_3d_id) if f.object_3d_id else None,
            "detail": f.detail} for f in rows]}


@router.post("/lidar/sessions/{session_id}/scene3d")
async def scene3d(session_id: uuid.UUID):
    """Classify the 3D structure (tunnel/flyover/open) of a session and merge it into Frame.scene (M-L3.3)."""
    from services.lidar.quality3d import classify_session_3d
    return await classify_session_3d(session_id)


@router.post("/lidar/sessions/{session_id}/rare3d")
async def rare3d(session_id: uuid.UUID):
    """Mine 3D rare cues (flooded road, animal, emergency vehicle, debris) into ScenarioCandidate rows."""
    from services.lidar.quality3d import mine_session_3d
    return await mine_session_3d(session_id)


@router.get("/lidar/sessions/{session_id}/static_elements")
async def list_static_elements(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(StaticElement).where(StaticElement.session_id == session_id)
            .order_by(StaticElement.kind))).scalars().all()
    return {"session_id": str(session_id), "count": len(rows),
            "elements": [{"element_id": str(e.element_id), "kind": e.kind, "attrs": e.attrs,
                          "confidence": e.confidence, "method": e.method,
                          "map_element_id": str(e.map_element_id) if e.map_element_id else None} for e in rows]}
