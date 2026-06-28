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


@router.get("/lidar/sessions/{session_id}/static_elements")
async def list_static_elements(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(StaticElement).where(StaticElement.session_id == session_id)
            .order_by(StaticElement.kind))).scalars().all()
    return {"session_id": str(session_id), "count": len(rows),
            "elements": [{"element_id": str(e.element_id), "kind": e.kind, "attrs": e.attrs,
                          "confidence": e.confidence, "method": e.method,
                          "map_element_id": str(e.map_element_id) if e.map_element_id else None} for e in rows]}
