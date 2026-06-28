"""Per-object dynamics endpoints (P3): compute the derived motion state for a session, and read it back
per object or per frame (the behavioral readout the editor shows)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ObjectDynamics
from services.api.deps import db_session

router = APIRouter()


def _row(d: ObjectDynamics) -> dict:
    return {"object_id": str(d.object_id), "track_id": str(d.track_id) if d.track_id else None,
            "distance_m": d.distance_m, "lateral_m": d.lateral_m, "speed_kmh": d.speed_kmh,
            "closing_speed_kmh": d.closing_speed_kmh, "heading_deg": d.heading_deg, "ttc_s": d.ttc_s,
            "risk_level": d.risk_level, "method": d.method, "confidence": d.confidence}


@router.post("/dynamics/compute")
async def compute(session_id: str):
    from services.dynamics.compute import compute_session_dynamics

    return await compute_session_dynamics(UUID(session_id))


@router.get("/dynamics/object/{object_id}")
async def get_object(object_id: UUID, db: AsyncSession = Depends(db_session)):
    d = await db.get(ObjectDynamics, object_id)
    return _row(d) if d else {"object_id": str(object_id), "computed": False}


@router.get("/dynamics/frame/{frame_id}")
async def get_frame(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(
        select(ObjectDynamics).where(ObjectDynamics.frame_id == frame_id))).scalars().all()
    return {"frame_id": str(frame_id), "dynamics": [_row(d) for d in rows]}
