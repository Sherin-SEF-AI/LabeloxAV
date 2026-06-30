"""Adverse-condition region tagging. A frame can carry several polygon regions, each labelled with the
condition affecting it (glare, reflection, shadow, rain, fog, lowlight), so downstream models know which
pixels to distrust. Frame-level and multi-region, unlike the single drivable mask.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AdverseRegion, Frame
from services.api.deps import db_session, require_role

router = APIRouter()

_CONDITIONS = {"glare", "reflection", "shadow", "rain", "fog", "lowlight"}


class AdverseIn(BaseModel):
    geometry: list[float]    # polygon, flattened [x,y,x,y,...] image pixels
    condition: str
    confidence: float = 1.0


def _row(r: AdverseRegion) -> dict:
    return {"region_id": str(r.region_id), "frame_id": str(r.frame_id), "geometry": r.geometry,
            "condition": r.condition, "source": r.source, "confidence": r.confidence}


@router.post("/frames/{frame_id}/adverse", dependencies=[Depends(require_role("annotator"))])
async def create_adverse(frame_id: str, payload: AdverseIn, db: AsyncSession = Depends(db_session)):
    if payload.condition not in _CONDITIONS:
        raise HTTPException(400, f"unknown condition '{payload.condition}'")
    if len(payload.geometry) < 6:
        raise HTTPException(400, "geometry needs at least 3 points")
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    r = AdverseRegion(frame_id=frame.frame_id, geometry=payload.geometry, condition=payload.condition,
                      source="human", confidence=payload.confidence)
    db.add(r)
    await db.commit()
    return _row(r)


@router.get("/frames/{frame_id}/adverse")
async def list_adverse(frame_id: str, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(AdverseRegion)
            .where(AdverseRegion.frame_id == UUID(frame_id)))).scalars().all()
    return [_row(r) for r in rows]


@router.delete("/adverse/{region_id}", dependencies=[Depends(require_role("annotator"))])
async def delete_adverse(region_id: str, db: AsyncSession = Depends(db_session)):
    r = await db.get(AdverseRegion, UUID(region_id))
    if r is not None:
        await db.delete(r)
        await db.commit()
    return {"deleted": region_id}
