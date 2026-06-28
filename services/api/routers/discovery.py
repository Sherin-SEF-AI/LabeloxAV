"""Rare-scenario discovery queue (M1.5): trigger discovery for a session, list pending candidates for
human review, and confirm/dismiss/tag them. Feeds active learning and sellable rare slices."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, ScenarioCandidate
from db.models import Session as DbSession
from services.api.deps import db_session

router = APIRouter()


class CandidateStateIn(BaseModel):
    state: str            # confirmed | dismissed | pending
    tag: str | None = None


@router.post("/discovery/run")
async def run(session_id: str):
    from services.intelligence.discovery import discover_session

    return await discover_session(UUID(session_id))


@router.get("/discovery/queue")
async def queue(state: str = "pending", limit: int = 200, db: AsyncSession = Depends(db_session)):
    limit = min(max(limit, 1), 1000)
    rows = (await db.execute(
        select(ScenarioCandidate, Frame.session_id, DbSession.vehicle_id)
        .join(Frame, Frame.frame_id == ScenarioCandidate.frame_id)
        .join(DbSession, DbSession.session_id == ScenarioCandidate.session_id)
        .where(ScenarioCandidate.state == state)
        .order_by(ScenarioCandidate.score.desc()).limit(limit))).all()
    return [{
        "candidate_id": str(c.candidate_id), "frame_id": str(c.frame_id), "session_id": str(c.session_id),
        "vehicle_id": veh, "kind": c.kind, "score": c.score, "cluster_id": c.cluster_id,
        "rare_classes": c.rare_classes or [], "state": c.state, "tag": c.tag,
        "image_url": f"/api/frames/{c.frame_id}/image",
    } for c, _, veh in rows]


@router.post("/discovery/{candidate_id}/state")
async def set_state(candidate_id: UUID, body: CandidateStateIn, db: AsyncSession = Depends(db_session)):
    cand = await db.get(ScenarioCandidate, candidate_id)
    if cand is None:
        raise HTTPException(404, "candidate not found")
    cand.state = body.state
    if body.tag is not None:
        cand.tag = body.tag
    await db.commit()
    return {"candidate_id": str(candidate_id), "state": cand.state, "tag": cand.tag}
