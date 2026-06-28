"""Ontology and session listing endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, Object, OntologyClass
from db.models import Session as DbSession
from services.api.deps import OntologyClassOut, db_session
from services.autolabel.ontology import add_custom_class, get_ontology

router = APIRouter()


class NewClassIn(BaseModel):
    name: str
    l0: str = "object"
    l1: str = "custom"
    india: bool = True


@router.post("/ontology/classes")
async def create_class(payload: NewClassIn, db: AsyncSession = Depends(db_session)):
    """Add an annotator-defined custom class. It lands in the custom id block, is marked rare so the gate
    routes it to human review, and is mirrored into the DB ontology table for the current version."""
    try:
        cls = add_custom_class(payload.name, payload.l0, payload.l1, payload.india)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    onto = get_ontology()
    if await db.get(OntologyClass, cls["id"]) is None:
        db.add(OntologyClass(id=cls["id"], version=onto.version, name=cls["name"], l0=cls["l0"],
                             l1=cls["l1"], india=cls["india"], map_to={}))
        await db.commit()
    return cls


@router.get("/ontology")
async def ontology():
    onto = get_ontology()
    return {
        "version": onto.version,
        "hierarchy_levels": onto.hierarchy_levels,
        "attributes": {
            n: {"type": a.type, "values": a.values, "range": list(a.range) if a.range else None}
            for n, a in onto.attributes.items()
        },
        "classes": [OntologyClassOut(id=c.id, name=c.name, l0=c.l0, l1=c.l1, india=c.india).model_dump()
                    for c in sorted(onto.classes, key=lambda c: c.id)],
    }


@router.get("/sessions")
async def sessions(db: AsyncSession = Depends(db_session), limit: int = Query(200, ge=1, le=1000)):
    rows = (await db.execute(select(DbSession).order_by(DbSession.created_at.desc()).limit(limit))).scalars().all()
    return [
        {
            "session_id": str(s.session_id),
            "vehicle_id": s.vehicle_id,
            "city": s.city,
            "route": s.route,
            "start_ts_ns": s.start_ts_ns,
            "end_ts_ns": s.end_ts_ns,
            "ontology_version": s.ontology_version,
        }
        for s in rows
    ]


@router.get("/sessions/{session_id}/stats")
async def session_stats(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Per-session progress for the Open Annotation browser: frame count + object counts by state, and a
    progress fraction (auto-accepted + human-accepted over total)."""
    frames = (await db.execute(
        select(func.count()).select_from(Frame).where(Frame.session_id == session_id))).scalar_one()
    rows = (await db.execute(
        select(Object.state, func.count()).join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == session_id).group_by(Object.state))).all()
    by_state = {k: int(v) for k, v in rows}
    objects = sum(by_state.values())
    done = by_state.get("auto_accept", 0) + by_state.get("accepted", 0)
    return {"session_id": str(session_id), "frames": int(frames), "objects": objects,
            "by_state": by_state, "done": done,
            "progress": round(done / objects, 3) if objects else 0.0}


@router.get("/sessions/{session_id}/first-frame")
async def first_frame(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """The chronological first frame of a session, so a card can open it directly in the editor."""
    fid = (await db.execute(select(Frame.frame_id).where(Frame.session_id == session_id)
           .order_by(Frame.ts_ns.asc()).limit(1))).scalar_one_or_none()
    if fid is None:
        raise HTTPException(404, "no frames in session")
    return {"frame_id": str(fid)}
