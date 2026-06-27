"""Ontology and session listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import OntologyClass
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
async def sessions(db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(DbSession).order_by(DbSession.created_at.desc()))).scalars().all()
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
