"""Sample drives: list them with their credit, and load one as a session (services/samples.py)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.observability import spawn
from db.models import ImportJob
from db.models import Session as DbSession
from services.api.deps import db_session, require_role
from services.samples import ROUTE_PREFIX, SAMPLES, get_sample, load_sample, public

router = APIRouter()


@router.get("/samples", dependencies=[Depends(require_role("annotator"))])
async def list_samples(db: AsyncSession = Depends(db_session)):
    """Every sample, its licence and credit, and whether it is already loaded (and as which session)."""
    rows = (await db.execute(
        select(DbSession.route, DbSession.session_id).where(DbSession.route.like(f"{ROUTE_PREFIX}%")))).all()
    loaded: dict[str, uuid.UUID] = {str(route): sid for route, sid in rows if route}
    return {"samples": [{**public(s), "session_id": str(loaded[f"{ROUTE_PREFIX}{s.id}"])
                         if f"{ROUTE_PREFIX}{s.id}" in loaded else None} for s in SAMPLES]}


@router.post("/samples/{sample_id}/load", dependencies=[Depends(require_role("annotator"))])
async def load(sample_id: str, db: AsyncSession = Depends(db_session)):
    """Fetch the sample and import it as a session. Returns the import job, which /jobs and
    GET /api/imports/{job_id} follow like any upload. A sample already loaded is not loaded twice."""
    s = get_sample(sample_id)
    if s is None:
        raise HTTPException(404, "no such sample")
    existing = (await db.execute(select(DbSession.session_id)
                                 .where(DbSession.route == f"{ROUTE_PREFIX}{s.id}").limit(1))).scalar_one_or_none()
    if existing:
        return {"status": "already_loaded", "session_id": str(existing)}
    job_id = uuid.uuid4()
    db.add(ImportJob(job_id=job_id, status="pending", format="video", source_uri=s.url,
                     target_vehicle=s.vehicle_id, city=s.city, progress=0.0, counts={"stage": "queued"}))
    await db.commit()
    spawn(load_sample(s, job_id), name="load_sample")
    return {"status": "pending", "job_id": str(job_id)}
