"""Flywheel endpoints: run a session through the plane gates automatically and trace a deployed model back to
its source sessions and their health and calibration state. This is the closed loop of the data engine.
Mounted under /api."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Session as DbSession
from orchestration.dag import STAGES, run_session, trace_lineage
from services.api.deps import db_session

router = APIRouter()


@router.get("/flywheel/stages")
async def stages():
    """The flywheel stage order, from the platform registry."""
    return {"stages": STAGES}


@router.post("/flywheel/session/{session_id}/run")
async def run(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Run the flywheel for a session: enforce the SANYX and CALYX gates, and report how far it proceeds."""
    if await db.get(DbSession, session_id) is None:
        raise HTTPException(404, "session not found")
    return await run_session(db, session_id)


@router.get("/flywheel/lineage/{deployment_id}")
async def lineage(deployment_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Resolve a deployed model all the way back to its source sessions and their gate state, one query."""
    return await trace_lineage(db, deployment_id)
