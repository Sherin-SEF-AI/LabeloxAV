"""Flywheel endpoints: run a session through the plane gates automatically and trace a deployed model back to
its source sessions and their health and calibration state. This is the closed loop of the data engine.
Mounted under /api."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Session as DbSession
from orchestration.dag import STAGES, run_session, trace_lineage
from services.api.deps import db_session
from services.flywheel.auto import run_auto_cycle
from services.flywheel.dispatch import dispatch_worklist
from services.flywheel.run import recent_cycles, run_and_record

router = APIRouter()


@router.post("/flywheel/adaptive/auto")
async def adaptive_auto(total_label_budget: int = 2000, safety_floor: int = 200,
                        min_share: float = 0.003, db: AsyncSession = Depends(db_session)):
    """Run one adaptive cycle from real corpus signals: class starvation (VRU/animal/two-wheeler classes below
    their floor) drives the label budget, ODD scene gaps drive collection, and the response carries the per-class
    work order of real candidate objects a reviewer would open. This closes the loop on live data."""
    return await run_auto_cycle(db, total_label_budget, safety_floor, min_share)


class DispatchIn(BaseModel):
    cycle_id: str | None = None            # default: the latest cycle
    slices: list[str] | None = None        # default: every class the cycle allocated labels to
    per_slice_cap: int = 300


@router.post("/flywheel/adaptive/dispatch")
async def adaptive_dispatch(payload: DispatchIn, db: AsyncSession = Depends(db_session)):
    """Action a cycle's work order: bump its real labelable candidates into the review queue, stamped with the
    cycle provenance, as one reversible AgentRun. Never touches a human-accepted object."""
    try:
        return await dispatch_worklist(db, payload.cycle_id, payload.slices, payload.per_slice_cap)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class AdaptiveIn(BaseModel):
    regressions: list[dict]          # VERDYX: [{slice, delta, protected}]
    odd_gaps: list[dict]             # SIEVYX: coverage_gaps() cell records
    total_label_budget: int
    total_fleet_samples: int
    safety_slices: list[str] | None = None
    safety_floor: int = 0


@router.post("/flywheel/adaptive/cycle")
async def adaptive_cycle_run(payload: AdaptiveIn, db: AsyncSession = Depends(db_session)):
    """Run one adaptive cycle: route VERDYX regressions to a label-budget allocation and SIEVYX ODD gaps to
    collection tasks, then record the cycle to the ledger. This is the controller that closes the loop."""
    return await run_and_record(db, payload.regressions, payload.odd_gaps, payload.total_label_budget,
                                payload.total_fleet_samples, payload.safety_slices, payload.safety_floor)


@router.get("/flywheel/adaptive/cycles")
async def adaptive_cycles(limit: int = 20, db: AsyncSession = Depends(db_session)):
    """The adaptive-cycle history: how the data engine steered its label spend and collection over time."""
    return await recent_cycles(db, limit)


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
