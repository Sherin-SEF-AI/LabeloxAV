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


@router.post("/flywheel/adaptive/collection-orders")
async def adaptive_collection_orders(cycle_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Turn a cycle's collection tasks into ranked fleet collection orders: scene cells become windowed,
    forecast-aware drives and empty species become class-collection orders, each with its target count. View
    the result on the fleet board (GET /api/agent/fleet/orders)."""
    from services.agent.fleet_dispatch import plan_from_flywheel
    try:
        return await plan_from_flywheel(db, cycle_id=uuid.UUID(cycle_id) if cycle_id else None)
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


@router.get("/flywheel/gate-directed")
async def gate_directed_latest(budget: int = 500, db: AsyncSession = Depends(db_session)):
    """Plan a batch for the most recent unpromoted run, so a caller can ask what to label next without first
    having to know which run last failed the gate."""
    from services.flywheel.gate_directed import plan_gate_batch
    from services.flywheel.gate_signals import latest_blocked_run

    run_id = await latest_blocked_run(db)
    if run_id is None:
        return {"run_id": None, "blocking": False, "allocations": [], "total_frames": 0,
                "rationale": "no unpromoted training run to diagnose"}
    return await plan_gate_batch(db, run_id, budget=budget)


@router.get("/flywheel/gate-directed/{run_id}")
async def gate_directed_plan(run_id: str, budget: int = 500, db: AsyncSession = Depends(db_session)):
    """Diagnose why a training run cannot be promoted and plan the review batch that would unblock it.

    Read-only: it reports which safety classes miss their recall floor or regressed, how the label budget
    splits across them, and how many frames a reviewer would open. A run the gate does not block returns no
    allocations rather than manufacturing work.
    """
    from services.flywheel.gate_directed import plan_gate_batch

    try:
        return await plan_gate_batch(db, run_id, budget=budget)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class GateBatchIn(BaseModel):
    project_id: str
    budget: int = 500
    jobs_of: int = 50


@router.post("/flywheel/gate-directed/{run_id}/materialize")
async def gate_directed_materialize(run_id: str, payload: GateBatchIn,
                                    db: AsyncSession = Depends(db_session)):
    """Create one labelops task per blocked safety class, holding exactly the frames worth reviewing.

    These are review batches, not training data. Nothing here writes an accepted label: iteration 5 trained on
    unreviewed machine labels and drove pedestrian recall from 0.73 to 0.004, which is the failure this
    endpoint exists to route around.
    """
    from services.flywheel.gate_directed import materialize_gate_batch

    try:
        return await materialize_gate_batch(db, run_id, payload.project_id,
                                            budget=payload.budget, jobs_of=payload.jobs_of)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
