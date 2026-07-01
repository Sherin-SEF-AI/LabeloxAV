"""The annotation-agent API: dry-run a frame (plan), commit a reversible run, list/inspect runs, and revert
one. The plan endpoint writes nothing; commit auto-accepts the confident objects and routes the rest, all
recorded in one AgentRun; revert restores the exact prior state. Auto-accept is gated on reviewer role.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun
from services.agent.flywheel import run_flywheel
from services.agent.frame_agent import commit_frame, plan_frame
from services.agent.policy import PolicyThresholds
from services.agent.reconcile import reconcile_frame
from services.agent.runs import list_runs, revert_run, run_dict
from services.api.deps import current_user, db_session, require_role

router = APIRouter()


class AgentPolicyIn(BaseModel):
    auto_accept_conf: float | None = None
    review_low: float | None = None
    require_agreement: bool | None = None


def _thresholds(body: AgentPolicyIn | None) -> PolicyThresholds:
    d = PolicyThresholds()
    if not body:
        return d
    return PolicyThresholds(
        auto_accept_conf=body.auto_accept_conf if body.auto_accept_conf is not None else d.auto_accept_conf,
        review_low=body.review_low if body.review_low is not None else d.review_low,
        require_agreement=body.require_agreement if body.require_agreement is not None else d.require_agreement,
    )


@router.post("/agent/frames/{frame_id}/plan", dependencies=[Depends(require_role("annotator"))])
async def plan(frame_id: str, body: AgentPolicyIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: what the agent would auto-accept / route on this frame. Writes nothing."""
    try:
        return await plan_frame(db, uuid.UUID(frame_id), _thresholds(body))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/frames/{frame_id}/run", dependencies=[Depends(require_role("reviewer"))])
async def run(frame_id: str, body: AgentPolicyIn | None = None,
              db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Commit the plan as one reversible run: auto-accept the confident, route the rest."""
    try:
        return await commit_frame(db, uuid.UUID(frame_id), _thresholds(body),
                                  created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class FlywheelIn(AgentPolicyIn):
    ticks: int = 1
    max_frames: int = 25
    session_id: str | None = None
    dry_run: bool = True  # default: report what it would auto-accept without writing


@router.post("/agent/flywheel", dependencies=[Depends(require_role("reviewer"))])
async def flywheel(body: FlywheelIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Launch the autonomous loop in the background: mine by value, auto-accept the sure ones / route the
    rest, then retrain if enough corrections have accumulated. Poll GET /agent/runs/{run_id} for progress.
    dry_run (default) plans only and writes nothing."""
    run_id = uuid.uuid4()
    run = AgentRun(
        run_id=run_id, kind="flywheel",
        scope={"ticks": body.ticks, "max_frames": body.max_frames, "session_id": body.session_id},
        status="running", policy=_thresholds(body).to_dict(), counts={}, changes={}, critic={},
        created_by=str(user.user_id) if user else "flywheel",
    )
    db.add(run)
    await db.commit()
    asyncio.create_task(run_flywheel(
        run_id, ticks=max(1, body.ticks), max_frames=max(1, body.max_frames),
        policy=_thresholds(body), session_id=body.session_id, dry_run=body.dry_run,
        created_by=str(user.user_id) if user else "flywheel",
    ))
    return {"run_id": str(run_id), "status": "running", "dry_run": body.dry_run}


class CommandIn(BaseModel):
    text: str
    frame_id: str


@router.post("/agent/command", dependencies=[Depends(require_role("annotator"))])
async def command(body: CommandIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Natural-language control: turn an instruction ('auto-accept the two-wheelers above 0.9') into a
    scoped agent action on a frame. Returns the parsed intent, the result, and a plain-language summary.
    plan/find are read-only; accept/revert write and are reversible."""
    from services.agent.nl import execute_command
    from services.api.deps import role_rank

    can_write = bool(user) and role_rank(user.role) >= role_rank("reviewer")
    try:
        return await execute_command(db, body.text, uuid.UUID(body.frame_id),
                                     created_by=str(user.user_id) if user else None, can_write=can_write)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class ReconcileIn(BaseModel):
    object_ids: list[str] | None = None  # None = the whole frame's machine objects


@router.post("/agent/frames/{frame_id}/reconcile", dependencies=[Depends(require_role("annotator"))])
async def reconcile(frame_id: str, body: ReconcileIn | None = None, db: AsyncSession = Depends(db_session)):
    """Adjudicate the frame's objects with an independent model (SigLIP 2 zero-shot): confirm / correct /
    unsure per object. Read-only -- returns opinions and suggested relabels, never mutates."""
    try:
        return await reconcile_frame(db, uuid.UUID(frame_id), body.object_ids if body else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/agent/runs", dependencies=[Depends(require_role("annotator"))])
async def runs(limit: int = 50, db: AsyncSession = Depends(db_session)):
    return await list_runs(db, limit)


@router.get("/agent/runs/{run_id}", dependencies=[Depends(require_role("annotator"))])
async def run_detail(run_id: str, db: AsyncSession = Depends(db_session)):
    r = await db.get(AgentRun, uuid.UUID(run_id))
    if r is None:
        raise HTTPException(404, "run not found")
    return {**run_dict(r), "changes": r.changes}


@router.post("/agent/runs/{run_id}/revert", dependencies=[Depends(require_role("reviewer"))])
async def revert(run_id: str, db: AsyncSession = Depends(db_session)):
    try:
        return await revert_run(db, uuid.UUID(run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
