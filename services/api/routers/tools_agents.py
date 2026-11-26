"""API for the agents annotation tools, agent F: agents (#17 language bulk edits, #18 overnight autopilot).

Mounted under /api; every path here starts with /tools/agents/ so the area's routes stay together.

#17 Bulk edits in plain language. `POST /tools/agents/edit/preview` reads a sentence ("relabel every pickup in
Delhi sessions as LCV") and returns what it would change, per session and class, with sample crops and what it
leaves alone. Nothing changes. `POST /tools/agents/edit/propose` stores the same preview as an assistant
proposal carrying the exact object ids; a reviewer approves it through the assistant's approve route
(`POST /api/assistant/proposals/{id}/approve`), which applies it as one reversible AgentRun, and
`POST /tools/agents/edit/proposals/{id}/revert` undoes it.

#18 Overnight autopilot. Settings, the kill switch, a dry run, "run tonight now", and the history of nights.
The nightly start itself is the scheduler's (`maybe_autopilot` in services/agent/runtime/schedule.py).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun, AssistantProposal, AutopilotNight
from services.activity import record_activity
from services.api.deps import current_user, db_session, require_role
from services.api.project_guard import caller_hidden_projects
from services.project_access import as_hidden

router = APIRouter()


# ---- #17 bulk edits in plain language -------------------------------------------------------------------------

class EditIn(BaseModel):
    command: str = Field(min_length=3, max_length=500)
    session_id: str | None = None          # narrow to one session (the page's "working in"), optional


async def _bulk(db: AsyncSession, body: EditIn, hidden: frozenset) -> tuple[dict, dict]:
    from services.agent.nl_edit import parse_bulk_edit, resolve_bulk

    plan = parse_bulk_edit(body.command)
    if body.session_id:
        try:
            uuid.UUID(body.session_id)
        except ValueError:
            raise HTTPException(400, "invalid session id") from None
        plan["scope"]["session_ids"] = [body.session_id]
        plan["scope"]["described"].append("in the chosen session")
    res = await resolve_bulk(db, plan, hidden=hidden)
    return plan, res


def _public(res: dict) -> dict:
    """The preview without the id list (it can be twenty thousand long; the proposal keeps it)."""
    return {k: v for k, v in res.items() if k != "object_ids"}


@router.post("/tools/agents/edit/preview", dependencies=[Depends(require_role("annotator"))])
async def edit_preview(body: EditIn, db: AsyncSession = Depends(db_session), hidden=Depends(caller_hidden_projects)):
    """What a sentence would change across sessions. Reads only."""
    _, res = await _bulk(db, body, as_hidden(hidden))
    return _public(res)


@router.post("/tools/agents/edit/propose")
async def edit_propose(body: EditIn, db: AsyncSession = Depends(db_session), user=Depends(require_role("annotator")),
                       hidden=Depends(caller_hidden_projects)):
    """Resolve again (the data may have moved since the preview) and store it for a reviewer to approve."""
    from services.assistant.tools import Context, ToolError, store_bulk_proposal

    plan, res = await _bulk(db, body, as_hidden(hidden))
    ctx = Context(db=db, user=user, hidden=as_hidden(hidden), request=body.command)
    try:
        await store_bulk_proposal(ctx, plan, res)
    except ToolError as exc:
        raise HTTPException(422, str(exc)) from None
    p = await db.get(AssistantProposal, uuid.UUID(ctx.proposals[-1]["proposal_id"]))
    assert p is not None
    await record_activity(db, user=user, verb="bulk_edit_proposed", subject_type="assistant_proposal",
                          subject_id=str(p.proposal_id), summary=p.summary)
    return await _proposal_out(db, p)


async def _proposal_out(db: AsyncSession, p: AssistantProposal) -> dict:
    run = None
    run_id = (p.result or {}).get("run_id")
    if run_id:
        r = await db.get(AgentRun, uuid.UUID(run_id))
        if r is not None:
            run = {"run_id": str(r.run_id), "status": r.status, "edited": len(r.changes or {}),
                   "reverted_at": r.reverted_at.isoformat() if r.reverted_at else None}
    return {"proposal_id": str(p.proposal_id), "tool": p.tool, "summary": p.summary, "preview": p.preview,
            "status": p.status, "result": p.result, "request": p.request, "run": run,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "decided_at": p.decided_at.isoformat() if p.decided_at else None}


async def _bulk_proposal(db: AsyncSession, proposal_id: str) -> AssistantProposal:
    try:
        p = await db.get(AssistantProposal, uuid.UUID(proposal_id))
    except ValueError:
        raise HTTPException(400, "invalid proposal id") from None
    if p is None or p.tool != "propose_bulk_edit":
        raise HTTPException(404, "bulk edit proposal not found")
    return p


@router.get("/tools/agents/edit/proposals", dependencies=[Depends(require_role("annotator"))])
async def edit_proposals(limit: Annotated[int, Query(ge=1, le=100)] = 20, db: AsyncSession = Depends(db_session)):
    """Recent bulk edits, newest first, with their approval state and the run each one became."""
    rows = (await db.execute(select(AssistantProposal).where(AssistantProposal.tool == "propose_bulk_edit")
                             .order_by(AssistantProposal.created_at.desc()).limit(limit))).scalars().all()
    return {"proposals": [await _proposal_out(db, p) for p in rows]}


@router.get("/tools/agents/edit/proposals/{proposal_id}", dependencies=[Depends(require_role("annotator"))])
async def edit_proposal(proposal_id: str, db: AsyncSession = Depends(db_session)):
    return await _proposal_out(db, await _bulk_proposal(db, proposal_id))


@router.post("/tools/agents/edit/proposals/{proposal_id}/revert")
async def edit_revert(proposal_id: str, db: AsyncSession = Depends(db_session), user=Depends(require_role("reviewer"))):
    """Undo an applied bulk edit: every label it changed goes back, except any a person has touched since."""
    from services.agent.runs import revert_run

    p = await _bulk_proposal(db, proposal_id)
    run_id = (p.result or {}).get("run_id")
    if p.status != "approved" or not run_id:
        raise HTTPException(409, "only an approved and applied edit can be undone")
    try:
        res = await revert_run(db, uuid.UUID(run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    await record_activity(db, user=user, verb="bulk_edit_reverted", subject_type="assistant_proposal",
                          subject_id=str(p.proposal_id), summary=p.summary)
    return {**res, "proposal": await _proposal_out(db, p)}


# ---- #18 overnight autopilot ------------------------------------------------------------------------------------

@router.get("/tools/agents/autopilot", dependencies=[Depends(require_role("annotator"))])
async def autopilot_status(db: AsyncSession = Depends(db_session)):
    """Settings, whether tonight's window is open, the night in progress, and the last real night."""
    from services.agent.autopilot import config_dict, get_config, night_dict, window_state

    cfg = await get_config(db)
    active = (await db.execute(select(AutopilotNight).where(
        AutopilotNight.dry_run.is_(False), AutopilotNight.status.in_(("planned", "running")))
        .order_by(AutopilotNight.created_at.desc()).limit(1))).scalar_one_or_none()
    last = (await db.execute(select(AutopilotNight).where(
        AutopilotNight.dry_run.is_(False), AutopilotNight.status.notin_(("planned", "running")))
        .order_by(AutopilotNight.created_at.desc()).limit(1))).scalar_one_or_none()
    return {"settings": config_dict(cfg), "window": window_state(cfg),
            "active": night_dict(active) if active else None, "last": night_dict(last) if last else None}


class SettingsIn(BaseModel):
    enabled: bool | None = None
    gpu_minutes: int | None = None
    max_sessions: int | None = None
    vlm_max_checks: int | None = None
    earliest_start_ist: str | None = Field(default=None, max_length=5)
    steps: dict[str, bool] | None = None


@router.put("/tools/agents/autopilot/settings")
async def autopilot_settings(body: SettingsIn, db: AsyncSession = Depends(db_session),
                             user=Depends(require_role("admin"))):
    """Change what the autopilot may do at night. Spending the GPU on the whole corpus unattended is an admin
    decision."""
    from services.agent.autopilot import config_dict, get_config, validate_settings

    try:
        changes = validate_settings(body.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    cfg = await get_config(db)
    for k, v in changes.items():
        setattr(cfg, k, v)
    cfg.updated_by = getattr(user, "user_id", None)
    await db.commit()
    await db.refresh(cfg)
    await record_activity(db, user=user, verb="autopilot_settings", subject_type="autopilot",
                          subject_id="config", summary=", ".join(f"{k}={v}" for k, v in changes.items())[:200])
    return config_dict(cfg)


class KillIn(BaseModel):
    reason: str = Field(default="stopped from the autopilot page", max_length=500)


@router.post("/tools/agents/autopilot/kill")
async def autopilot_kill(body: KillIn, db: AsyncSession = Depends(db_session), user=Depends(require_role("reviewer"))):
    """Stop the autopilot now and keep it stopped until an admin re-arms it. Any reviewer may pull it."""
    from services.agent.autopilot import kill

    res = await kill(db, reason=body.reason, user_id=getattr(user, "user_id", None))
    await record_activity(db, user=user, verb="autopilot_killed", subject_type="autopilot", subject_id="config",
                          summary=body.reason[:200])
    return res


@router.post("/tools/agents/autopilot/rearm")
async def autopilot_rearm(db: AsyncSession = Depends(db_session), user=Depends(require_role("admin"))):
    from services.agent.autopilot import config_dict, rearm

    cfg = await rearm(db, user_id=getattr(user, "user_id", None))
    await record_activity(db, user=user, verb="autopilot_rearmed", subject_type="autopilot", subject_id="config",
                          summary="kill switch released")
    return config_dict(cfg)


@router.post("/tools/agents/autopilot/dry-run")
async def autopilot_dry_run(db: AsyncSession = Depends(db_session), user=Depends(require_role("reviewer"))):
    """What a night would do if it started now. Changes no labels; the plan is kept in the history."""
    from services.agent.autopilot import night_dict, record_dry_run

    n = await record_dry_run(db, user_id=getattr(user, "user_id", None))
    return night_dict(n)


@router.post("/tools/agents/autopilot/run-now")
async def autopilot_run_now(db: AsyncSession = Depends(db_session), user=Depends(require_role("admin"))):
    """Queue tonight's run now instead of waiting for the scheduler. The GPU worker still will not start it in
    office hours, and the kill switch still stops it."""
    from services.agent.autopilot import (
        AlreadyRan,
        get_config,
        night_dict,
        start_night,
        window_state,
    )

    cfg = await get_config(db)
    if cfg.killed:
        raise HTTPException(409, "the kill switch is on; an admin re-arms it first")
    try:
        n = await start_night(db, trigger="manual", user_id=getattr(user, "user_id", None))
    except AlreadyRan as exc:
        raise HTTPException(409, str(exc)) from None
    await record_activity(db, user=user, verb="autopilot_run_now", subject_type="autopilot_night",
                          subject_id=str(n.night_id), summary="queued tonight's autopilot run")
    w = window_state(cfg)
    return {**night_dict(n), "starts": "when the GPU worker is free" if not w["quiet"] else
            f"after office hours ({w['quiet_hours_ist']} IST)"}


@router.get("/tools/agents/autopilot/nights", dependencies=[Depends(require_role("annotator"))])
async def autopilot_nights(limit: Annotated[int, Query(ge=1, le=200)] = 30, dry_runs: bool = True,
                           db: AsyncSession = Depends(db_session)):
    """Past nights, newest first, with a one-line summary each."""
    from services.agent.autopilot import night_dict

    q = select(AutopilotNight).order_by(AutopilotNight.created_at.desc()).limit(limit)
    if not dry_runs:
        q = q.where(AutopilotNight.dry_run.is_(False))
    return {"nights": [night_dict(n, full=False) for n in (await db.execute(q)).scalars().all()]}


@router.get("/tools/agents/autopilot/nights/{night_id}", dependencies=[Depends(require_role("annotator"))])
async def autopilot_night(night_id: str, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    from services.agent.autopilot import night_dict

    try:
        n = await db.get(AutopilotNight, uuid.UUID(night_id))
    except ValueError:
        raise HTTPException(400, "invalid night id") from None
    if n is None:
        raise HTTPException(404, "night not found")
    return night_dict(n)
