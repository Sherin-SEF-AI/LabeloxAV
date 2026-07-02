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


class CuboidIn(BaseModel):
    min_iou: float = 0.35
    high: float = 0.60


@router.post("/agent/frames/{frame_id}/cuboids/plan", dependencies=[Depends(require_role("annotator"))])
async def cuboids_plan(frame_id: str, body: CuboidIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: which 2D vehicle/VRU boxes on this frame lift to a valid 3D cuboid (monocular, reprojection-
    validated). Writes nothing."""
    from services.agent.cuboid_agent import plan_cuboids

    b = body or CuboidIn()
    try:
        return await plan_cuboids(db, uuid.UUID(frame_id), min_iou=b.min_iou, high=b.high)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/frames/{frame_id}/cuboids", dependencies=[Depends(require_role("reviewer"))])
async def cuboids_run(frame_id: str, body: CuboidIn | None = None,
                      db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Attach fitted 3D cuboids to the frame's 2D objects as one reversible run."""
    from services.agent.cuboid_agent import commit_cuboids

    b = body or CuboidIn()
    try:
        return await commit_cuboids(db, uuid.UUID(frame_id), min_iou=b.min_iou, high=b.high,
                                    created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class PropagateIn(BaseModel):
    span: int = 24
    drift: float = 0.62
    high: float = 0.80


@router.post("/agent/objects/{object_id}/propagate/plan", dependencies=[Depends(require_role("annotator"))])
async def propagate_plan(object_id: str, body: PropagateIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: what the track-propagation agent would carry across the clip from this keyframe (both
    directions, stopping where the box drifts). Writes nothing."""
    from services.agent.propagate_agent import plan_propagate

    b = body or PropagateIn()
    try:
        return await plan_propagate(db, uuid.UUID(object_id), span=b.span, drift=b.drift, high=b.high)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/objects/{object_id}/propagate", dependencies=[Depends(require_role("reviewer"))])
async def propagate_run(object_id: str, body: PropagateIn | None = None,
                        db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Propagate the keyframe across its track and persist the boxes as one reversible run."""
    from services.agent.propagate_agent import commit_propagate

    b = body or PropagateIn()
    try:
        return await commit_propagate(db, uuid.UUID(object_id), span=b.span, drift=b.drift, high=b.high,
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


class FreshIn(AgentPolicyIn):
    commit: bool = False  # also apply the agent decision; False plans over the freshly-detected objects


@router.post("/agent/frames/{frame_id}/fresh", dependencies=[Depends(require_role("reviewer"))])
async def fresh(frame_id: str, body: FreshIn | None = None,
                db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Fresh-inference: detect the frame (Path A/B + fuse + gate + persist) then run the agent on the new
    objects in one shot. commit applies the decision; otherwise it plans. GPU-bound (reviewer role)."""
    from services.agent.fresh import label_and_decide

    body = body or FreshIn()
    try:
        return await label_and_decide(db, uuid.UUID(frame_id), commit=body.commit, policy=_thresholds(body),
                                      created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


class ReconcileIn(BaseModel):
    object_ids: list[str] | None = None  # None = the whole frame's machine objects
    apply: bool = False                  # apply strong 'correct' verdicts as reversible relabels
    apply_min_conf: float = 0.55


@router.post("/agent/frames/{frame_id}/reconcile", dependencies=[Depends(require_role("annotator"))])
async def reconcile(frame_id: str, body: ReconcileIn | None = None,
                    db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Adjudicate the frame's objects with an independent model (SigLIP 2 zero-shot): confirm / correct /
    unsure per object. Read-only unless apply=True, which relabels the strong 'correct' verdicts as one
    reversible AgentRun (reviewer role required to apply)."""
    from services.api.deps import role_rank

    body = body or ReconcileIn()
    if body.apply and not (user and role_rank(user.role) >= role_rank("reviewer")):
        raise HTTPException(403, "applying relabels requires reviewer role")
    try:
        return await reconcile_frame(db, uuid.UUID(frame_id), body.object_ids, apply=body.apply,
                                     apply_min_conf=body.apply_min_conf,
                                     created_by=str(user.user_id) if user else None)
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
