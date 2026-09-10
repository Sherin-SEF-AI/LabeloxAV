"""Copy-paste synthesis: plan a build (no writes), launch one, and list what the generator has made.

A build writes frames that only the trainer can opt into (`BuildSpec.include_synthetic`); everything that
measures a model is structurally blind to them (see `services/synth`). Launching one is a reviewer action:
it creates a session and thousands of rows, all of them revertible through the run it records.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun
from services.api.deps import current_user, db_session, require_role
from services.synth.copy_paste import BATCH_KIND, KIND, SYNTH_BATCH, build, plan_build

router = APIRouter()

MAX_FRAMES_PER_BUILD = 5000


class PlanIn(BaseModel):
    class_names: list[str] = Field(min_length=1, max_length=16)
    n_frames: int = Field(default=500, ge=1, le=MAX_FRAMES_PER_BUILD)


class BuildIn(PlanIn):
    seed: str | None = None


@router.post("/synth/plan", dependencies=[Depends(require_role("annotator"))])
async def synth_plan(body: PlanIn, db: AsyncSession = Depends(db_session)):
    """Donors per class, usable backgrounds, and the batch count a build would run in. Writes nothing."""
    return await plan_build(db, class_names=body.class_names, n_frames=body.n_frames)


@router.post("/synth/build", dependencies=[Depends(require_role("reviewer"))])
async def synth_build(body: BuildIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Launch a build. Refuses up front when no requested class has a donor; otherwise returns the parent
    run id to poll at GET /agent/runs/{run_id}. Frames land in batches of SYNTH_BATCH, each its own run."""
    from services.agent.runtime.report import launch

    plan = await plan_build(db, class_names=body.class_names, n_frames=body.n_frames)
    if not plan["feasible"]:
        raise HTTPException(409, plan["reason"])
    seed = body.seed or uuid.uuid4().hex[:8]
    created_by = str(user.user_id) if user else "api"

    async def worker(run_id: uuid.UUID):
        await build(run_id, class_names=body.class_names, n_frames=body.n_frames, seed=seed,
                    created_by=created_by)

    res = await launch(db, KIND, worker, created_by=created_by,
                       policy={"class_names": body.class_names, "n_frames": body.n_frames, "seed": seed})
    return {**res, "plan": plan, "batch_size": SYNTH_BATCH, "seed": seed}


@router.get("/synth/runs", dependencies=[Depends(require_role("annotator"))])
async def synth_runs(limit: int = 50, db: AsyncSession = Depends(db_session)):
    """Builds newest first, each with its batches. `report` is the build's live counts while it runs."""
    limit = max(1, min(limit, 200))
    parents = (await db.execute(select(AgentRun).where(AgentRun.kind == KIND)
                                .order_by(AgentRun.created_at.desc()).limit(limit))).scalars().all()
    batches = (await db.execute(select(AgentRun).where(AgentRun.kind == BATCH_KIND)
                                .order_by(AgentRun.created_at.asc()))).scalars().all()
    by_parent: dict[str, list[dict]] = {}
    for b in batches:
        pid = (b.scope or {}).get("parent_run_id") or ""
        by_parent.setdefault(pid, []).append({
            "run_id": str(b.run_id), "status": b.status, "counts": b.counts or {},
            "frames": len((b.changes or {}).get("frame_ids") or []),
            "created_at": b.created_at.isoformat() if b.created_at else None})
    return {"runs": [{
        "run_id": str(p.run_id), "status": p.status, "policy": p.policy or {}, "report": p.counts or {},
        "created_by": p.created_by, "created_at": p.created_at.isoformat() if p.created_at else None,
        "batches": by_parent.get(str(p.run_id), [])} for p in parents]}
