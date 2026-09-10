"""Shadow mode: what the champion and its challengers disagree about, and who was right.

Reading is an annotator's business, because the disagreement list is a worklist. Launching a sweep puts
two detectors on the card for several minutes, so it is a reviewer action, and it goes through the same
`maybe_shadow_sweep` the nightly scheduler calls rather than a second path that could drift from it.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun, ShadowDisagreement
from services.api.deps import db_session, require_role
from services.govern.shadow_agent import KIND, maybe_shadow_sweep, sweep_summary

router = APIRouter()


@router.get("/shadow/summary", dependencies=[Depends(require_role("annotator"))])
async def shadow_summary(limit: int = Query(default=5, ge=1, le=50),
                         db: AsyncSession = Depends(db_session)):
    """Recent sweeps, disagreement counts by state and kind, and each challenger's adjudicated win share."""
    return await sweep_summary(db, limit=limit)


@router.get("/shadow/disagreements", dependencies=[Depends(require_role("annotator"))])
async def list_disagreements(sweep_run_id: str | None = None, state: str | None = None,
                             kind: str | None = None, limit: int = Query(default=100, ge=1, le=500),
                             db: AsyncSession = Depends(db_session)):
    """The worklist itself, worst first. Filters are all optional and all exact."""
    stmt = select(ShadowDisagreement).order_by(ShadowDisagreement.score.desc()).limit(limit)
    if sweep_run_id:
        try:
            stmt = stmt.where(ShadowDisagreement.sweep_run_id == uuid.UUID(sweep_run_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="sweep_run_id is not a uuid") from None
    if state:
        stmt = stmt.where(ShadowDisagreement.state == state)
    if kind:
        stmt = stmt.where(ShadowDisagreement.kind == kind)
    rows = (await db.execute(stmt)).scalars().all()
    return {"disagreements": [
        {"disagreement_id": str(d.disagreement_id), "frame_id": str(d.frame_id), "kind": d.kind,
         "champion_class_id": d.champion_class_id, "challenger_class_id": d.challenger_class_id,
         "iou": d.iou, "conf_gap": d.conf_gap, "score": d.score, "bbox": [float(v) for v in d.bbox],
         "state": d.state, "verdict": d.verdict,
         "task_id": str(d.task_id) if d.task_id else None,
         "frame_at": f"/frame/{d.frame_id}",
         "adjudicated_at": d.adjudicated_at.isoformat() if d.adjudicated_at else None}
        for d in rows]}


class SweepIn(BaseModel):
    # Naming challengers makes this an operator's sweep rather than the scheduler's: the once-a-day marker
    # and the thirty-day discovery window stop applying, and every guard that protects the card still does.
    challengers: list[str] = Field(default_factory=list, max_length=4)
    frame_limit: int | None = Field(default=None, ge=1, le=20000)


@router.post("/shadow/sweep", dependencies=[Depends(require_role("reviewer"))])
async def start_sweep(body: SweepIn | None = None, db: AsyncSession = Depends(db_session)):
    """Run a sweep now. Declines with the same reason the scheduler would give rather than forcing one."""
    body = body or SweepIn()
    res = await maybe_shadow_sweep(db, challengers=body.challengers or None,
                                   frame_limit=body.frame_limit)
    if not res.get("ran"):
        raise HTTPException(status_code=409, detail=res.get("reason", "the sweep declined"))
    return res


@router.get("/shadow/runs", dependencies=[Depends(require_role("annotator"))])
async def shadow_runs(limit: int = Query(default=20, ge=1, le=100),
                      db: AsyncSession = Depends(db_session)):
    runs = (await db.execute(select(AgentRun).where(AgentRun.kind == KIND)
                             .order_by(AgentRun.created_at.desc()).limit(limit))).scalars().all()
    pending = {k: int(v) for k, v in (await db.execute(
        select(ShadowDisagreement.state, func.count()).group_by(ShadowDisagreement.state))).all()}
    return {"runs": [{"run_id": str(r.run_id), "status": r.status, "policy": r.policy,
                      "report": r.counts,
                      "created_at": r.created_at.isoformat() if r.created_at else None} for r in runs],
            "disagreements_by_state": pending}
