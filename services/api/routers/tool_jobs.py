"""Status of the annotation tools' background jobs (the `tool_job` queue, services/tools/registry.py).

Every tool page polls the job it started here, rather than each tool growing its own status route.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ToolJob
from services.api.deps import db_session, require_user, role_rank

router = APIRouter()


class ToolJobOut(BaseModel):
    job_id: str
    kind: str
    label: str
    status: str
    gpu: bool
    session_id: str | None = None
    params: dict
    result: dict | None = None
    progress: float
    stage: str | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    quiet_hours: bool = False   # waiting because auto-labelling is paused for office hours


def to_out(j: ToolJob) -> dict:
    from services.tools.registry import in_quiet_hours, load_kinds

    tk = load_kinds().get(j.kind)
    iso = lambda d: d.isoformat() if d else None  # noqa: E731
    return {"job_id": str(j.job_id), "kind": j.kind, "label": tk.label if tk else j.kind, "status": j.status,
            "gpu": j.gpu, "session_id": str(j.session_id) if j.session_id else None, "params": j.params or {},
            "result": j.result, "progress": float(j.progress or 0), "stage": j.stage, "error": j.error,
            "created_at": iso(j.created_at), "started_at": iso(j.started_at), "finished_at": iso(j.finished_at),
            "quiet_hours": bool(j.status == "pending" and tk is not None and tk.autolabel and in_quiet_hours())}


@router.get("/tools/jobs/{job_id}", response_model=ToolJobOut)
async def get_tool_job(job_id: uuid.UUID, db: AsyncSession = Depends(db_session), user=Depends(require_user)):
    job = await db.get(ToolJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such tool job")
    return to_out(job)


@router.get("/tools/jobs", response_model=list[ToolJobOut])
async def list_tool_jobs(kind: str | None = None, session_id: uuid.UUID | None = None,
                         limit: int = Query(20, ge=1, le=200), db: AsyncSession = Depends(db_session),
                         user=Depends(require_user)):
    q = select(ToolJob).order_by(ToolJob.created_at.desc()).limit(limit)
    if kind:
        q = q.where(ToolJob.kind == kind)
    if session_id:
        q = q.where(ToolJob.session_id == session_id)
    return [to_out(j) for j in (await db.execute(q)).scalars()]


@router.post("/tools/jobs/{job_id}/cancel", response_model=ToolJobOut)
async def cancel_tool_job(job_id: uuid.UUID, db: AsyncSession = Depends(db_session), user=Depends(require_user)):
    """Cancel a tool job: the person who started it, or a reviewer. A running job stops at its next checkpoint."""
    job = await db.get(ToolJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such tool job")
    if job.created_by != user.user_id and role_rank(user.role) < role_rank("reviewer"):
        raise HTTPException(status_code=403, detail="only the person who started it, or a reviewer, can cancel it")
    if job.status in ("pending", "running"):
        job.status = "canceled"
        job.finished_at = datetime.now(UTC) if job.started_at is None else job.finished_at
        await db.commit()
        await db.refresh(job)
    return to_out(job)
