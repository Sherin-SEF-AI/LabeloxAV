"""UI-triggered autolabel: run the Path A/B/C + fusion + gate sweep over a session from the UI as a
background job, instead of `make label` in a terminal. Single-GPU discipline: refuse if a training job
holds the GPU, and only one autolabel runs at a time.

The sweep runs as an asyncio task in the API process (the simplest robust model on one box); it is GPU
heavy, so the API is somewhat less responsive while it runs. A dedicated worker is the cloud seam.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AutolabelJob, TrainingJob
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.api.deps import AutolabelStartIn, db_session

log = get_logger("api_autolabel")
router = APIRouter()


def _job_dict(j: AutolabelJob) -> dict:
    return {
        "job_id": str(j.job_id), "session_id": str(j.session_id), "status": j.status,
        "progress": j.progress, "counts": j.counts, "error": j.error,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "updated_at": j.updated_at.isoformat() if j.updated_at else None,
    }


async def _bump(job_id, **fields) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(AutolabelJob, uuid.UUID(str(job_id)))
        if j:
            for k, v in fields.items():
                setattr(j, k, v)
            await db.commit()


async def _run_guarded(job_id, session_id, limit) -> None:
    from services.autolabel.runner import autolabel_session

    await _bump(job_id, status="running", progress=0.05)
    try:
        result = await autolabel_session(session_id, limit)
        await _bump(job_id, status="done", progress=1.0, counts=result)
        log.info("autolabel.done", job_id=str(job_id), **{k: result[k] for k in result if k in ("n_frames", "n_objects")})
    except Exception as exc:  # noqa: BLE001
        log.error("autolabel.failed", job_id=str(job_id), error=str(exc))
        await _bump(job_id, status="error", error=str(exc))


@router.post("/autolabel/start")
async def start(payload: AutolabelStartIn, db: AsyncSession = Depends(db_session)):
    if (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first():
        raise HTTPException(503, "GPU reserved for a training job; autolabel is paused until it finishes")
    if (await db.execute(select(AutolabelJob.job_id).where(AutolabelJob.status == "running").limit(1))).first():
        raise HTTPException(409, "an autolabel job is already running")
    sess = await db.get(DbSession, uuid.UUID(payload.session_id))
    if sess is None:
        raise HTTPException(404, "session not found")
    job_id = uuid.uuid4()
    db.add(AutolabelJob(job_id=job_id, session_id=uuid.UUID(payload.session_id), status="pending"))
    await db.commit()

    if payload.compute_target == "cloud":
        # Park it for the A100 heavy stack; the local API never runs a cloud job (GPU discipline).
        from services.autolabel.cloud import mark_queued_for_cloud

        await mark_queued_for_cloud(job_id, payload.session_id, payload.limit)
        return {"job_id": str(job_id), "status": "queued-cloud"}

    asyncio.create_task(_run_guarded(job_id, uuid.UUID(payload.session_id), payload.limit))
    return {"job_id": str(job_id), "status": "pending"}


@router.get("/autolabel/{job_id}")
async def status(job_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    j = await db.get(AutolabelJob, job_id)
    if j is None:
        raise HTTPException(404, "autolabel job not found")
    return _job_dict(j)
