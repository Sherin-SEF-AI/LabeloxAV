"""UI-triggered autolabel: run the Path A/B/C + fusion + gate sweep over a session from the UI as a
background job, instead of `make label` in a terminal.

Who runs it depends on `jobs.autolabel_executor`:
  "api"     an asyncio task in this process. Single-GPU discipline is enforced here, by refusing while a
            training job holds the GPU or another auto-label is running.
  "worker"  the row is queued and the GPU worker claims it. Nothing is refused for being busy: the worker
            runs one job at a time and the queue is the discipline. This is the Docker deployment.
Either way the job itself is `services/autolabel/jobs.run_autolabel_job`, read from its row.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.observability import spawn
from db.models import AutolabelJob
from db.models import Session as DbSession
from services.api.deps import AutolabelStartIn, db_session
from services.training.gpu_lease import training_holds_gpu

log = get_logger("api_autolabel")
router = APIRouter()


def _job_dict(j: AutolabelJob) -> dict:
    return {
        "job_id": str(j.job_id), "session_id": str(j.session_id), "status": j.status,
        "executor": (j.params or {}).get("executor", "api"), "attempts": j.attempts,
        "progress": j.progress, "counts": j.counts, "error": j.error,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "updated_at": j.updated_at.isoformat() if j.updated_at else None,
    }


@router.post("/autolabel/start")
async def start(payload: AutolabelStartIn, db: AsyncSession = Depends(db_session)):
    # Both of these guard one thing: the single GPU in this box. A cloud-targeted job never touches it, it
    # is parked for a pod, so refusing to park one because a local job is busy turned a scheduling rule into
    # a queueing rule and made the cloud branch unreachable exactly when somebody would reach for it.
    from services.autolabel.jobs import API, WORKER, job_params, queue_position, run_autolabel_job

    executor = get_settings().jobs.autolabel_executor
    local = payload.compute_target != "cloud"
    if local and executor == API:
        if await training_holds_gpu(db):
            raise HTTPException(503,
                                "GPU reserved for a training job; autolabel is paused until it finishes")
        if (await db.execute(
                select(AutolabelJob.job_id).where(AutolabelJob.status == "running").limit(1))).first():
            raise HTTPException(409, "an autolabel job is already running")
    sess = await db.get(DbSession, uuid.UUID(payload.session_id))
    if sess is None:
        raise HTTPException(404, "session not found")
    # Inspector health gate: a session whose recording failed health checks is excluded from auto-labeling
    # until a human reviews, exactly as calibration validation gates 3D work.
    from services.inspector.health import is_gated

    if await is_gated(db, uuid.UUID(payload.session_id)):
        raise HTTPException(409, "session failed inspector health checks; review it before auto-labeling")
    job_id = uuid.uuid4()
    db.add(AutolabelJob(job_id=job_id, session_id=uuid.UUID(payload.session_id), status="pending",
                        params=job_params(payload.limit, payload.only_unlabelled,
                                          executor if local else "cloud")))
    await db.commit()

    if payload.compute_target == "cloud":
        # Park it for the A100 heavy stack; the local API never runs a cloud job (GPU discipline).
        from services.autolabel.cloud import mark_queued_for_cloud

        await mark_queued_for_cloud(job_id, payload.session_id, payload.limit)
        return {"job_id": str(job_id), "status": "queued-cloud"}

    if executor == WORKER:
        # Queued, not started: the GPU worker claims it. A queue position tells the person who pressed the
        # button why nothing is moving yet, instead of a bar sitting at zero.
        job = await db.get(AutolabelJob, job_id)
        return {"job_id": str(job_id), "status": "pending", "executor": WORKER,
                "queue_position": await queue_position(db, job) if job else None}
    spawn(run_autolabel_job(job_id), name="run_autolabel_job")
    return {"job_id": str(job_id), "status": "pending", "executor": API}


@router.post("/autolabel/ego-masks/estimate")
async def estimate_ego_masks(force: bool = False, db: AsyncSession = Depends(db_session)):
    """Estimate + cache each camera's ego-hood mask from temporal stability. Idempotent unless force=True.
    Fast (no GPU); run once before re-detection so the re-run can drop the hood."""
    from services.autolabel.redetect import estimate_all_ego_masks

    return await estimate_all_ego_masks(force=force)


@router.post("/autolabel/pii-backfill")
async def pii_backfill(limit: int = 2000, session_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Blur faces/plates on frames that predate the anonymization gate (no PII audit), overwriting the stored
    image in place. Closes the DPDPA exposure on the existing corpus. Launched in the background."""
    from services.anonymize.backfill import backfill_unaudited

    async def _go():
        try:
            await backfill_unaudited(limit=limit, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            log.error("pii_backfill.failed", error=str(exc))

    spawn(_go(), name="_go")
    return {"status": "running", "limit": limit}


@router.post("/autolabel/redetect-all")
async def redetect_all(backfill_pii: bool = True, db: AsyncSession = Depends(db_session)):
    """Full re-detection of the corpus with the new gates (thing/stuff, ego-hood, fusion de-dup, oversize),
    plus a PII backfill of pre-gate frames. Sequential on one GPU, yields to training. Background; poll the
    returned run and the per-session autolabel jobs."""
    from db.models import AgentRun
    from services.autolabel.redetect import redetect_and_backfill

    if await training_holds_gpu(db):
        raise HTTPException(503, "GPU reserved for a training job; re-detection is paused until it finishes")
    run_id = uuid.uuid4()
    db.add(AgentRun(run_id=run_id, kind="redetect_all", scope={}, status="running", policy={}, counts={},
                    changes={}, critic={}, created_by="redetect"))
    await db.commit()
    spawn(redetect_and_backfill(run_id, backfill_pii=backfill_pii), name="redetect_and_backfill")
    return {"run_id": str(run_id), "status": "running"}


@router.get("/autolabel/{job_id}")
async def status(job_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    j = await db.get(AutolabelJob, job_id)
    if j is None:
        raise HTTPException(404, "autolabel job not found")
    from services.autolabel.jobs import queue_position

    return {**_job_dict(j), "queue_position": await queue_position(db, j)}
