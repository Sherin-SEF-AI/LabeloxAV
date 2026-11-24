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
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.observability import spawn
from db.models import AutolabelJob
from db.models import Session as DbSession
from services.api.deps import ApiOut, AutolabelStartIn, db_session
from services.training.gpu_lease import training_holds_gpu

log = get_logger("api_autolabel")
router = APIRouter()


class AutolabelJobOut(ApiOut):
    job_id: str
    session_id: str
    status: str
    executor: str                               # api | worker | cloud
    attempts: int
    progress: float
    counts: dict[str, Any] | None
    error: str | None
    created_at: str | None
    updated_at: str | None
    # How many worker jobs are ahead of this one. Null whenever the job is not waiting in the worker's
    # queue, which is every job on an installation where the API runs auto-label itself.
    queue_position: int | None


def _job_dict(j: AutolabelJob) -> dict:
    return {
        "job_id": str(j.job_id), "session_id": str(j.session_id), "status": j.status,
        "executor": (j.params or {}).get("executor", "api"), "attempts": j.attempts,
        "progress": j.progress, "counts": j.counts, "error": j.error,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "updated_at": j.updated_at.isoformat() if j.updated_at else None,
    }


# One key for every route in this module that starts GPU work in the API process.
_GPU_START_LOCK_KEY = 815_002
# How long a pending API-executor row is taken to be a job that is about to start.
_LAUNCH_GAP = timedelta(minutes=2)


async def _gpu_start_lock(db: AsyncSession) -> None:
    """Take the transaction-scoped lock that makes a busy check and the row it guards one step."""
    from sqlalchemy import text

    await db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _GPU_START_LOCK_KEY})


async def _api_autolabel_busy(db: AsyncSession) -> bool:
    """Whether an auto-label job is running, or about to, in an API process.

    `pending` counts when the API is the executor: that row is a job whose task has been launched and
    has not yet written `running`, which is exactly the gap two simultaneous starts fell through. A
    pending row queued for the worker or parked for the cloud is somebody else's and does not hold this
    process's GPU. Nor does one that has sat pending for minutes: the launch gap is milliseconds, so an
    old pending row is a job whose process went away before it began, and letting it block every start
    until the reaper fails it would turn one lost job into an outage.
    """
    from datetime import UTC, datetime

    from services.autolabel.jobs import API

    rows = (await db.execute(select(AutolabelJob.status, AutolabelJob.params, AutolabelJob.created_at)
                             .where(AutolabelJob.status.in_(("running", "pending"))))).all()
    fresh_after = datetime.now(UTC) - _LAUNCH_GAP
    for st, params, created in rows:
        if st == "running":
            return True
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if (params or {}).get("executor", API) == API and created is not None and created > fresh_after:
            return True
    return False


async def _redetect_busy(db: AsyncSession) -> bool:
    from db.models import AgentRun

    return (await db.execute(select(AgentRun.run_id).where(
        AgentRun.kind.in_(("redetect_all", "pii_backfill")), AgentRun.status == "running").limit(1))
    ).first() is not None


@router.post("/autolabel/start")
async def start(payload: AutolabelStartIn, db: AsyncSession = Depends(db_session)):
    # Both of these guard one thing: the single GPU in this box. A cloud-targeted job never touches it, it
    # is parked for a pod, so refusing to park one because a local job is busy turned a scheduling rule into
    # a queueing rule and made the cloud branch unreachable exactly when somebody would reach for it.
    from services.autolabel.jobs import API, WORKER, job_params, queue_position, run_autolabel_job

    executor = get_settings().jobs.autolabel_executor
    local = payload.compute_target != "cloud"
    if local and executor == API:
        # Serialise "is anything running" with the insert below. They were a read and then a write with
        # nothing between them, so two starts arriving together both saw an idle GPU and both began. The
        # lock is released by the commit that writes the row, by which point the second caller's check
        # can see it.
        await _gpu_start_lock(db)
        if await training_holds_gpu(db):
            raise HTTPException(503,
                                "GPU reserved for a training job; autolabel is paused until it finishes")
        if await _api_autolabel_busy(db):
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
async def pii_backfill(limit: Annotated[int, Query(ge=1, le=40_000)] = 2000, session_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Blur faces/plates on frames that predate the anonymization gate (no PII audit), overwriting the stored
    image in place. Closes the DPDPA exposure on the existing corpus. Launched in the background."""
    from db.models import AgentRun
    from db.session import get_sessionmaker
    from services.anonymize.backfill import backfill_unaudited

    # This overwrites stored images with detector passes over thousands of frames. It had no row, so
    # nothing showed it was running, a second press started a second pass over the same frames, and it
    # would start beside a training run. It now has a run row, refuses while the GPU is taken, and one
    # runs at a time.
    await _gpu_start_lock(db)
    if await training_holds_gpu(db):
        raise HTTPException(503, "GPU reserved for a training job; the PII backfill is paused until it finishes")
    if await _redetect_busy(db):
        raise HTTPException(409, "a PII backfill or re-detection is already running")
    run_id = uuid.uuid4()
    db.add(AgentRun(run_id=run_id, kind="pii_backfill", scope={"session_id": session_id, "limit": limit},
                    status="running", policy={}, counts={}, changes={}, critic={}, created_by="pii_backfill"))
    await db.commit()

    async def _finish(status: str, counts: dict | None = None, error: str | None = None) -> None:
        async with get_sessionmaker()() as fdb:
            run = await fdb.get(AgentRun, run_id)
            if run is not None:
                run.status = status
                if counts is not None:
                    run.counts = counts
                if error:
                    run.critic = {"error": error}
                await fdb.commit()

    async def _go():
        try:
            res = await backfill_unaudited(limit=limit, session_id=session_id)
            await _finish("committed", res if isinstance(res, dict) else {"result": res})
        except Exception as exc:  # noqa: BLE001
            log.error("pii_backfill.failed", error=str(exc))
            await _finish("error", error=str(exc))

    spawn(_go(), name="pii_backfill", run_id=str(run_id))
    return {"status": "running", "limit": limit, "run_id": str(run_id)}


@router.post("/autolabel/redetect-all")
async def redetect_all(backfill_pii: bool = True, db: AsyncSession = Depends(db_session)):
    """Full re-detection of the corpus with the new gates (thing/stuff, ego-hood, fusion de-dup, oversize),
    plus a PII backfill of pre-gate frames. Sequential on one GPU, yields to training. Background; poll the
    returned run and the per-session autolabel jobs."""
    from db.models import AgentRun
    from services.autolabel.redetect import redetect_and_backfill

    await _gpu_start_lock(db)
    if await training_holds_gpu(db):
        raise HTTPException(503, "GPU reserved for a training job; re-detection is paused until it finishes")
    # It checked training only, so a corpus-wide re-detection could start on top of a running auto-label
    # job, or on top of itself.
    if await _api_autolabel_busy(db):
        raise HTTPException(409, "an autolabel job is running; re-detection would share the GPU with it")
    if await _redetect_busy(db):
        raise HTTPException(409, "a re-detection or PII backfill is already running")
    run_id = uuid.uuid4()
    db.add(AgentRun(run_id=run_id, kind="redetect_all", scope={}, status="running", policy={}, counts={},
                    changes={}, critic={}, created_by="redetect"))
    await db.commit()
    spawn(redetect_and_backfill(run_id, backfill_pii=backfill_pii), name="redetect_and_backfill")
    return {"run_id": str(run_id), "status": "running"}


@router.get("/autolabel/{job_id}", response_model=AutolabelJobOut, response_model_exclude_unset=True)
async def status(job_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    j = await db.get(AutolabelJob, job_id)
    if j is None:
        raise HTTPException(404, "autolabel job not found")
    from services.autolabel.jobs import queue_position

    return {**_job_dict(j), "queue_position": await queue_position(db, j)}
