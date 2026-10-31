"""One auto-label job, run from its row, by whichever process claimed it.

The API used to hold the whole job as an asyncio task: the row carried status and progress, while the
request itself (frame limit, skip frames that already have labels) lived only in the task's arguments. That
made the API process the one place the job could run, so a restart lost it and a Docker deployment, whose
API image carries no ML libraries, could not run it at all.

Now the request is on the row (`params`, migration 0117) and this module is the single runner. The API calls
it in-process when `jobs.autolabel_executor` is "api"; the GPU worker (`services/gpu_worker.py`) claims rows
and calls it when the executor is "worker". Both paths run exactly this code, so they cannot drift.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AutolabelJob
from db.session import get_sessionmaker

log = get_logger("autolabel.jobs")

API = "api"
WORKER = "worker"

# Why the row was failed after its worker kept dying on it. Reads first in the jobs list, so it says what to
# do next rather than only what happened.
GAVE_UP = ("stopped after {n} attempts: the GPU worker crashed or was restarted each time this job ran. "
           "Check the worker log before starting it again.")


def job_params(limit: int | None, only_unlabelled: bool, executor: str) -> dict:
    return {"limit": limit, "only_unlabelled": bool(only_unlabelled), "executor": executor}


async def _bump(job_id: uuid.UUID, **fields) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(AutolabelJob, job_id)
        if j:
            for k, v in fields.items():
                setattr(j, k, v)
            await db.commit()


async def run_autolabel_job(job_id: uuid.UUID) -> None:
    """Run one job to a terminal status. Never raises: the outcome is written to the row."""
    from services.autolabel.progress import ProgressBand
    from services.autolabel.runner import autolabel_session
    from services.job_control import JobCanceled, raise_if_canceled

    async with get_sessionmaker()() as db:
        job = await db.get(AutolabelJob, job_id)
        if job is None:
            log.warning("autolabel.job_missing", job_id=str(job_id))
            return
        session_id, params = job.session_id, dict(job.params or {})

    await _bump(job_id, status="running", progress=0.05)
    band = ProgressBand()

    async def report(done: int, total: int) -> None:
        # Rationed by the band: one write per percent, not one per frame. A five thousand frame session
        # would otherwise commit five thousand times to move a bar a few pixels.
        f = band.next_if_due(done, total)
        if f is None:
            return
        # The same write asks whether the job is still wanted. A cancel that only set a column would leave
        # this loop labelling for another twenty minutes on a GPU somebody has taken back.
        async with get_sessionmaker()() as db:
            await raise_if_canceled(db, AutolabelJob, job_id, progress=f)

    try:
        result = await autolabel_session(session_id, params.get("limit"), on_progress=report,
                                         only_unlabelled=bool(params.get("only_unlabelled")))
        await _bump(job_id, status="done", progress=1.0, counts=result)
        log.info("autolabel.done", job_id=str(job_id),
                 **{k: result[k] for k in result if k in ("n_frames", "n_objects")})
    except JobCanceled:
        # Not a failure. The row already says `canceled`; overwriting it with `error` would turn somebody's
        # deliberate decision into a fault report.
        log.info("autolabel.canceled", job_id=str(job_id))
    except Exception as exc:  # noqa: BLE001
        log.error("autolabel.failed", job_id=str(job_id), error=str(exc))
        await _bump(job_id, status="error", error=str(exc))


def _worker_rows():
    """Rows queued for the GPU worker. A row the API runs itself is never touched: between its commit and
    the task starting it reads `pending`, and claiming it there would run the same job twice."""
    return AutolabelJob.params["executor"].astext == WORKER


async def claim_autolabel_job(db: AsyncSession) -> uuid.UUID | None:
    """Atomically claim the oldest queued worker job, marking it running. None when the queue is empty.

    FOR UPDATE SKIP LOCKED, so two claimers can never take the same row; the GPU worker's lifetime lock
    means there is only one claimer anyway, and this keeps that true even if somebody starts a second.
    """
    async with db.begin():
        row = (await db.execute(
            select(AutolabelJob)
            .where(AutolabelJob.status == "pending", _worker_rows())
            .order_by(AutolabelJob.created_at).limit(1).with_for_update(skip_locked=True)
        )).scalar_one_or_none()
        if row is None:
            return None
        row.status = "running"
        row.progress = max(float(row.progress or 0.0), 0.01)
        return row.job_id


async def requeue_orphans(db: AsyncSession, max_attempts: int) -> dict[str, list[str]]:
    """Put worker jobs a crash left `running` back in the queue, or fail them once they have used up their
    attempts. Only safe to call while holding the worker's lifetime lock, which is what makes every `running`
    worker row an orphan: no other worker can be running it.

    A requeued job resumes rather than restarts. It is switched to only_unlabelled, so frames the crashed
    run already labelled are skipped instead of being labelled a second time.
    """
    rows = (await db.execute(
        select(AutolabelJob).where(AutolabelJob.status == "running", _worker_rows()))).scalars().all()
    requeued: list[str] = []
    failed: list[str] = []
    for r in rows:
        attempts = int(r.attempts or 0) + 1
        r.attempts = attempts
        if attempts >= max_attempts:
            r.status = "error"
            r.error = GAVE_UP.format(n=attempts)
            failed.append(str(r.job_id))
            continue
        r.status = "pending"
        r.params = {**(r.params or {}), "only_unlabelled": True}
        requeued.append(str(r.job_id))
    await db.commit()
    if rows:
        log.warning("autolabel.orphans", requeued=len(requeued), failed=len(failed))
    return {"requeued": requeued, "failed": failed}


async def queue_position(db: AsyncSession, job: AutolabelJob) -> int | None:
    """How many worker jobs are ahead of this one, or None when it is not waiting in the worker queue."""
    if job.status != "pending" or (job.params or {}).get("executor") != WORKER:
        return None
    ahead = (await db.execute(
        select(func.count()).select_from(AutolabelJob)
        .where(AutolabelJob.status.in_(("pending", "running")), _worker_rows(),
               AutolabelJob.created_at < job.created_at))).scalar_one()
    return int(ahead)


__all__ = ["API", "WORKER", "claim_autolabel_job", "job_params", "queue_position", "requeue_orphans",
           "run_autolabel_job"]
