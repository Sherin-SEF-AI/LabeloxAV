"""The GPU worker: one process that runs every queued GPU job on a single-GPU box, one at a time.

Two queues used to need two answers. Training already had a worker (`services/training/worker.py`) that held
the GPU for its lifetime; auto-label ran inside the API process, which is why the Docker deployment, whose API
image carries no ML libraries, could queue an auto-label and never run it, and why an API restart lost every
job in flight.

This worker drains both queues. It takes the same lifetime lock the training worker takes, so the two can
never run beside each other and fight over the card, and inside that it runs one job at a time:

  1. auto-label jobs queued for the worker (`jobs.autolabel_executor = "worker"`), oldest first. They come
     first because somebody is usually waiting on one; a training run is hours either way.
  2. training jobs, exactly as the training worker runs them, including its VRAM preflight.

On start it requeues whatever its predecessor left `running`. Holding the lifetime lock is what makes that
safe: no other worker can be running those rows. An auto-label job resumes rather than restarts (frames the
crashed run already labelled are skipped), and one that has taken the worker down `jobs.max_attempts` times is
failed with a message saying so instead of crashing it again.

A stop (SIGTERM from `docker compose restart`, Ctrl-C) puts the running auto-label job back in the queue
without counting an attempt, because a deliberate restart is not the job's fault.

    python -m services.gpu_worker              (or: make gpu-worker)
    python -m services.gpu_worker --once       run whatever is queued, then exit
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import uuid
from pathlib import Path

import click
from sqlalchemy import text

from core.config import REPO_ROOT, get_settings
from core.logging import get_logger, setup_logging
from db.models import AutolabelJob, TrainingJob
from db.session import get_engine, get_sessionmaker
from services.autolabel.jobs import claim_autolabel_job, requeue_orphans, run_autolabel_job

log = get_logger("gpu_worker")

# Written by scripts/resource_watch.py while memory, swap or VRAM is over its limit. While it exists the
# worker finishes the job in hand but claims nothing new: starting a second model load on a box that is
# already short is how this workstation froze before the watcher existed.
PAUSE_FLAG = REPO_ROOT / ".scratch" / "watch" / "PAUSE.json"


# Set when a stop interrupted a training run whose executor thread is still going. main() then exits without
# joining it: interpreter shutdown otherwise waits for that thread, which can be hours of epochs, until the
# container runtime gives up and kills the process anyway.
_training_abandoned = False


def paused(flag: Path = PAUSE_FLAG) -> bool:
    return flag.exists()


async def _requeue_interrupted(job_id: uuid.UUID) -> None:
    """Return a job interrupted by a deliberate stop to the queue. No attempt is counted."""
    async with get_sessionmaker()() as db:
        j = await db.get(AutolabelJob, job_id)
        if j is not None and j.status == "running":
            j.status = "pending"
            j.params = {**(j.params or {}), "only_unlabelled": True}
            await db.commit()
            log.info("gpu_worker.requeued_on_stop", job_id=str(job_id))


async def _fail_training(job_id: str, message: str) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
        if j:
            j.status = "error"
            j.error = message
            await db.commit()


async def _lock_alive(lock_conn) -> bool:
    """Whether the connection holding the lifetime lock is still open.

    An advisory lock lives exactly as long as its database session. A Postgres restart, or a restore that
    drops and recreates the database (scripts/restore.sh terminates every session), ends that session while
    this process keeps running and polling on fresh pooled connections. Found exactly that way: after a
    restore the worker carried on claiming jobs with no lock held, so a second worker could have started
    beside it and the one-job-on-the-card guarantee was silently gone.
    """
    try:
        await lock_conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - any failure here means the session, and so the lock, is gone
        return False


async def run_next() -> str | None:
    """Claim and run one job. Returns what ran ("autolabel" / "training"), or None if both queues are empty."""
    from services.training.jobs import run_job_guarded
    from services.training.worker import claim_training_job, vram_preflight

    async with get_sessionmaker()() as db:
        auto_id = await claim_autolabel_job(db)
    if auto_id is not None:
        log.info("gpu_worker.claimed", kind="autolabel", job_id=str(auto_id))
        task = asyncio.ensure_future(run_autolabel_job(auto_id))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            await _requeue_interrupted(auto_id)
            raise
        return "autolabel"

    train_id = await claim_training_job()
    if train_id is not None:
        log.info("gpu_worker.claimed", kind="training", job_id=train_id)
        err = vram_preflight()
        if err:
            await _fail_training(train_id, f"GPU preflight failed: {err}")
            return "training"
        task = asyncio.ensure_future(run_job_guarded(train_id))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Training runs in executor threads, which a cancel cannot reach. Put the job back exactly as a
            # crash would (pending, resuming from the last epoch Ultralytics saved) and let main() exit
            # without joining a thread that may have hours of epochs left.
            from services.training.worker import reset_training_orphans

            global _training_abandoned
            _training_abandoned = True
            await reset_training_orphans()
            log.info("gpu_worker.requeued_on_stop", kind="training", job_id=train_id)
            raise
        return "training"
    return None


async def worker_loop(*, once: bool = False) -> int:
    from services.training.worker import reset_training_orphans

    settings = get_settings()
    setup_logging(settings.log_level)
    key = settings.training.advisory_lock_key
    poll_s = settings.training.worker_poll_s

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    async with get_engine().connect() as lock_conn:
        got = (await lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
        if not got:
            log.error("gpu_worker.lock_held",
                      note="another GPU or training worker is running on this database; exiting")
            return 1
        log.info("gpu_worker.start", poll_s=poll_s, advisory_lock=key,
                 autolabel_executor=settings.jobs.autolabel_executor)
        exit_code = 0
        try:
            async with get_sessionmaker()() as db:
                await requeue_orphans(db, settings.jobs.max_attempts)
            await reset_training_orphans()
            was_paused = False
            while not stop.is_set():
                if not await _lock_alive(lock_conn):
                    # Exit rather than carry on unguarded: the container's restart policy brings the worker
                    # back, and it takes the lock afresh (or finds another worker already has it).
                    log.error("gpu_worker.lock_lost",
                              note="the database session holding the GPU lock ended; exiting to re-acquire it")
                    exit_code = 1
                    break
                if paused():
                    if not was_paused:
                        log.warning("gpu_worker.paused", flag=str(PAUSE_FLAG))
                        was_paused = True
                    ran = None
                else:
                    if was_paused:
                        log.info("gpu_worker.resumed")
                        was_paused = False
                    current = asyncio.ensure_future(run_next())
                    stopper = asyncio.ensure_future(stop.wait())
                    done, _ = await asyncio.wait({current, stopper}, return_when=asyncio.FIRST_COMPLETED)
                    if current not in done:
                        current.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await current
                        break
                    stopper.cancel()
                    ran = current.result()
                if ran is None:
                    if once and not was_paused:
                        break
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(stop.wait(), timeout=poll_s)
        finally:
            # A lost session has already released the lock, and unlocking through it would raise over the
            # real reason this loop ended.
            with contextlib.suppress(Exception):
                await lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                await lock_conn.commit()
            log.info("gpu_worker.stopped", exit_code=exit_code)
    return exit_code


@click.command()
@click.option("--once", is_flag=True, help="Run whatever is queued, then exit.")
def main(once: bool) -> None:
    code = asyncio.run(worker_loop(once=once))
    if _training_abandoned:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
