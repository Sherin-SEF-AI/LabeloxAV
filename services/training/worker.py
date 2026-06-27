"""Training worker: a separate process that drains the training_job queue serially on the GPU.

Why a separate process: training is a multi-hour blocking call and would freeze the API event loop;
and the cached async engine is bound to one loop per process. One worker = natural GPU serialization.
The worker holds a Postgres advisory lock for its lifetime (the GPU mutex), so a second worker refuses
to start, and on boot it resets jobs orphaned by a previous crash.

    python -m services.training.worker         (or: make train-worker)
"""

from __future__ import annotations

import asyncio

import click
from sqlalchemy import select, text, update

from core.config import get_settings
from core.logging import get_logger, setup_logging
from db.models import TrainingJob
from db.session import get_engine, get_sessionmaker
from services.training.jobs import run_job_guarded

log = get_logger("train_worker")


async def _reset_orphans() -> None:
    async with get_sessionmaker()() as db:
        res = await db.execute(update(TrainingJob).where(TrainingJob.status == "running").values(status="pending"))
        await db.commit()
        if res.rowcount:
            log.info("worker.reset_orphans", n=res.rowcount)


async def _claim() -> str | None:
    """Atomically claim the oldest pending LOCAL job (FOR UPDATE SKIP LOCKED), marking it running.
    Cloud jobs (compute_target='cloud') are intentionally skipped: they run on the RunPod A100."""
    async with get_sessionmaker()() as db:
        async with db.begin():
            row = (await db.execute(
                select(TrainingJob)
                .where(TrainingJob.status == "pending", TrainingJob.compute_target == "local")
                .order_by(TrainingJob.created_at).limit(1).with_for_update(skip_locked=True)
            )).scalar_one_or_none()
            if row is None:
                return None
            row.status = "running"
            return str(row.job_id)


def _vram_preflight() -> str | None:
    """Return None if there is enough free VRAM to train, else an error message."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None  # let the train step fail with a clear CUDA message
        from services.autolabel.runner import VramGuard

        VramGuard().require(get_settings().training.vram_required_mb, "training")
        return None
    except Exception as exc:  # noqa: BLE001
        log.error("worker.vram_preflight_failed", error=str(exc))
        return str(exc)


async def worker_loop() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    key = settings.training.advisory_lock_key
    poll_s = settings.training.worker_poll_s

    # Hold the GPU mutex for the worker's lifetime; a second worker refuses to start.
    async with get_engine().connect() as lock_conn:
        got = (await lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
        if not got:
            log.error("worker.gpu_lock_held", note="another training worker is running; exiting")
            return
        log.info("worker.start", poll_s=poll_s, advisory_lock=key)
        await _reset_orphans()
        try:
            while True:
                job_id = await _claim()
                if job_id is None:
                    await asyncio.sleep(poll_s)
                    continue
                log.info("worker.claimed", job_id=job_id)
                vram_err = _vram_preflight()
                if vram_err:
                    async with get_sessionmaker()() as db:
                        j = await db.get(TrainingJob, __import__("uuid").UUID(job_id))
                        if j:
                            j.status = "error"
                            j.error = f"GPU preflight failed: {vram_err}"
                            await db.commit()
                    continue
                await run_job_guarded(job_id)
        finally:
            await lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            await lock_conn.commit()


@click.command()
def main() -> None:
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()
