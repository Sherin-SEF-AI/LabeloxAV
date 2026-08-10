"""A dead training job must not hold the GPU forever.

Ten GPU-using features yielded to training by asking whether any TrainingJob row said `running`. A status
column records intent, not liveness, so a run killed by a crash, an OOM or a stopped container left its row
at `running` and permanently disabled interactive segmentation, autolabel, embedding, redetection, VLM QA
and the relabel agent. This is the case that was live in the corpus: a job stuck at epoch 45 of 60 with its
process long gone, returning 503 from /segment a day later.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest


async def _job(db, *, status: str, age: timedelta, purpose: str):
    from db.models import TrainingJob

    job_id = uuid.uuid4()
    db.add(TrainingJob(job_id=job_id, purpose=purpose, task_type="detection", status=status,
                       compute_target="local", config={}, progress=0.5))
    await db.flush()
    # updated_at carries onupdate=func.now(), so it cannot be set through the ORM on the same flush that
    # created the row. Age it in SQL, which is also what a stalled heartbeat looks like from the outside.
    from sqlalchemy import text
    await db.execute(text("update training_job set updated_at = :ts where job_id = :j"),
                     {"ts": datetime.now(UTC) - age, "j": job_id})
    return job_id


@pytest.mark.asyncio
async def test_a_live_job_holds_the_gpu():
    from db.session import get_sessionmaker
    from services.training.gpu_lease import training_holds_gpu

    async with get_sessionmaker()() as db:
        await _job(db, status="running", age=timedelta(seconds=30), purpose="lease-live")
        assert await training_holds_gpu(db) is True
        await db.rollback()


@pytest.mark.asyncio
async def test_a_job_whose_heartbeat_stopped_does_not():
    """The defect. Without this the GPU is claimed by a process that no longer exists."""
    from db.session import get_sessionmaker
    from services.training.gpu_lease import STALE_AFTER, training_holds_gpu

    async with get_sessionmaker()() as db:
        await _job(db, status="running", age=STALE_AFTER + timedelta(minutes=5), purpose="lease-dead")
        assert await training_holds_gpu(db) is False, \
            "a running row with a stopped heartbeat must not reserve the GPU forever"
        await db.rollback()


@pytest.mark.asyncio
async def test_a_live_job_still_wins_alongside_a_dead_one():
    """Recovering from one crashed job must not release the GPU out from under a real run."""
    from db.session import get_sessionmaker
    from services.training.gpu_lease import STALE_AFTER, training_holds_gpu

    async with get_sessionmaker()() as db:
        await _job(db, status="running", age=STALE_AFTER + timedelta(hours=3), purpose="lease-dead-2")
        await _job(db, status="running", age=timedelta(seconds=5), purpose="lease-live-2")
        assert await training_holds_gpu(db) is True
        await db.rollback()


@pytest.mark.asyncio
async def test_finished_jobs_never_hold_the_gpu():
    from db.session import get_sessionmaker
    from services.training.gpu_lease import training_holds_gpu

    async with get_sessionmaker()() as db:
        for status in ("done", "error", "canceled", "pending"):
            await _job(db, status=status, age=timedelta(seconds=1), purpose=f"lease-{status}")
        assert await training_holds_gpu(db) is False
        await db.rollback()


@pytest.mark.asyncio
async def test_the_refusal_message_is_absent_when_the_gpu_is_free():
    """Callers branch on this being None, so an empty corpus must not read as busy."""
    from db.session import get_sessionmaker
    from services.training.gpu_lease import gpu_busy_detail

    async with get_sessionmaker()() as db:
        assert await gpu_busy_detail(db) is None
        await _job(db, status="running", age=timedelta(seconds=2), purpose="lease-msg")
        detail = await gpu_busy_detail(db)
        assert detail and "training job" in detail
        await db.rollback()


@pytest.mark.asyncio
async def test_the_old_check_would_have_stayed_blocked():
    """What the ten call sites did before, kept executable so the regression stays demonstrable."""
    from sqlalchemy import select

    from db.models import TrainingJob
    from db.session import get_sessionmaker
    from services.training.gpu_lease import STALE_AFTER, training_holds_gpu

    async with get_sessionmaker()() as db:
        await _job(db, status="running", age=STALE_AFTER + timedelta(days=1), purpose="lease-old-check")

        old_says_busy = (await db.execute(
            select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first() is not None
        assert old_says_busy, "precondition: the status column still says running"
        assert await training_holds_gpu(db) is False, "the heartbeat says otherwise, and it is right"
        await db.rollback()
