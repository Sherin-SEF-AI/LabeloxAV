"""Two requests to start GPU work in the API process cannot both find it idle.

`POST /api/autolabel/start` read "is a job running" and then wrote its own row, with nothing between the
two. Two starts arriving together both passed the check and both began on one card. Re-detection checked
training only, so it could start on top of a running auto-label job; the PII backfill checked nothing and
left no row at all.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, update

from db.models import AgentRun, AutolabelJob
from db.session import get_sessionmaker
from services.api.routers import autolabel as mod

pytestmark = pytest.mark.db


@pytest.fixture
async def clean():
    async def _wipe():
        async with get_sessionmaker()() as db:
            await db.execute(update(AutolabelJob).where(AutolabelJob.status.in_(("running", "pending")))
                             .values(status="canceled"))
            await db.execute(delete(AgentRun).where(AgentRun.kind.in_(("redetect_all", "pii_backfill"))))
            await db.commit()
    await _wipe()
    yield
    await _wipe()


async def _job(status: str, executor: str | None = "api", age: timedelta = timedelta()) -> uuid.UUID:
    jid = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(AutolabelJob(job_id=jid, session_id=uuid.uuid4(), status=status,
                            params={"executor": executor} if executor else {},
                            created_at=datetime.now(UTC) - age))
        await db.commit()
    return jid


@pytest.mark.asyncio
async def test_a_job_that_has_been_launched_but_not_started_counts_as_busy(clean):
    async with get_sessionmaker()() as db:
        assert not await mod._api_autolabel_busy(db)
    await _job("pending", "api")
    async with get_sessionmaker()() as db:
        assert await mod._api_autolabel_busy(db), "the gap two simultaneous starts fell through"


@pytest.mark.asyncio
async def test_work_queued_for_the_worker_or_the_cloud_does_not_hold_this_gpu(clean):
    await _job("pending", "worker")
    await _job("pending", "cloud")
    async with get_sessionmaker()() as db:
        assert not await mod._api_autolabel_busy(db)


@pytest.mark.asyncio
async def test_a_pending_row_whose_process_is_long_gone_does_not_block_every_start(clean):
    await _job("pending", "api", age=timedelta(minutes=30))
    async with get_sessionmaker()() as db:
        assert not await mod._api_autolabel_busy(db)


@pytest.mark.asyncio
async def test_a_running_job_is_busy_whoever_runs_it(clean):
    await _job("running", "worker")
    async with get_sessionmaker()() as db:
        assert await mod._api_autolabel_busy(db)


@pytest.mark.asyncio
async def test_the_start_lock_makes_the_second_caller_wait_for_the_first_commit(clean):
    """The property itself: while one transaction holds the lock, another cannot take it."""
    import asyncio

    from sqlalchemy import text

    async with get_sessionmaker()() as first:
        await mod._gpu_start_lock(first)
        async with get_sessionmaker()() as second:
            got = (await second.execute(text("SELECT pg_try_advisory_xact_lock(:k)"),
                                        {"k": mod._GPU_START_LOCK_KEY})).scalar()
            assert got is False
            waiter = asyncio.ensure_future(mod._gpu_start_lock(second))
            await asyncio.sleep(0.2)
            assert not waiter.done(), "the second start must block, not proceed"
            await first.commit()
            await asyncio.wait_for(waiter, timeout=5)
            await second.rollback()


@pytest.mark.asyncio
async def test_a_running_backfill_or_redetection_is_visible(clean):
    async with get_sessionmaker()() as db:
        assert not await mod._redetect_busy(db)
        db.add(AgentRun(run_id=uuid.uuid4(), kind="pii_backfill", scope={}, status="running", policy={},
                        counts={}, changes={}, critic={}, created_by="test"))
        await db.commit()
        assert await mod._redetect_busy(db)
