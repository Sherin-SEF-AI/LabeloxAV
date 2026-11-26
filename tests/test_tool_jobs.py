"""The annotation tools' shared job queue (services/tools/registry.py, migration 0126).

What it has to get right: a CPU kind runs to a terminal state in the background and records its result or its
failure; a GPU kind waits for the GPU worker; an auto-labelling kind is not started during office hours; a
cancel stops it; and a job shows on /jobs like every other kind.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from db.models import ToolJob
from db.session import get_sessionmaker
from services.tools import registry as R

pytestmark = pytest.mark.db


@R.register("_test_ok", label="Test job")
async def _ok(ctx: R.JobContext) -> dict:
    await ctx.progress(0.5, "halfway")
    return {"echo": ctx.params.get("x")}


@R.register("_test_boom")
async def _boom(ctx: R.JobContext) -> dict:
    raise ValueError("no frames")


@R.register("_test_gpu", gpu=True, autolabel=True)
async def _gpu(ctx: R.JobContext) -> dict:
    return {"ran": True}


@R.register("_test_gpu_plain", gpu=True)
async def _gpu_plain(ctx: R.JobContext) -> dict:
    return {"ran": True}


@R.register("_test_slow")
async def _slow(ctx: R.JobContext) -> dict:
    for _ in range(50):
        await ctx.progress(0.1, "working")
        await asyncio.sleep(0.05)
    return {}


async def _wait(job_id: uuid.UUID, timeout: float = 10.0) -> ToolJob:
    t0 = asyncio.get_running_loop().time()
    while True:
        async with get_sessionmaker()() as db:
            job = await db.get(ToolJob, job_id)
        # A cancel is terminal the moment the row changes; the runner adds the finish time when it notices.
        if (job.status in R.TERMINAL and job.finished_at is not None) or asyncio.get_running_loop().time() - t0 > timeout:
            return job
        await asyncio.sleep(0.05)


IST_NOON = datetime(2026, 11, 26, 6, 30, tzinfo=UTC)      # 12:00 IST
IST_NIGHT = datetime(2026, 11, 26, 17, 30, tzinfo=UTC)    # 23:00 IST


class TestQuietHours:
    def test_office_hours_are_quiet(self):
        assert R.in_quiet_hours(IST_NOON, "10:00-18:00")
        assert not R.in_quiet_hours(IST_NIGHT, "10:00-18:00")

    def test_the_window_is_half_hour_precise_in_ist(self):
        # 04:29 UTC is 09:59 IST, 04:30 UTC is 10:00 IST.
        assert not R.in_quiet_hours(datetime(2026, 11, 26, 4, 29, tzinfo=UTC), "10:00-18:00")
        assert R.in_quiet_hours(datetime(2026, 11, 26, 4, 30, tzinfo=UTC), "10:00-18:00")

    def test_a_window_across_midnight_and_an_empty_one(self):
        assert R.in_quiet_hours(IST_NIGHT, "22:00-06:00")
        assert not R.in_quiet_hours(IST_NOON, "")


class TestCpuJobs:
    async def test_a_cpu_job_runs_and_records_its_result(self):
        async with get_sessionmaker()() as db:
            job = await R.enqueue(db, "_test_ok", {"x": 7})
        done = await _wait(job.job_id)
        assert done.status == "done"
        assert done.result == {"echo": 7}
        assert done.progress == 1.0
        assert done.finished_at is not None

    async def test_a_failure_is_recorded_not_raised(self):
        async with get_sessionmaker()() as db:
            job = await R.enqueue(db, "_test_boom")
        done = await _wait(job.job_id)
        assert done.status == "error"
        assert "no frames" in done.error

    async def test_a_cancel_stops_a_running_job_at_its_next_checkpoint(self):
        async with get_sessionmaker()() as db:
            job = await R.enqueue(db, "_test_slow")
        await asyncio.sleep(0.2)
        async with get_sessionmaker()() as db:
            row = await db.get(ToolJob, job.job_id)
            row.status = "canceled"
            await db.commit()
        done = await _wait(job.job_id)
        assert done.status == "canceled"
        assert done.finished_at is not None

    async def test_an_unknown_kind_is_refused(self):
        async with get_sessionmaker()() as db:
            with pytest.raises(KeyError):
                await R.enqueue(db, "_no_such_kind")


class TestGpuJobs:
    async def _clear_gpu_queue(self):
        from sqlalchemy import update

        async with get_sessionmaker()() as db:
            await db.execute(update(ToolJob).where(ToolJob.status == "pending", ToolJob.gpu.is_(True))
                             .values(status="canceled"))
            await db.commit()

    async def test_a_gpu_job_waits_for_the_worker(self):
        await self._clear_gpu_queue()
        async with get_sessionmaker()() as db:
            job = await R.enqueue(db, "_test_gpu_plain")
        await asyncio.sleep(0.2)
        async with get_sessionmaker()() as db:
            assert (await db.get(ToolJob, job.job_id)).status == "pending"
            claimed = await R.claim_gpu_job(db, IST_NOON)
        assert claimed == job.job_id
        assert await R.run_job(claimed) == "done"

    async def test_auto_labelling_does_not_start_in_office_hours(self):
        await self._clear_gpu_queue()
        async with get_sessionmaker()() as db:
            job = await R.enqueue(db, "_test_gpu")
            assert await R.claim_gpu_job(db, IST_NOON) is None
        async with get_sessionmaker()() as db:
            assert await R.claim_gpu_job(db, IST_NIGHT) == job.job_id

    async def test_orphans_are_requeued_then_given_up_on(self):
        async with get_sessionmaker()() as db:
            a = ToolJob(kind="_test_gpu_plain", gpu=True, status="running", attempts=1)
            b = ToolJob(kind="_test_gpu_plain", gpu=True, status="running", attempts=3)
            db.add_all([a, b])
            await db.commit()
            await R.requeue_orphans(db, gpu=True, max_attempts=3)
            await db.refresh(a)
            await db.refresh(b)
        assert a.status == "pending"
        assert b.status == "error"


class TestRoutes:
    def test_a_tool_job_is_readable_listed_and_cancellable_by_its_owner(self):
        from _authutil import headers_for, seed_user
        from fastapi.testclient import TestClient

        from services.api.main import app

        uid = seed_user("annotator")
        other = seed_user("annotator")

        async def _make() -> str:
            async with get_sessionmaker()() as db:
                job = ToolJob(kind="_test_gpu_plain", gpu=True, created_by=uuid.UUID(uid))
                db.add(job)
                await db.commit()
                return str(job.job_id)

        from _authutil import _clear_db_cache
        _clear_db_cache()
        jid = asyncio.run(_make())
        _clear_db_cache()
        with TestClient(app) as c:
            r = c.get(f"/api/tools/jobs/{jid}", headers=headers_for(uid))
            assert r.status_code == 200, r.text
            assert r.json()["status"] == "pending"
            assert any(j["job_id"] == jid for j in c.get("/api/tools/jobs?kind=_test_gpu_plain",
                                                         headers=headers_for(uid)).json())
            assert any(j["job_id"] == jid and j["kind"] == "tool"
                       for j in c.get("/api/jobs?limit=500", headers=headers_for(uid)).json())
            assert c.post(f"/api/tools/jobs/{jid}/cancel", headers=headers_for(other)).status_code == 403
            r = c.post(f"/api/tools/jobs/{jid}/cancel", headers=headers_for(uid))
            assert r.status_code == 200 and r.json()["status"] == "canceled"
