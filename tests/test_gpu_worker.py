"""Auto-label as a queued job the GPU worker runs, instead of a task that lives and dies with the API.

The Docker deployment's API image carries no ML libraries, so an auto-label started there was queued and
never run, and on any install an API restart lost every job in flight. These pin the queue: the API only
writes the row in worker mode, the worker claims only rows meant for it, a crash resumes the job instead of
losing it, a job that keeps crashing the worker is stopped, and a deliberate stop costs the job nothing.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete

from core.timebase import now_ns, seconds_to_ns
from db.models import AutolabelJob, TrainingJob
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.jobs import (
    GAVE_UP,
    claim_autolabel_job,
    job_params,
    queue_position,
    requeue_orphans,
)

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _empty_worker_queue():
    """Rows from other tests would be claimed first, since the queue is oldest-first."""
    async def clear():
        async with get_sessionmaker()() as db:
            await db.execute(delete(AutolabelJob).where(AutolabelJob.status.in_(("pending", "running"))))
            await db.execute(delete(TrainingJob).where(TrainingJob.status.in_(("pending", "running"))))
            await db.commit()

    asyncio.run(clear())
    from db.session import get_engine
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


async def _job(db, *, status="pending", executor="worker", attempts=0, age_s=0.0, only_unlabelled=False):
    jid = uuid.uuid4()
    db.add(AutolabelJob(job_id=jid, session_id=uuid.uuid4(), status=status, attempts=attempts,
                        params=job_params(None, only_unlabelled, executor)))
    await db.commit()
    if age_s:
        await db.execute(AutolabelJob.__table__.update().where(AutolabelJob.job_id == jid)
                         .values(created_at=datetime.now(UTC) - timedelta(seconds=age_s)))
        await db.commit()
    return jid


async def _row(jid):
    async with get_sessionmaker()() as db:
        return await db.get(AutolabelJob, jid)


class TestClaiming:
    async def test_the_oldest_queued_job_is_claimed_and_marked_running(self):
        async with get_sessionmaker()() as db:
            newer = await _job(db)
            older = await _job(db, age_s=60)
        async with get_sessionmaker()() as db:
            assert await claim_autolabel_job(db) == older
        assert (await _row(older)).status == "running"
        assert (await _row(newer)).status == "pending"

    async def test_a_job_the_api_runs_itself_is_never_claimed(self):
        """Between the API's commit and its task starting, its row reads `pending`. Claiming it there would
        run one job twice on one GPU."""
        async with get_sessionmaker()() as db:
            await _job(db, executor="api")
            await _job(db, executor="cloud")
        async with get_sessionmaker()() as db:
            assert await claim_autolabel_job(db) is None

    async def test_an_empty_queue_claims_nothing(self):
        async with get_sessionmaker()() as db:
            assert await claim_autolabel_job(db) is None

    async def test_two_claimers_never_take_the_same_job(self):
        async with get_sessionmaker()() as db:
            only = await _job(db)

        async def claim():
            async with get_sessionmaker()() as db:
                return await claim_autolabel_job(db)

        got = await asyncio.gather(claim(), claim())
        assert sorted(map(str, filter(None, got))) == [str(only)]


class TestAfterACrash:
    async def test_an_orphan_goes_back_in_the_queue_and_resumes(self):
        """It resumes rather than restarts: frames the crashed run already labelled are skipped."""
        async with get_sessionmaker()() as db:
            jid = await _job(db, status="running")
        async with get_sessionmaker()() as db:
            out = await requeue_orphans(db, max_attempts=3)
        row = await _row(jid)
        assert out["requeued"] == [str(jid)]
        assert row.status == "pending" and row.attempts == 1
        assert row.params["only_unlabelled"] is True

    async def test_a_job_that_keeps_crashing_the_worker_is_stopped(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db, status="running", attempts=2)
        async with get_sessionmaker()() as db:
            out = await requeue_orphans(db, max_attempts=3)
        row = await _row(jid)
        assert out["failed"] == [str(jid)]
        assert row.status == "error" and row.error == GAVE_UP.format(n=3)

    async def test_a_running_api_job_is_not_the_workers_to_requeue(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db, status="running", executor="api")
        async with get_sessionmaker()() as db:
            await requeue_orphans(db, max_attempts=3)
        assert (await _row(jid)).status == "running"


class TestTheWorker:
    async def test_it_runs_the_claimed_job(self, monkeypatch):
        import services.gpu_worker as gw

        ran: list[uuid.UUID] = []

        async def fake_run(job_id):
            ran.append(job_id)

        monkeypatch.setattr(gw, "run_autolabel_job", fake_run)
        async with get_sessionmaker()() as db:
            jid = await _job(db)
        assert await gw.run_next() == "autolabel"
        assert ran == [jid]

    async def test_a_deliberate_stop_requeues_the_job_without_counting_an_attempt(self, monkeypatch):
        """`docker compose restart` is not the job's fault, so it must not move the job toward giving up."""
        import services.gpu_worker as gw

        started = asyncio.Event()

        async def slow_run(job_id):
            started.set()
            await asyncio.sleep(60)

        monkeypatch.setattr(gw, "run_autolabel_job", slow_run)
        async with get_sessionmaker()() as db:
            jid = await _job(db)
        task = asyncio.ensure_future(gw.run_next())
        await asyncio.wait_for(started.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        row = await _row(jid)
        assert row.status == "pending" and row.attempts == 0
        assert row.params["only_unlabelled"] is True

    async def test_a_stop_during_training_requeues_it_to_resume(self, monkeypatch):
        """Training runs in executor threads a cancel cannot reach, so the row has to be put back the way a
        crash would leave it: pending, resuming from the last saved epoch."""
        import services.gpu_worker as gw
        import services.training.jobs as tj
        import services.training.worker as tw

        started = asyncio.Event()

        async def slow_train(job_id):
            started.set()
            await asyncio.sleep(60)

        monkeypatch.setattr(tj, "run_job_guarded", slow_train)
        monkeypatch.setattr(tw, "vram_preflight", lambda: None)
        monkeypatch.setattr(gw, "_training_abandoned", False)
        jid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(TrainingJob(job_id=jid, purpose="stop-test", config={"hparams": {"epochs": 3}}))
            await db.commit()
        task = asyncio.ensure_future(gw.run_next())
        await asyncio.wait_for(started.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with get_sessionmaker()() as db:
            row = await db.get(TrainingJob, jid)
        assert row.status == "pending"
        assert gw._training_abandoned is True

    async def test_it_claims_nothing_new_while_resources_are_paused(self, tmp_path, monkeypatch):
        import services.gpu_worker as gw

        flag = tmp_path / "PAUSE.json"
        assert not gw.paused(flag)
        flag.write_text("{}")
        assert gw.paused(flag)


class TestQueuePosition:
    async def test_it_counts_the_worker_jobs_ahead(self):
        async with get_sessionmaker()() as db:
            await _job(db, status="running", age_s=120)
            await _job(db, age_s=60)
            await _job(db, executor="api", age_s=90)
            mine = await _job(db)
        async with get_sessionmaker()() as db:
            assert await queue_position(db, await db.get(AutolabelJob, mine)) == 2

    async def test_a_job_not_waiting_for_the_worker_has_no_position(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db, executor="api")
        async with get_sessionmaker()() as db:
            assert await queue_position(db, await db.get(AutolabelJob, jid)) is None


class TestTheReaperLeavesWorkerJobsAlone:
    def test_autolabel_drops_out_of_the_sweep_in_worker_mode(self, monkeypatch):
        from core.config import get_settings
        from services.job_reaper import job_kinds

        assert "autolabel" in {k.name for k in job_kinds()}
        monkeypatch.setenv("LBX_JOBS__AUTOLABEL_EXECUTOR", "worker")
        get_settings.cache_clear()
        assert "autolabel" not in {k.name for k in job_kinds()}


def _client():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    c = TestClient(app)
    c.headers.update(auth_headers("admin"))
    return c


async def _session() -> str:
    sid = uuid.uuid4()
    start = now_ns()
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="QUEUE-01", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version="labelox-in-0.1.0"))
        await db.commit()
    return str(sid)


def _worker_mode(monkeypatch):
    from core.config import get_settings

    monkeypatch.setenv("LBX_JOBS__AUTOLABEL_EXECUTOR", "worker")
    get_settings.cache_clear()


class TestStartingAJob:
    def test_worker_mode_queues_and_never_refuses_for_a_busy_gpu(self, monkeypatch):
        """In-process mode refuses while another auto-label runs. Worker mode is a queue: the second request
        waits its turn and says where it is in line."""
        _worker_mode(monkeypatch)

        async def busy():
            async with get_sessionmaker()() as db:
                await _job(db, status="running")
            return await _session()

        sid = asyncio.run(busy())
        from db.session import get_engine
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
        with _client() as c:
            r = c.post("/api/autolabel/start", json={"session_id": sid, "limit": 5})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["status"] == "pending" and body["executor"] == "worker"
            assert body["queue_position"] == 1
            j = c.get(f"/api/autolabel/{body['job_id']}").json()
            assert j["executor"] == "worker" and j["queue_position"] == 1

    def test_the_request_is_stored_on_the_row(self, monkeypatch):
        _worker_mode(monkeypatch)
        sid = asyncio.run(_session())
        from db.session import get_engine
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
        with _client() as c:
            jid = c.post("/api/autolabel/start",
                         json={"session_id": sid, "limit": 7, "only_unlabelled": True}).json()["job_id"]
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
        row = asyncio.run(_row(uuid.UUID(jid)))
        assert row.params == {"limit": 7, "only_unlabelled": True, "executor": "worker"}

    def test_in_process_mode_still_refuses_while_another_job_runs(self):
        async def busy():
            async with get_sessionmaker()() as db:
                await _job(db, status="running", executor="api")
            return await _session()

        sid = asyncio.run(busy())
        from db.session import get_engine
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
        with _client() as c:
            r = c.post("/api/autolabel/start", json={"session_id": sid})
            assert r.status_code == 409


class TestTheLockOutlivingItsSession:
    async def test_a_worker_whose_lock_session_ends_exits_to_take_it_again(self, monkeypatch):
        """A Postgres restart or a restore ends the session holding the lock while the process lives on. The
        worker used to keep claiming jobs with no lock held; it now exits so its restart policy brings it
        back holding the lock again."""
        from sqlalchemy import text

        import services.gpu_worker as gw
        from core.config import get_settings
        from db.session import get_engine

        monkeypatch.setenv("LBX_TRAINING__WORKER_POLL_S", "0.2")
        monkeypatch.setenv("LBX_TRAINING__ADVISORY_LOCK_KEY", "991771")   # this test's own key
        get_settings.cache_clear()
        task = asyncio.ensure_future(gw.worker_loop())

        async def lock_pid():
            async with get_engine().connect() as c:
                return (await c.execute(text(
                    "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted "
                    "AND ((classid::bigint << 32) | objid::bigint) = 991771"))).scalar()

        pid = None
        for _ in range(50):
            pid = await lock_pid()
            if pid:
                break
            await asyncio.sleep(0.1)
        assert pid, "the worker never took its lock"
        async with get_engine().connect() as c:
            await c.execute(text("SELECT pg_terminate_backend(:p)"), {"p": pid})
        assert await asyncio.wait_for(task, 10) == 1
