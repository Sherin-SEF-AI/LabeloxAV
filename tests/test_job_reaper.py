"""One orphaned row disabled auto-labeling for the whole deployment, permanently.

`services/api/routers/autolabel.py` refuses to start a job while any `AutolabelJob` has status `running`,
which is the right guard for a single-GPU box. Nothing reset a stale one: `reap_interrupted` sweeps
`AgentRun` and nothing else, and `services/api/routers/training.py` holds the only cancel route among 610.
So an API restart mid-auto-label left a row nobody owned, and every later attempt to auto-label anything, by
anyone, returned 409 until somebody ran an UPDATE by hand.

The 67 rows this corpus carried at `pending` are the sibling case: the process died between the INSERT and
the first progress write.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from db.models import AutolabelJob, ExportJob, ImportJob, RelabelJob, TrainingJob
from db.session import get_sessionmaker
from services.agent.resume import STALE_AFTER
from services.job_reaper import JOB_KINDS, is_stale, reap_stale_jobs, sweepable_statuses

pytestmark = pytest.mark.db

FRESH = datetime.now(UTC)
DEAD = datetime.now(UTC) - STALE_AFTER - timedelta(minutes=5)


class _Row:
    """The two timestamp fields the sweep reads, without a database round trip."""

    def __init__(self, updated_at=None, created_at=None):
        self.updated_at = updated_at
        self.created_at = created_at


class TestStaleness:
    def test_a_row_written_recently_is_alive(self):
        assert is_stale(_Row(updated_at=FRESH)) is False

    def test_a_row_untouched_past_the_window_is_dead(self):
        assert is_stale(_Row(updated_at=DEAD)) is True

    def test_a_row_exactly_inside_the_window_is_left_alone(self):
        # Killing a slow job and letting a second copy start is worse than one more stale row.
        just_inside = datetime.now(UTC) - STALE_AFTER + timedelta(seconds=30)
        assert is_stale(_Row(updated_at=just_inside)) is False

    def test_created_at_is_the_fallback_when_nothing_has_been_written(self):
        # A row that predates the mechanism must be judged rather than left running forever.
        assert is_stale(_Row(updated_at=None, created_at=DEAD)) is True
        assert is_stale(_Row(updated_at=None, created_at=FRESH)) is False

    def test_a_row_with_no_timestamps_at_all_is_dead(self):
        assert is_stale(_Row()) is True

    def test_a_naive_timestamp_does_not_raise(self):
        # Postgres hands these back tz-aware, but a fixture or an older row may not.
        assert is_stale(_Row(updated_at=DEAD.replace(tzinfo=None))) is True


class TestWhichStatusesAreSwept:
    def test_pending_is_swept_only_where_the_api_is_the_executor(self):
        # `pending` means two different things. For a job the API spawns it lasts milliseconds, so ten
        # minutes of it means the process died in that gap. For a worker-drained job it is a queue that may
        # last days, and reaping it would delete work somebody is waiting for.
        by_name = {k.name: k for k in JOB_KINDS}
        assert "pending" in sweepable_statuses(by_name["autolabel"])
        assert "pending" in sweepable_statuses(by_name["import"])
        assert "pending" in sweepable_statuses(by_name["export"])
        assert "pending" not in sweepable_statuses(by_name["relabel"])
        assert "pending" not in sweepable_statuses(by_name["map_fusion"])

    def test_running_is_swept_for_every_kind(self):
        assert all("running" in sweepable_statuses(k) for k in JOB_KINDS)

    def test_no_terminal_status_is_ever_swept(self):
        for k in JOB_KINDS:
            assert not ({"done", "error", "canceled"} & set(sweepable_statuses(k)))

    def test_training_is_not_in_the_sweep_at_all(self):
        # It has its own orphan reset in `services/training/worker.py`, which knows about the GPU lease.
        assert "training" not in {k.name for k in JOB_KINDS}


async def _autolabel(db, status: str, age: datetime) -> uuid.UUID:
    jid = uuid.uuid4()
    db.add(AutolabelJob(job_id=jid, session_id=uuid.uuid4(), status=status))
    await db.commit()
    await db.execute(AutolabelJob.__table__.update()
                     .where(AutolabelJob.job_id == jid).values(updated_at=age))
    await db.commit()
    return jid


class TestTheSweep:
    async def test_the_deadlock_clears(self):
        """The reported failure, end to end: a stale running row no longer blocks the next job."""
        async with get_sessionmaker()() as db:
            jid = await _autolabel(db, "running", DEAD)
            await reap_stale_jobs(db)
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.status == "error"
            assert "interrupted" in (row.error or "")

    async def test_a_live_job_is_never_touched(self):
        """A second replica running its own jobs must survive this process starting."""
        async with get_sessionmaker()() as db:
            jid = await _autolabel(db, "running", FRESH)
            await reap_stale_jobs(db)
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.status == "running"

    async def test_a_job_that_never_started_is_reaped(self):
        """The 67-row case: committed, then the process died before the first progress write."""
        async with get_sessionmaker()() as db:
            jid = await _autolabel(db, "pending", DEAD)
            await reap_stale_jobs(db)
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.status == "error"

    async def test_a_finished_job_keeps_its_result(self):
        async with get_sessionmaker()() as db:
            jid = await _autolabel(db, "done", DEAD)
            await reap_stale_jobs(db)
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.status == "done"

    async def test_an_existing_error_message_is_not_overwritten(self):
        """The original cause is more useful than the fact that the process later went away."""
        async with get_sessionmaker()() as db:
            jid = await _autolabel(db, "running", DEAD)
            await db.execute(AutolabelJob.__table__.update()
                             .where(AutolabelJob.job_id == jid).values(error="CUDA out of memory"))
            await db.commit()
            await reap_stale_jobs(db)
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.error == "CUDA out of memory"

    async def test_a_queued_worker_job_is_left_for_its_worker(self):
        """A relabel job parked for a worker that starts tomorrow is not a dead job."""
        jid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(RelabelJob(job_id=jid, status="pending", model_version="reaper-test"))
            await db.commit()
            await db.execute(RelabelJob.__table__.update()
                             .where(RelabelJob.job_id == jid).values(updated_at=DEAD))
            await db.commit()
            await reap_stale_jobs(db)
            row = await db.get(RelabelJob, jid)
            await db.refresh(row)
            assert row.status == "pending"

    async def test_training_is_left_to_its_own_worker(self):
        """`services/training/worker.py` resets its orphans with knowledge of the GPU lease."""
        jid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(TrainingJob(job_id=jid, purpose="reaper-test", status="pending"))
            await db.commit()
            await db.execute(TrainingJob.__table__.update()
                             .where(TrainingJob.job_id == jid).values(updated_at=DEAD))
            await db.commit()
            await reap_stale_jobs(db)
            row = await db.get(TrainingJob, jid)
            await db.refresh(row)
            assert row.status == "pending"

    async def test_a_dead_export_becomes_resumable_rather_than_lost(self):
        """`services/export/resumable.py` calls a job resumable when it is `error` with a checkpoint, so
        reaping puts a half-finished export back on the resume list instead of stranding it."""
        jid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(ExportJob(job_id=jid, status="running", name=f"reap-{jid.hex[:8]}",
                             spec={"format": "coco"}, checkpoint={"chunks_done": 3}))
            await db.commit()
            await db.execute(ExportJob.__table__.update()
                             .where(ExportJob.job_id == jid).values(updated_at=DEAD))
            await db.commit()
            await reap_stale_jobs(db)
            row = await db.get(ExportJob, jid)
            await db.refresh(row)
            assert row.status == "error"
            assert row.checkpoint == {"chunks_done": 3}

    async def test_the_sweep_reports_what_it_did(self):
        """Silent recovery is how a recurring crash stays invisible."""
        async with get_sessionmaker()() as db:
            jid = await _autolabel(db, "running", DEAD)
            reaped = await reap_stale_jobs(db)
        assert str(jid) in reaped.get("autolabel", [])

    async def test_an_import_left_running_is_reaped_too(self):
        jid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(ImportJob(job_id=jid, status="running", format="video",
                             source_uri="s3://x/y.mp4", target_vehicle="REAP-01"))
            await db.commit()
            await db.execute(ImportJob.__table__.update()
                             .where(ImportJob.job_id == jid).values(updated_at=DEAD))
            await db.commit()
            await reap_stale_jobs(db)
            row = await db.get(ImportJob, jid)
            await db.refresh(row)
            assert row.status == "error"
