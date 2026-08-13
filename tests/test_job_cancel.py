"""There was one cancel route among 610 endpoints, and it was training's.

Everything else needed an UPDATE against the database to stop. That is worse than an inconvenience for
auto-label, because `services/api/routers/autolabel.py` refuses to start while any row reads `running`: one
unwanted job blocked every user of the deployment until an operator opened psql.

The other half of the fix is that a cancel has to actually stop the work. A cancel that only wrote a column
would leave a labelling run on the GPU for another twenty minutes while the dashboard claimed it was
cancelled, which is the same class of lie as a progress bar frozen at five percent.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import AutolabelJob, ImportJob
from db.session import get_sessionmaker
from services.job_control import CANCELED, JobCanceled, cancel_job, note_progress, raise_if_canceled

pytestmark = pytest.mark.db


async def _job(db, status: str = "running") -> uuid.UUID:
    jid = uuid.uuid4()
    db.add(AutolabelJob(job_id=jid, session_id=uuid.uuid4(), status=status))
    await db.commit()
    return jid


class TestCancel:
    async def test_a_running_job_is_marked_canceled(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db)
            out = await cancel_job(db, AutolabelJob, jid)
            assert out["canceled"] is True
            assert (await db.get(AutolabelJob, jid)).status == CANCELED

    async def test_a_live_job_is_asked_to_stop_rather_than_reported_stopped(self):
        """An operator waiting for the GPU needs to know the difference."""
        async with get_sessionmaker()() as db:
            jid = await _job(db, "running")
            out = await cancel_job(db, AutolabelJob, jid)
            assert out["stopped"] is False
            assert "next checkpoint" in out["detail"]

    async def test_a_queued_job_is_stopped_outright(self):
        """Nothing is running, so there is nothing left to stop."""
        async with get_sessionmaker()() as db:
            jid = await _job(db, "pending")
            out = await cancel_job(db, AutolabelJob, jid)
            assert out["stopped"] is True

    async def test_a_finished_job_is_refused_rather_than_rewritten(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db, "done")
            out = await cancel_job(db, AutolabelJob, jid)
            assert out["canceled"] is False
            assert (await db.get(AutolabelJob, jid)).status == "done"

    async def test_cancelling_twice_is_not_an_error_and_not_a_second_cancel(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db)
            await cancel_job(db, AutolabelJob, jid)
            assert (await cancel_job(db, AutolabelJob, jid))["canceled"] is False

    async def test_an_unknown_job_is_reported_missing_not_crashed(self):
        async with get_sessionmaker()() as db:
            out = await cancel_job(db, AutolabelJob, uuid.uuid4())
            assert out["canceled"] is False and out["detail"] == "not found"

    async def test_an_existing_error_survives_the_cancel(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db)
            await db.execute(AutolabelJob.__table__.update()
                             .where(AutolabelJob.job_id == jid).values(error="CUDA out of memory"))
            await db.commit()
            await cancel_job(db, AutolabelJob, jid)
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.error == "CUDA out of memory"


class TestTheRunningJobNotices:
    async def test_progress_writes_while_the_job_is_wanted(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db)
            assert await note_progress(db, AutolabelJob, jid, progress=0.4) is True
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.progress == pytest.approx(0.4)

    async def test_progress_stops_writing_once_it_is_cancelled(self):
        """This is the signal the loop reads. Without it a cancelled run keeps labelling."""
        async with get_sessionmaker()() as db:
            jid = await _job(db)
            await cancel_job(db, AutolabelJob, jid)
            assert await note_progress(db, AutolabelJob, jid, progress=0.9) is False

    async def test_a_cancelled_job_keeps_the_progress_it_had_reached(self):
        """The conditional update must not write, not merely be ignored: the number is a record of how far
        the run actually got before somebody stopped it."""
        async with get_sessionmaker()() as db:
            jid = await _job(db)
            await note_progress(db, AutolabelJob, jid, progress=0.4)
            await cancel_job(db, AutolabelJob, jid)
            await note_progress(db, AutolabelJob, jid, progress=0.9)
            row = await db.get(AutolabelJob, jid)
            await db.refresh(row)
            assert row.progress == pytest.approx(0.4)

    async def test_the_loop_unwinds_by_exception(self):
        async with get_sessionmaker()() as db:
            jid = await _job(db)
            await cancel_job(db, AutolabelJob, jid)
            with pytest.raises(JobCanceled):
                await raise_if_canceled(db, AutolabelJob, jid, progress=0.9)

    async def test_a_finished_job_also_stops_the_loop(self):
        """Terminal is terminal. A run whose row was completed by something else must not keep writing."""
        async with get_sessionmaker()() as db:
            jid = await _job(db, "done")
            assert await note_progress(db, AutolabelJob, jid, progress=0.9) is False

    async def test_an_import_reads_the_same_signal(self):
        """`services/imports/run.py` bumps every hundred frames; that write is the cancel check."""
        jid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(ImportJob(job_id=jid, status="running", format="video",
                             source_uri="s3://x/y.mp4", target_vehicle="CANCEL-01"))
            await db.commit()
            await cancel_job(db, ImportJob, jid)
            assert await note_progress(db, ImportJob, jid, progress=0.5) is False
