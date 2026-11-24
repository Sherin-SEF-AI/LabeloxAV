"""Cancelling a training job answers with the job, not a 500.

The handler committed and then built its response from the row it had just committed. The commit expires
the row and `updated_at` is assigned by the database, so reading it was a lazy load outside a greenlet:
every cancel raised MissingGreenlet after having, in fact, cancelled. The person saw an error and the
job was gone.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import TrainingJob
from db.session import get_sessionmaker
from services.api.routers.training import cancel

pytestmark = pytest.mark.db


async def _job(status: str) -> uuid.UUID:
    jid = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(TrainingJob(job_id=jid, status=status, purpose="cancel-test", task_type="detection",
                           compute_target="local", config={}))
        await db.commit()
    return jid


@pytest.mark.asyncio
async def test_a_queued_job_is_cancelled_and_returned():
    jid = await _job("pending")
    out = await cancel(jid)
    assert out["job_id"] == str(jid) and out["status"] == "canceled"
    assert out["updated_at"], "the response must carry the row as it is after the write"


@pytest.mark.asyncio
async def test_a_running_job_is_asked_to_stop_and_returned():
    jid = await _job("running")
    out = await cancel(jid)
    assert out["status"] == "running"
    async with get_sessionmaker()() as db:
        assert (await db.get(TrainingJob, jid)).cancel_requested is True
        (await db.get(TrainingJob, jid)).status = "canceled"
        await db.commit()


@pytest.mark.asyncio
async def test_an_unknown_job_is_a_404():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await cancel(uuid.uuid4())
    assert exc.value.status_code == 404
