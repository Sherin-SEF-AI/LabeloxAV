"""The top bar said "1 queued" while sixty-eight jobs sat untouched.

`_job_snapshot` sends a recent tail per job kind: twenty training rows, ten of everything else. That is right
for showing progress, because nobody watching a bar move needs the whole history. It is wrong for answering
"why is nothing happening", which is a question about the total.

This deployment holds 67 autolabel jobs parked for a cloud A100 since late June. Every one of them is older
than the ten most recent, so a client counting what it received saw only the single pending training job.
Under-reporting is the worst of the options here: nothing at all reads as "no information", while "1 queued"
reads as a healthy system with a stray job in it.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import AutolabelJob, ImportJob
from db.session import get_sessionmaker
from services.api.routers.events import _job_snapshot

pytestmark = pytest.mark.db


async def _seed(db, model, status: str, n: int, **extra) -> None:
    for _ in range(n):
        db.add(model(job_id=uuid.uuid4(), session_id=uuid.uuid4(), status=status, **extra))
    await db.commit()


async def test_the_queued_total_counts_past_the_window():
    """More parked jobs than the list carries, which is the reported case."""
    async with get_sessionmaker()() as db:
        before = (await _job_snapshot(db))["waiting"]["autolabel"]
        await _seed(db, AutolabelJob, "queued-cloud", 25)
        snap = await _job_snapshot(db)

    assert len(snap["autolabel"]) <= 10, "the list is still a tail, which is the whole reason for the count"
    assert snap["waiting"]["autolabel"] >= before + 25


async def test_every_kind_of_holding_status_counts_as_waiting():
    """`pending` is work nobody picked up and `queued-cloud` is work parked on purpose. Neither is progress,
    and a person asking why the system is idle needs both."""
    async with get_sessionmaker()() as db:
        before = (await _job_snapshot(db))["waiting"]["import"]
        await _seed(db, ImportJob, "pending", 3, format="video", source_uri="s3://x/y.mp4", target_vehicle="TEST-SSE")
        await _seed(db, ImportJob, "queued-cloud", 2, format="video", source_uri="s3://x/y.mp4", target_vehicle="TEST-SSE")
        snap = await _job_snapshot(db)

    assert snap["waiting"]["import"] >= before + 5


async def test_finished_and_running_work_is_not_counted_as_waiting():
    """Otherwise the number grows forever and stops meaning anything."""
    async with get_sessionmaker()() as db:
        before = (await _job_snapshot(db))["waiting"]["import"]
        await _seed(db, ImportJob, "done", 4, format="video", source_uri="s3://x/y.mp4", target_vehicle="TEST-SSE")
        await _seed(db, ImportJob, "running", 2, format="video", source_uri="s3://x/y.mp4", target_vehicle="TEST-SSE")
        after = (await _job_snapshot(db))["waiting"]["import"]

    assert after == before


async def test_every_job_kind_reports_a_number():
    """A missing key would read as zero on the client, which is the same failure in a different place."""
    async with get_sessionmaker()() as db:
        waiting = (await _job_snapshot(db))["waiting"]

    assert set(waiting) == {"training", "import", "export", "autolabel"}
    assert all(isinstance(v, int) for v in waiting.values())
