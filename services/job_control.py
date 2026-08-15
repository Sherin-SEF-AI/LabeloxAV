"""Cancelling a job, and the difference between asking and stopping.

There was exactly one cancel route among 610 endpoints (`services/api/routers/training.py`), so the only way
to clear an auto-label, import, export, relabel or map-fusion job was an UPDATE against the database. That
mattered more than it sounds: the autolabel router refuses to start while any row reads `running`, so one
unwanted job blocked everybody until an operator opened psql.

The mechanism is the progress write, not a new column. Every one of these jobs already writes progress as it
goes, and that write can carry the question with it: update the row only while it is not canceled, and read
the row count. Zero rows updated means somebody canceled it between the last checkpoint and this one. One
round trip, no extra column, no migration, and no window where a job reads its own status and then continues
past a cancel that landed a millisecond later.

The honest part is what cancel can promise. A job whose task is already gone, or that never started, is
stopped the moment the row changes: there is nothing left to stop. A job that is genuinely running stops at
its next checkpoint, which is a frame for auto-label and a file for import. So the API answers with which of
those happened rather than a bare 200, because "cancelled" and "asked to cancel" are different facts and an
operator watching a GPU cares which one they got.
"""

from __future__ import annotations

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("job_control")

CANCELED = "canceled"
# Work parked for a GPU pod that this box will never run. It is a status rather than a note on the side
# because three layers used to disagree about it: the row said `pending`, the start response said
# `queued-cloud`, and the jobs list translated `pending` back to `queued-cloud` at read time for display.
# Anything reading the database directly, including the server-sent-events waiting totals and the top-bar
# chip, saw parked work as merely queued and could not tell it from work a local runner was about to pick up.
QUEUED_CLOUD = "queued-cloud"
# States a cancel must refuse. Cancelling finished work would rewrite history, and cancelling a cancel is a
# no-op dressed up as an action.
TERMINAL = frozenset({"done", "error", CANCELED})


class JobCanceled(Exception):
    """Raised inside a running job when its row was canceled. Caught by the job's own guard, which records
    the outcome; it is not an error and must not be reported as one."""


async def note_progress(db: AsyncSession, model, job_id, **fields) -> bool:
    """Write a job's progress unless it has been canceled. False means stop.

    The conditional UPDATE is the whole point: a read-then-write would leave a window in which a cancel lands
    after the read and the job carries on for another checkpoint having been told it was still wanted.
    """
    res = await db.execute(
        update(model).where(model.job_id == job_id, model.status.notin_(list(TERMINAL))).values(**fields))
    await db.commit()
    return bool(res.rowcount)


async def raise_if_canceled(db: AsyncSession, model, job_id, **fields) -> None:
    """`note_progress`, for callers whose loop is easier to unwind with an exception than with a flag."""
    if not await note_progress(db, model, job_id, **fields):
        raise JobCanceled(str(job_id))


async def cancel_job(db: AsyncSession, model, job_id) -> dict:
    """Cancel a job. Reports whether it was stopped outright or merely asked to stop.

    `stopped` is true when nothing was running: the row was queued, or its process is already gone. It is
    false when a live task still has to reach its next checkpoint, which is a frame for auto-label and a
    file for import. Collapsing those two into one answer would tell an operator the GPU is free when it is
    not.
    """
    job = await db.get(model, job_id)
    if job is None:
        return {"job_id": str(job_id), "canceled": False, "detail": "not found"}
    if job.status in TERMINAL:
        return {"job_id": str(job_id), "canceled": False, "status": job.status,
                "detail": f"already {job.status}"}
    was_running = job.status == "running"
    job.status = CANCELED
    job.error = job.error or "canceled"
    await db.commit()
    log.info("job.canceled", job_id=str(job_id), table=model.__tablename__, was_running=was_running)
    return {"job_id": str(job_id), "canceled": True, "status": CANCELED, "stopped": not was_running,
            "detail": "stopped" if not was_running
                      else "asked to stop; it will end at its next checkpoint"}
