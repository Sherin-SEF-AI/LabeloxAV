"""Job rows whose process died, and why one of them can brick the whole deployment.

`services/agent/resume.py` already solves this for `AgentRun`, and says why: background work runs as a task
inside the API process, so when the process goes away the task goes with it and the row is never written
again. What it does not do is cover the five job tables that are not `AgentRun`, and for one of them the
consequence is not a stale row on a dashboard.

`services/api/routers/autolabel.py` refuses to start a new job while ANY `AutolabelJob` has status
`running`, which is the correct guard for a single-GPU box. Nothing resets a stale one. So a single API
restart in the middle of an auto-label leaves a row that no longer belongs to any process, and every
subsequent attempt to auto-label anything, by anyone, returns 409 forever. There is no button, route or
sweep that clears it; the only recovery is an UPDATE against the database.

The heartbeat these tables already have is `updated_at`, which carries `onupdate=func.now()`, so every
progress write refreshes it. A `running` row that has not been written for longer than the staleness window
is a row whose writer is gone, and a sweep can conclude that mechanically instead of a person guessing from
a timestamp.

`pending` is treated differently per job kind, because the word means two different things. For a job the
API spawns in-process (auto-label, import, export) `pending` lasts milliseconds: the row is committed and
the task starts. Ten minutes of it means the process died in that gap, which is exactly how a corpus ends
up holding rows that were never going to run. For a job drained by a separate worker (training, relabel,
map fusion) `pending` is a legitimate queue state that may last days, and reaping it would delete work
somebody is waiting for. So `pending` is only swept where the API itself was the executor.

Reaped jobs land in `error` rather than a new status word, both because it is already the vocabulary these
tables use and because it is true: the job did fail, it failed by dying. That also has a useful side effect
for exports, where `services/export/resumable.py` treats `status == "error"` with a non-empty checkpoint as
resumable, so reaping a half-finished export puts it back on the resume list instead of losing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AutolabelJob, ExportJob, ImportJob, MapFusionJob, RelabelJob

# Borrowed rather than redefined, and for the reason stated there: generously longer than any single unit of
# work these jobs do between writes, because declaring a slow job dead and letting a second copy start is
# worse than leaving a stale row for another few minutes. Two windows that could drift apart would be two
# different answers to one question.
from services.agent.resume import STALE_AFTER

log = get_logger("job_reaper")

REAPED_STATUS = "error"
REAPED_ERROR = "interrupted: the process running this job stopped before it finished"


@dataclass(frozen=True)
class JobKind:
    """One job table, and whether the API process is the thing that runs it.

    `in_process` is the whole reason this is a table rather than a loop: it decides whether a long-lived
    `pending` row is a dead task or a queued one, and getting that backwards either leaves the deadlock in
    place or cancels work a worker was about to pick up.
    """

    name: str
    model: type
    in_process: bool


JOB_KINDS: tuple[JobKind, ...] = (
    # Spawned by the API itself, so `pending` is a millisecond window.
    JobKind("autolabel", AutolabelJob, in_process=True),
    JobKind("import", ImportJob, in_process=True),
    JobKind("export", ExportJob, in_process=True),
    # Drained by a separate worker or by a cloud pod, so `pending` is a queue and must be left alone.
    JobKind("relabel", RelabelJob, in_process=False),
    JobKind("map_fusion", MapFusionJob, in_process=False),
)


def _now() -> datetime:
    return datetime.now(UTC)


def sweepable_statuses(kind: JobKind) -> tuple[str, ...]:
    """Which statuses this kind's sweep may touch. Never a terminal one, and never a parked one."""
    return ("running", "pending") if kind.in_process else ("running",)


def is_stale(row, *, now: datetime | None = None) -> bool:
    """Whether a job row's writer is gone.

    Falls back to `created_at` when `updated_at` is somehow absent, so a row that predates the mechanism is
    judged rather than left running forever, which is the state this module exists to clear.
    """
    last = getattr(row, "updated_at", None) or getattr(row, "created_at", None)
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return (now or _now()) - last > STALE_AFTER


async def reap_stale_jobs(db: AsyncSession, *, now: datetime | None = None) -> dict[str, list[str]]:
    """Fail every job row whose process is gone. Returns the reaped ids per kind.

    Run at API startup, which is the one moment the previous process's tasks are known to be dead, and
    then on the governance daemon's cadence - a long-lived API whose in-process task dies without writing a
    terminal status otherwise leaves a row reading `running` forever, and autolabel refuses to start while
    one does. The staleness window still applies, because a second API replica may be running jobs of its
    own and a sweep that ignored the window would kill them.
    """
    at = now or _now()
    reaped: dict[str, list[str]] = {}
    for kind in JOB_KINDS:
        rows = (await db.execute(
            select(kind.model).where(kind.model.status.in_(sweepable_statuses(kind))))).scalars().all()
        dead = [r for r in rows if is_stale(r, now=at)]
        if not dead:
            continue
        for r in dead:
            r.status = REAPED_STATUS
            # The error field is what a reader sees first, and "no error" beside a job that stopped mid-way
            # is the misleading answer.
            r.error = r.error or REAPED_ERROR
        reaped[kind.name] = [str(r.job_id) for r in dead]
    if reaped:
        await db.commit()
        log.warning("job_reaper.reaped", **{k: len(v) for k, v in reaped.items()})
    return reaped
