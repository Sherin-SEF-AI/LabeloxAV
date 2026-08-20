"""Interrupted jobs: noticing them, and picking them up where they stopped.

Background work runs as `asyncio.create_task` inside the API process. That is a reasonable shape for this
system, and it has one consequence nobody had handled: when the process goes away, the task goes with it and
the row is never written again. The run stays `running` forever, indistinguishable from live work.

The live corpus was carrying five of these when this module was written, two of them stranded by an API
restart an hour earlier and one marked running for 863 hours. The cost is not only the stale row. A stuck
`running` job also satisfies the guards that refuse to start a second job while one holds the GPU, so one
dead run can quietly block the loop it belonged to.

A heartbeat is what makes the difference visible. A job writes one as it goes; a `running` row whose
heartbeat has gone stale is a job whose process died, and a sweep at startup can conclude that mechanically
rather than a person guessing from a timestamp.

Resuming is then a separate question, and the honest answer is per job: only a job that recorded what it had
already finished can skip it. `progress` is that cursor and it is opaque here on purpose. This module owns
noticing and bookkeeping; each job owns what "where it left off" means for its own unit of work.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun

log = get_logger("agent.resume")

RUNNING = "running"
INTERRUPTED = "interrupted"
TERMINAL = frozenset({"committed", "reverted", "error"})

# How quiet a running job has to be before it counts as dead. Generously longer than any single unit of work
# these jobs do between heartbeats (a session sweep, a frame batch), because declaring a slow job dead and
# letting a second copy start is worse than leaving a stale row for another few minutes.
STALE_AFTER = timedelta(minutes=10)


def _now() -> datetime:
    return datetime.now(UTC)


async def beat(db: AsyncSession, run_id: uuid.UUID, *, progress: dict | None = None,
               counts: dict | None = None) -> None:
    """Record that a run is alive, and optionally how far it has got.

    Deliberately tolerant of a missing run: a job whose row was deleted underneath it should finish its
    current unit of work and stop, not raise inside a background task where nothing is waiting to catch it.
    """
    run = await db.get(AgentRun, run_id)
    if run is None:
        return
    run.heartbeat_at = _now()
    if run.status not in TERMINAL:
        run.status = RUNNING
    if progress is not None:
        run.progress = dict(progress)
    if counts is not None:
        run.counts = dict(counts)
    await db.commit()


async def is_stale(run: AgentRun, *, now: datetime | None = None) -> bool:
    """Whether a run should be treated as dead.

    A null heartbeat falls back to `created_at`, so runs that predate the mechanism are judged rather than
    left running forever, which is the state this module exists to clear.
    """
    if run.status in TERMINAL or run.status == INTERRUPTED:
        return False
    last = run.heartbeat_at or run.created_at
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return (now or _now()) - last > STALE_AFTER


async def reap_interrupted(db: AsyncSession) -> list[dict]:
    """Mark every stale non-terminal run as interrupted. Returns what was reaped.

    Run at API startup, which is the moment the previous process's tasks are known to be gone. It cannot
    reap a job belonging to another live process, because it only touches rows whose heartbeat has already
    aged past the staleness window.
    """
    candidates = (await db.execute(
        select(AgentRun).where(AgentRun.status.notin_(list(TERMINAL) + [INTERRUPTED])))).scalars().all()
    now = _now()
    reaped: list[dict] = []
    for run in candidates:
        if not await is_stale(run, now=now):
            continue
        run.status = INTERRUPTED
        # The error field is what a reader sees first, and "no error" beside a job that stopped mid-way is
        # the misleading answer. It did fail; it failed by dying.
        run.error = run.error or "interrupted: the process running this job stopped before it finished"
        reaped.append({"run_id": str(run.run_id), "kind": run.kind,
                       "progress": dict(run.progress or {}), "counts": dict(run.counts or {})})
    if reaped:
        await db.commit()
        log.warning("agent.reaped_interrupted", n=len(reaped),
                    kinds=sorted({r["kind"] for r in reaped}))
    return reaped


async def list_interrupted(db: AsyncSession, limit: int = 50) -> list[dict]:
    """Interrupted runs, newest first, with enough context for someone to decide whether to resume."""
    rows = (await db.execute(
        select(AgentRun).where(AgentRun.status == INTERRUPTED)
        .order_by(AgentRun.created_at.desc()).limit(limit))).scalars().all()
    return [{
        "run_id": str(r.run_id), "kind": r.kind, "scope": dict(r.scope or {}),
        "progress": dict(r.progress or {}), "counts": dict(r.counts or {}),
        # Fraction of the work its cursor says was finished, for a progress bar that reflects a measurement
        # rather than an animation. None when the job never recorded a total.
        "fraction": fraction_done(dict(r.progress or {})),
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "heartbeat_at": r.heartbeat_at.isoformat() if r.heartbeat_at else None,
        "resumable": bool(r.progress),
        "error": r.error,
    } for r in rows]


async def claim_for_resume(db: AsyncSession, run_id: uuid.UUID) -> dict | None:
    """Take an interrupted run back to running so its job can continue, returning its cursor.

    Refuses anything not currently interrupted. Resuming a run that is genuinely still going would put two
    workers on the same cursor, and resuming a committed one would re-apply finished work; both are worse
    than telling the caller no.
    """
    run = await db.get(AgentRun, run_id)
    if run is None:
        return None
    if run.status != INTERRUPTED:
        return {"error": f"run is {run.status}, not interrupted", "run_id": str(run_id),
                "status": run.status}
    run.status = RUNNING
    run.heartbeat_at = _now()
    run.error = None
    await db.commit()
    return {"run_id": str(run_id), "kind": run.kind, "scope": dict(run.scope or {}),
            "progress": dict(run.progress or {}), "counts": dict(run.counts or {}), "resumed": True}


async def release_claim(db: AsyncSession, run_id: uuid.UUID, reason: str) -> bool:
    """Put a claimed run back to interrupted, for a caller that claimed it and then could not proceed.

    `claim_for_resume` flips the row to running before the job is launched, so a caller that discovers
    afterwards that it cannot launch one must undo that. Leaving it running strands the run in exactly the
    state this whole feature exists to clear: a row that looks like work in flight with no process behind
    it, waiting for the reaper to notice it hours later.

    The cursor and counts are untouched, so the run remains resumable once the reason is fixed.
    """
    run = await db.get(AgentRun, run_id)
    if run is None or run.status != RUNNING:
        return False
    run.status = INTERRUPTED
    run.error = f"interrupted: {reason}"
    run.heartbeat_at = _now()
    await db.commit()
    log.warning("agent.resume_released", run_id=str(run_id), kind=run.kind, reason=reason)
    return True


def fraction_done(progress: dict) -> float | None:
    """How far a cursor says a run got, in [0, 1], or None when it cannot say.

    None rather than 0.0 for an absent total: a bar sitting at zero claims no work was done, which is a
    different statement from not knowing.
    """
    total = progress.get("total")
    if not isinstance(total, int) or total <= 0:
        return None
    return round(min(1.0, len(done_set(progress)) / total), 4)


def done_set(progress: dict, key: str = "done") -> set[str]:
    """The identifiers a job has already finished, from its cursor.

    A helper rather than an inline expression because every resumable job needs the same thing and the empty
    cases (no cursor, a cursor from before this job recorded one) must all read as "nothing done" instead of
    raising inside a background task.
    """
    raw = (progress or {}).get(key) or []
    return {str(x) for x in raw} if isinstance(raw, list) else set()
