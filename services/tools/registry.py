"""The annotation tools' job queue: register a kind, enqueue it, and let the right process run it.

A tool registers an async function under a kind name:

    @register("sam2_video", gpu=True, autolabel=True, label="Video mask tracking")
    async def run(ctx: JobContext) -> dict:
        await ctx.progress(0.5, "tracking")
        ...
        return {"frames": 20}

`enqueue` writes a `tool_job` row. GPU kinds wait for the GPU worker (services/gpu_worker.py claims them after
auto-label and training, one at a time); CPU kinds are run by the process that enqueued them, in the
background. Either way the row carries progress, stage, result and error, shows on /jobs and the job stream,
and can be cancelled through the shared cancel route (routers/jobs.py CANCELABLE).

`autolabel=True` marks a kind that runs models over footage. Those are not started during the office-hours
quiet window (`phase4.govern.quiet_hours_ist`, default 10:00-18:00 IST): they stay pending and start after it.
"""

from __future__ import annotations

import asyncio
import importlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ToolJob
from db.session import get_sessionmaker

log = get_logger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
TERMINAL = frozenset({"done", "error", "canceled"})

# The modules that register kinds. Each tool area owns one; importing it is what registers its kinds.
KIND_MODULES = (
    "services.tools.kinds_video",
    "services.tools.kinds_scene3d",
    "services.tools.kinds_agents",
    "services.tools.kinds_quality",
    "services.tools.kinds_driving",
    "services.tools.kinds_sensors",
)


class JobCanceled(Exception):
    """Raised inside a running job when somebody cancelled it; the runner records it as canceled."""


@dataclass
class ToolKind:
    kind: str
    fn: Callable[[JobContext], Awaitable[dict | None]]
    gpu: bool
    autolabel: bool
    label: str


REGISTRY: dict[str, ToolKind] = {}


def register(kind: str, *, gpu: bool = False, autolabel: bool = False, label: str | None = None):
    """Decorator registering `fn(ctx) -> dict | None` as the runner for `kind`."""
    def deco(fn):
        if kind in REGISTRY and REGISTRY[kind].fn is not fn:
            raise ValueError(f"tool job kind {kind!r} registered twice")
        REGISTRY[kind] = ToolKind(kind=kind, fn=fn, gpu=gpu, autolabel=autolabel,
                                  label=label or kind.replace("_", " ").capitalize())
        return fn
    return deco


_loaded = False


def load_kinds() -> dict[str, ToolKind]:
    """Import every kind module once, so the registry is complete in whichever process asks."""
    global _loaded
    if not _loaded:
        for mod in KIND_MODULES:
            importlib.import_module(mod)
        _loaded = True
    return REGISTRY


def _parse_window(spec: str) -> tuple[time, time] | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    a, b = spec.split("-", 1)
    ha, ma = (int(x) for x in a.strip().split(":"))
    hb, mb = (int(x) for x in b.strip().split(":"))
    return time(ha, ma), time(hb, mb)


def in_quiet_hours(now: datetime | None = None, spec: str | None = None) -> bool:
    """Whether `now` falls in the office-hours window during which auto-labelling must not start."""
    window = _parse_window(get_settings().phase4.govern.quiet_hours_ist if spec is None else spec)
    if window is None:
        return False
    t = (now or datetime.now(UTC)).astimezone(IST).time()
    start, end = window
    return start <= t < end if start <= end else (t >= start or t < end)


class JobContext:
    """What a running job sees: its id, its parameters, and a way to report progress and notice a cancel."""

    def __init__(self, job_id: uuid.UUID, kind: str, params: dict, session_id: uuid.UUID | None,
                 created_by: uuid.UUID | None):
        self.job_id = job_id
        self.kind = kind
        self.params = params
        self.session_id = session_id
        self.created_by = created_by

    async def progress(self, frac: float, stage: str | None = None) -> None:
        """Record progress (0..1) and an optional stage; raises JobCanceled if the job was cancelled."""
        async with get_sessionmaker()() as db:
            values: dict = {"progress": max(0.0, min(1.0, float(frac))), "heartbeat_at": datetime.now(UTC)}
            if stage is not None:
                values["stage"] = stage[:64]
            await db.execute(update(ToolJob).where(ToolJob.job_id == self.job_id, ToolJob.status == "running")
                             .values(**values))
            status = (await db.execute(select(ToolJob.status).where(ToolJob.job_id == self.job_id))).scalar()
            await db.commit()
        if status == "canceled":
            raise JobCanceled()

    async def check_canceled(self) -> None:
        async with get_sessionmaker()() as db:
            status = (await db.execute(select(ToolJob.status).where(ToolJob.job_id == self.job_id))).scalar()
        if status == "canceled":
            raise JobCanceled()


# Background tasks for CPU kinds, kept referenced so they are not garbage-collected mid-run.
_tasks: set[asyncio.Task] = set()


async def enqueue(db: AsyncSession, kind: str, params: dict | None = None, *, session_id: uuid.UUID | None = None,
                  user_id: uuid.UUID | None = None, start: bool = True) -> ToolJob:
    """Queue a job. Commits. A CPU kind starts at once in this process unless `start=False` (tests)."""
    tk = load_kinds().get(kind)
    if tk is None:
        raise KeyError(f"unknown tool job kind {kind!r}")
    job = ToolJob(kind=kind, gpu=tk.gpu, params=params or {}, session_id=session_id, created_by=user_id)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    if start and not tk.gpu:
        task = asyncio.get_running_loop().create_task(run_job(job.job_id))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    return job


async def claim_gpu_job(db: AsyncSession, now: datetime | None = None) -> uuid.UUID | None:
    """Claim the oldest pending GPU tool job, skipping auto-labelling kinds during quiet hours."""
    load_kinds()
    quiet = in_quiet_hours(now)
    # The claim is its own transaction (FOR UPDATE SKIP LOCKED, committed at once); finish whatever the caller's
    # session had open first, so a session that has just read something can still claim.
    if db.in_transaction():
        await db.commit()
    async with db.begin():
        q = (select(ToolJob).where(ToolJob.status == "pending", ToolJob.gpu.is_(True))
             .order_by(ToolJob.created_at).with_for_update(skip_locked=True))
        for row in (await db.execute(q.limit(50))).scalars():
            tk = REGISTRY.get(row.kind)
            if tk is None:
                row.status, row.error = "error", f"no runner registered for {row.kind!r}"
                continue
            if quiet and tk.autolabel:
                continue
            row.status = "running"
            row.started_at = datetime.now(UTC)
            row.attempts = (row.attempts or 0) + 1
            return row.job_id
    return None


async def run_job(job_id: uuid.UUID) -> str:
    """Run one claimed or freshly enqueued job to a terminal state. Returns the final status."""
    load_kinds()
    async with get_sessionmaker()() as db:
        job = await db.get(ToolJob, job_id)
        if job is None:
            return "missing"
        if job.status == "pending":
            job.status = "running"
            job.started_at = datetime.now(UTC)
            job.attempts = (job.attempts or 0) + 1
            await db.commit()
        elif job.status != "running":
            return job.status
        ctx = JobContext(job.job_id, job.kind, dict(job.params or {}), job.session_id, job.created_by)
    tk = REGISTRY.get(ctx.kind)
    status, result, error = "done", None, None
    try:
        if tk is None:
            raise RuntimeError(f"no runner registered for {ctx.kind!r}")
        result = await tk.fn(ctx)
    except JobCanceled:
        status = "canceled"
    except Exception as exc:  # noqa: BLE001  a job's failure is recorded on its row, not raised into the worker
        log.exception("tool_job.failed", kind=ctx.kind, job_id=str(job_id))
        status, error = "error", f"{type(exc).__name__}: {exc}"[:2000]
    async with get_sessionmaker()() as db:
        values: dict = {"finished_at": datetime.now(UTC), "heartbeat_at": datetime.now(UTC)}
        if status == "done":
            values.update(status="done", progress=1.0, result=result or {})
        elif status == "error":
            values.update(status="error", error=error)
        # A cancelled row is already `canceled`; only its finish time is added.
        await db.execute(update(ToolJob).where(ToolJob.job_id == job_id, ToolJob.status.notin_(["canceled"]))
                         .values(**values))
        if status == "canceled":
            await db.execute(update(ToolJob).where(ToolJob.job_id == job_id)
                             .values(finished_at=values["finished_at"]))
        await db.commit()
    log.info("tool_job.finished", kind=ctx.kind, job_id=str(job_id), status=status)
    return status


async def requeue_orphans(db: AsyncSession, *, gpu: bool, max_attempts: int = 3) -> int:
    """Put jobs a crash left `running` back in the queue (or fail them after `max_attempts`).

    Only safe while holding the lock that makes every `running` row of that class an orphan: the GPU worker's
    lifetime lock for GPU kinds, API startup for CPU kinds.
    """
    n = 0
    rows = (await db.execute(select(ToolJob).where(ToolJob.status == "running", ToolJob.gpu.is_(gpu)))).scalars()
    for row in rows:
        if (row.attempts or 0) >= max_attempts:
            row.status, row.error = "error", f"stopped {row.attempts} times before finishing; not retried"
        else:
            row.status = "pending"
        n += 1
    await db.commit()
    return n
