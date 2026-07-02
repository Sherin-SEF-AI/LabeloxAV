"""Corpus-wide error daemon: keep the fix queue fresh by sweeping the whole corpus with every error
detector (confident-learning, embedding-outlier, track/cross-cam consistency, and the consistency critic),
so likely-wrong labels surface proactively instead of only when someone opens a session. Runs in the
background, session by session (naturally bounded and resumable), tracked on a flywheel-style AgentRun; it
yields to a running training job. The ErrorCandidate queue and its confirm/dismiss workflow already exist;
this just drives detection across everything on a schedule.
"""

from __future__ import annotations

import uuid

from sqlalchemy import distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object, TrainingJob

log = get_logger("agent.error_daemon")


async def _training_running(db: AsyncSession) -> bool:
    return (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first() is not None


async def _sessions_with_machine_objects(db: AsyncSession, limit: int) -> list[uuid.UUID]:
    rows = await db.execute(
        select(distinct(Frame.session_id)).join(Object, Object.frame_id == Frame.frame_id)
        .where(Object.source != "human").limit(limit))
    return list(rows.scalars().all())


async def run_error_sweep(run_id: uuid.UUID, *, max_sessions: int = 10, kinds: list[str] | None = None) -> None:
    """Background: run every detector across up to max_sessions, updating the fix queue and the run."""
    from db.session import get_sessionmaker
    from services.errordetect.queue import run_detection

    maker = get_sessionmaker()
    async with maker() as db:
        if await _training_running(db):
            run = await db.get(AgentRun, run_id)
            if run is not None:
                run.status = "committed"
                run.counts = {"skipped": "training job holds the GPU"}
                await db.commit()
            return
        sessions = await _sessions_with_machine_objects(db, max_sessions)

    totals: dict = {"sessions": 0, "persisted": 0, "by_kind": {}}
    try:
        for sid in sessions:
            async with maker() as db:
                res = await run_detection(db, str(sid), kinds)
            totals["sessions"] += 1
            totals["persisted"] += int(res.get("persisted", 0))
            for k, n in (res.get("by_kind") or {}).items():
                totals["by_kind"][k] = totals["by_kind"].get(k, 0) + int(n)
            async with maker() as db:
                run = await db.get(AgentRun, run_id)
                if run is not None:
                    run.counts = dict(totals)
                    await db.commit()
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run is not None:
                run.status = "committed"
                run.counts = dict(totals)
                await db.commit()
        log.info("agent.error_sweep.done", run_id=str(run_id), **{k: totals[k] for k in ("sessions", "persisted")})
    except Exception as exc:  # noqa: BLE001
        log.error("agent.error_sweep.failed", run_id=str(run_id), error=str(exc))
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run is not None:
                run.status = "error"
                run.error = str(exc)
                await db.commit()
