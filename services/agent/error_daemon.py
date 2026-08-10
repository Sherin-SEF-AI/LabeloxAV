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
from db.models import AgentRun, Frame, Object
from services.training.gpu_lease import training_holds_gpu

log = get_logger("agent.error_daemon")


async def _training_running(db: AsyncSession) -> bool:
    return await training_holds_gpu(db)


async def _sessions_with_machine_objects(db: AsyncSession, limit: int) -> list[uuid.UUID]:
    rows = await db.execute(
        select(distinct(Frame.session_id)).join(Object, Object.frame_id == Frame.frame_id)
        .where(Object.source != "human").limit(limit))
    return list(rows.scalars().all())


async def run_error_sweep(run_id: uuid.UUID, *, max_sessions: int = 10, kinds: list[str] | None = None) -> None:
    """Background: run every detector across up to max_sessions, updating the fix queue and the run.

    Resumable. The unit of work is a whole session, so the cursor is the set of sessions already swept, and
    a resumed run skips those instead of re-detecting them. That matters beyond the wasted GPU time: the
    per-session totals are accumulated, and re-sweeping a session already counted would inflate `persisted`
    into a number about how many times the job was interrupted.
    """
    from db.session import get_sessionmaker
    from services.agent.resume import beat, done_set
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
        # Pick up an earlier attempt's cursor and totals. A fresh run has neither and starts from zero.
        prior = await db.get(AgentRun, run_id)
        done = done_set(dict(prior.progress or {})) if prior is not None else set()
        totals: dict = dict(prior.counts or {}) if prior is not None else {}

    totals.setdefault("sessions", 0)
    totals.setdefault("persisted", 0)
    totals.setdefault("by_kind", {})
    if done:
        log.info("agent.error_sweep.resuming", run_id=str(run_id), already_done=len(done))
    try:
        for sid in sessions:
            if str(sid) in done:
                continue
            async with maker() as db:
                res = await run_detection(db, str(sid), kinds)
            totals["sessions"] += 1
            totals["persisted"] += int(res.get("persisted", 0))
            for k, n in (res.get("by_kind") or {}).items():
                totals["by_kind"][k] = totals["by_kind"].get(k, 0) + int(n)
            done.add(str(sid))
            # Cursor and totals move together in one commit. Recording progress separately would let a crash
            # between the two land a session in the cursor whose counts were never saved, and a resume would
            # then skip work it had not actually counted.
            async with maker() as db:
                await beat(db, run_id, progress={"done": sorted(done), "total": len(sessions)},
                           counts=dict(totals))
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
