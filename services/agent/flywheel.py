"""The autonomous active-learning flywheel. One tick:

  1. mine   -- rank the corpus by value (uncertainty x diversity x rarity x error x fn) via the existing
               active-learning selector, and take the highest-value frames.
  2. decide -- run the frame agent on each: auto-accept what the system is sure about, route the rest to a
               human review queue. (Every commit is a reversible AgentRun.)
  3. learn  -- if enough fresh human corrections have accumulated, fire a closed-loop fine-tune via the
               existing maybe_retrain; the retrained champion then serves the next round of inference.

The tool gets smarter from being used: humans only ever see the uncertain tail, their corrections become
training signal, the model improves, and the next tick auto-accepts more. The controller reuses the mature
primitives (selector, frame agent, maybe_retrain) and adds the loop + guardrails around them.

Guardrails: dry_run defaults ON (it reports what it WOULD auto-accept without writing), it yields to any
running training job (single-GPU discipline), and it is bounded by max_frames per tick. Each tick is a pure
function returning a summary; the background runner just calls it in a loop and records progress on an
AgentRun of kind 'flywheel'.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, TrainingJob
from services.agent.frame_agent import commit_frame, plan_frame
from services.agent.policy import PolicyThresholds

log = get_logger("agent.flywheel")


async def _training_running(db: AsyncSession) -> bool:
    return (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first() is not None


async def _top_frames(db: AsyncSession, max_frames: int, session_id: str | None) -> list[tuple[str, float]]:
    """Highest-value frames to work next: score objects, then rank frames by their best object value."""
    from services.activelearn.selector import score_candidates

    scored = await score_candidates(db, session_id=session_id, pool_limit=2000)
    best: dict[str, float] = {}
    for c in scored:
        fid = str(c["frame_id"])
        best[fid] = max(best.get(fid, 0.0), float(c.get("value", 0.0)))
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    return ranked[:max_frames]


async def flywheel_tick(
    db: AsyncSession,
    *,
    max_frames: int = 25,
    policy: PolicyThresholds | None = None,
    session_id: str | None = None,
    dry_run: bool = True,
    created_by: str = "flywheel",
) -> dict:
    """One iteration of the loop. dry_run=True (default) plans only and writes nothing."""
    policy = policy or PolicyThresholds()
    if not dry_run and await _training_running(db):
        return {"skipped": "a training job holds the GPU; flywheel yields", "frames": 0}

    frames = await _top_frames(db, max_frames, session_id)
    agg = {"frames": 0, "auto_accept": 0, "review": 0, "annotate": 0, "demoted_by_critic": 0}
    child_runs: list[str] = []
    for fid, _value in frames:
        try:
            if dry_run:
                res = await plan_frame(db, uuid.UUID(fid), policy)
                counts = res["counts"]
            else:
                run = await commit_frame(db, uuid.UUID(fid), policy, created_by=created_by)
                counts = run["counts"]
                child_runs.append(run["run_id"])
        except Exception as exc:  # noqa: BLE001 -- one bad frame must not stall the loop
            log.warning("flywheel.frame_failed", frame_id=fid, error=str(exc))
            continue
        agg["frames"] += 1
        for k in ("auto_accept", "review", "annotate", "demoted_by_critic"):
            agg[k] += int(counts.get(k, 0))
    agg["dry_run"] = dry_run
    agg["child_runs"] = child_runs
    return agg


async def retrain_if_ready(db: AsyncSession, *, force: bool = False, compute_target: str = "cloud") -> dict:
    """Close the loop: fire a closed-loop fine-tune if enough human corrections have accumulated."""
    from services.activelearn.loop import maybe_retrain

    return await maybe_retrain(db, compute_target=compute_target, force=force)


async def run_flywheel(
    run_id: uuid.UUID,
    *,
    ticks: int,
    max_frames: int,
    policy: PolicyThresholds,
    session_id: str | None,
    dry_run: bool,
    created_by: str,
) -> None:
    """Background runner: tick the loop `ticks` times, record progress on the flywheel AgentRun, then try a
    retrain. Guarded and self-limiting; safe to launch as an asyncio task."""
    from db.session import get_sessionmaker

    maker = get_sessionmaker()
    totals = {"frames": 0, "auto_accept": 0, "review": 0, "annotate": 0, "demoted_by_critic": 0}
    child_runs: list[str] = []
    try:
        for t in range(ticks):
            async with maker() as db:
                res = await flywheel_tick(db, max_frames=max_frames, policy=policy,
                                          session_id=session_id, dry_run=dry_run, created_by=created_by)
                if "skipped" in res:
                    break
                for k in totals:
                    totals[k] += int(res.get(k, 0))
                child_runs.extend(res.get("child_runs", []))
                run = await db.get(AgentRun, run_id)
                if run is not None:
                    run.counts = {**totals, "ticks_done": t + 1, "dry_run": dry_run}
                    run.changes = {"child_runs": child_runs}
                    await db.commit()
        retrain = {}
        if not dry_run:
            async with maker() as db:
                retrain = await retrain_if_ready(db)
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run is not None:
                run.status = "committed" if not dry_run else "planned"
                run.counts = {**totals, "ticks_done": ticks, "dry_run": dry_run, "retrain": retrain}
                await db.commit()
        log.info("flywheel.done", run_id=str(run_id), **totals, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        log.error("flywheel.failed", run_id=str(run_id), error=str(exc))
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run is not None:
                run.status = "error"
                run.error = str(exc)
                await db.commit()
