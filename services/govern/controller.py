"""The autonomy controller (M4.4): the deterministic-first brain that drives the loop. The common path is
a pure state machine with zero model calls; it reads the governance state, queue depth, drift, and the
registry, and at genuine decision points it invokes the champion gate or schedules a burst. It watches new
signal and schedules retrains and relabels on the A100 in off-hours. Every tick is audited.

Judgement (a model call) would plug in only where the deterministic rules are genuinely ambiguous; the
hot path here stays deterministic so the loop is cheap, replayable, and safe to run unattended.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ErrorCandidate, ModelRegistry, Object
from services.activelearn.loop import maybe_retrain, new_signal_count
from services.govern.audit import record
from services.govern.champion import evaluate_and_promote
from services.govern.drift import run_drift_scan
from services.govern.killswitch import get_state

log = get_logger("govern_controller")


async def _queue_depth(db: AsyncSession) -> dict:
    pending_errors = (await db.execute(select(func.count()).select_from(ErrorCandidate).where(
        ErrorCandidate.status == "pending"))).scalar_one()
    review_backlog = (await db.execute(select(func.count()).select_from(Object).where(
        Object.state == "review"))).scalar_one()
    return {"pending_errors": int(pending_errors), "review_backlog": int(review_backlog)}


async def tick(db: AsyncSession, now_hour_utc: int | None = None, schedule_bursts: bool = True) -> dict:
    """One deterministic control step. Returns the actions taken and the state observed."""
    cfg = get_settings().phase4
    state = await get_state(db)
    actions: list[dict] = []

    if not state.loop_enabled:
        await record(db, "controller", "tick_paused", None, {"reason": state.paused_reason})
        return {"status": "paused", "reason": state.paused_reason, "actions": []}

    # 1. drift (cheap, deterministic) - may soft-pause auto-promotion
    drift = await run_drift_scan(db)
    if drift["breached"]:
        actions.append({"action": "drift_pause", "metrics": drift["breached"]})

    # 2. promotion decision points: evaluate registered challengers that are not yet champion
    challengers = (await db.execute(select(ModelRegistry).where(ModelRegistry.is_champion.is_(False)))).scalars().all()
    for ch in challengers:
        if not ch.gold_metrics:
            continue
        res = await evaluate_and_promote(db, ch.model_version, ch.task)
        actions.append({"action": "evaluate_challenger", "model": ch.model_version,
                        "promoted": res.get("promoted"), "paused": res.get("paused", False)})

    # 3. schedule a retrain when enough new signal has accrued and we are in off-hours
    hour = now_hour_utc if now_hour_utc is not None else datetime.now(UTC).hour
    signal = await new_signal_count(db)
    offhours = hour in cfg.govern.offhours_utc
    if schedule_bursts and offhours and signal["total"] >= cfg.activelearn.retrain_min_new:
        # Retrain locally (the on-box GPU worker drains it); cloud dispatch is an unimplemented seam,
        # so scheduling "cloud" here would park a job nothing runs and the loop would never close.
        r = await maybe_retrain(db, compute_target="local", force=True)
        if r.get("triggered"):
            actions.append({"action": "schedule_retrain", "job_id": r.get("job_id")})

    depth = await _queue_depth(db)
    await record(db, "controller", "tick", None,
                 {"actions": [a["action"] for a in actions], "queue": depth,
                  "new_signal": signal["total"], "offhours": offhours, "drift_breached": drift["breached"]})
    return {"status": "active", "actions": actions, "queue": depth, "new_signal": signal,
            "offhours": offhours, "drift": drift, "champion": state.champion_version}
