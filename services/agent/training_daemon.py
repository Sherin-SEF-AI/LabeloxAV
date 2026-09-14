"""Self-improving flywheel daemon: run the whole loop on a schedule and keep the champion honest.

flywheel_cycle is one turn of the loop -- mine by value, run the frame agent over the top frames
(auto-accept the sure, escalate the rest), then fire a closed-loop fine-tune if enough human corrections
have accumulated (the existing maybe_retrain, which the training pipeline evaluates and auto-promotes or
rejects through the champion gate). Call it on a schedule and the tool improves from use with a human only
in the exception path.

check_gold_drift watches the champion the other direction: it re-evaluates the serving champion on the gold
set and compares to the accuracy it was promoted at; if it has regressed beyond tolerance the world has
shifted under it, so it engages the kill switch (pause + roll back to the prior champion) and records a
drift governance event. Both degrade gracefully when there is no champion, gold set, or GPU.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun

log = get_logger("agent.training_daemon")


async def flywheel_cycle(db: AsyncSession, *, max_frames: int = 25, dry_run: bool = True,
                         retrain: bool = False, created_by: str | None = None) -> dict:
    """One full turn: a flywheel tick, then (when not dry_run and retrain) a retrain-if-ready. Recorded on an
    AgentRun so the cycle is auditable. dry_run reports what it would auto-accept without writing."""
    from services.agent.flywheel import flywheel_tick, retrain_if_ready
    from services.agent.policy import PolicyThresholds

    run_id = uuid.uuid4()
    tick = await flywheel_tick(db, max_frames=max_frames, policy=PolicyThresholds(),
                               dry_run=dry_run, created_by=created_by or "daemon")
    retrain_res: dict = {"attempted": False}
    if retrain and not dry_run and "skipped" not in tick:
        retrain_res = await retrain_if_ready(db)
        retrain_res["attempted"] = True

    db.add(AgentRun(run_id=run_id, kind="flywheel_cycle", scope={"max_frames": max_frames},
                    status="committed" if not dry_run else "planned",
                    policy={"dry_run": dry_run, "retrain": retrain}, counts={**tick, "retrain": retrain_res},
                    changes={}, critic={}, created_by=created_by or "daemon"))
    await db.commit()
    log.info("agent.flywheel_cycle", run_id=str(run_id), frames=tick.get("frames", 0), dry_run=dry_run)
    return {"run_id": str(run_id), "tick": tick, "retrain": retrain_res}


async def _eval_champion_on_gold(db: AsyncSession, champ) -> float | None:
    """Re-evaluate the champion weights on the latest sealed gold set; None if it cannot be run here."""
    try:
        from sqlalchemy import select

        from db.models import GoldSet
        from services.training.eval import evaluate

        gs = (await db.execute(select(GoldSet).where(GoldSet.data_yaml_uri.isnot(None)).limit(1))).scalar_one_or_none()
        if gs is None or not gs.data_yaml_uri or not champ.weights_uri:
            return None
        metrics = evaluate(champ.weights_uri, gs.data_yaml_uri)
        return float(metrics.get("map", metrics.get("map50", 0.0)))
    except Exception as exc:  # noqa: BLE001 -- no gold/weights/GPU here: skip, do not false-rollback
        log.info("agent.gold_drift.eval_unavailable", error=str(exc))
        return None


async def check_gold_drift(db: AsyncSession, *, tolerance: float = 0.03, task: str = "detection",
                           evaluator=None) -> dict:
    """Re-evaluate the champion on gold; roll back + pause the loop if it has regressed beyond tolerance."""
    from services.govern.champion import get_champion
    from services.govern.killswitch import engage

    champ = await get_champion(db, task)
    if champ is None:
        return {"status": "no_champion"}
    baseline = float((champ.gold_metrics or {}).get("map", (champ.gold_metrics or {}).get("map50", 0.0)))
    current = await (evaluator(db, champ) if evaluator else _eval_champion_on_gold(db, champ))
    if current is None:
        return {"status": "cannot_evaluate", "champion": champ.model_version, "baseline_map": round(baseline, 4)}
    drop = baseline - current
    if drop > tolerance:
        res = await engage(db, reason=f"gold drift: mAP {baseline:.3f} -> {current:.3f} (-{drop:.3f})", task=task)
        log.info("agent.gold_drift.rollback", champion=champ.model_version, baseline=baseline, current=current)
        return {"status": "rolled_back", "champion": champ.model_version, "baseline_map": round(baseline, 4),
                "current_map": round(current, 4), "drop": round(drop, 4), "governance": res}
    return {"status": "healthy", "champion": champ.model_version, "baseline_map": round(baseline, 4),
            "current_map": round(current, 4), "drop": round(drop, 4)}

async def maybe_check_gold_drift(db) -> dict:
    """Off-hours hook for the runtime scheduler: the champion-degradation check, once per calendar day.

    `check_gold_drift` has existed since the daemon work and sat behind `POST /api/agent/gold-drift`,
    a button nobody presses. It is the only mechanism that catches a promoted champion regressing on the
    gold yardstick after the fact, and its remedy path (killswitch engage: pause the loop, roll back to
    the previous champion) already exists and is tested. A safety check that only runs when somebody
    remembers to ask is a dashboard, not a check.

    Recorded as an AgentRun the same way the other nightly agents are, so the once-per-day guard, the
    audit trail and the morning digest all come from the shared spine rather than a private marker.
    """
    from datetime import UTC, datetime

    from services.agent.runtime.report import finish_run, launch, ran_since

    kind = "gold_drift_check"
    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, kind, day_start):
        return {"ran": False, "reason": "already ran today"}

    async def worker(run_id):
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as wdb:
            res = await check_gold_drift(wdb)
        # `rolled_back` is the one outcome that must not drown in the nightly noise: the loop just
        # unpromoted a model on its own. It is already audited and killswitched by check_gold_drift
        # itself; the run status carries it to the digest.
        await finish_run(run_id, status="committed" if res.get("status") != "error" else "error",
                         report=res)

    return {"ran": True, **(await launch(db, kind, worker, created_by="scheduler"))}
