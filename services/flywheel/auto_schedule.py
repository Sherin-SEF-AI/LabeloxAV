"""Nightly adaptive-flywheel hook for the governance daemon. Off-hours, once per calendar day, it runs one
adaptive cycle from the live corpus and refreshes the fleet collection orders, so the morning worklist and the
dispatch board reflect the current corpus. It proposes only: it runs the cycle (records it) and the collection
orders, but does not auto-dispatch objects into the review queue, so a human still decides what to label. The
FlywheelCycle table is the once-per-day marker, so a restart mid-window never double-runs."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import FlywheelCycle
from services.agent.fleet_dispatch import plan_from_flywheel
from services.flywheel.auto import run_auto_cycle

log = get_logger("flywheel.auto_schedule")


async def _ran_corpus_cycle_today(db: AsyncSession) -> bool:
    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    n = (await db.execute(
        select(func.count()).select_from(FlywheelCycle)
        .where(FlywheelCycle.created_at >= day_start,
               FlywheelCycle.signals["source"].astext == "corpus"))).scalar_one()
    return n > 0


async def maybe_run_flywheel(db: AsyncSession) -> dict:
    """Run one corpus cycle + refresh collection orders, at most once per day."""
    if await _ran_corpus_cycle_today(db):
        return {"ran": False, "reason": "already ran today"}
    plan = await run_auto_cycle(db, build_work_order=False)
    orders = await plan_from_flywheel(db, cycle_id=None, created_by="flywheel_nightly")
    log.info("flywheel.nightly", cycle=plan["cycle_id"], allocated=plan["allocated"], orders=orders["orders"])
    return {"ran": True, "cycle_id": plan["cycle_id"], "allocated": plan["allocated"],
            "orders": orders["orders"]}
