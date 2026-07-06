"""Adaptive flywheel persistence (M17). Run one adaptive cycle and record it to the FlywheelCycle ledger, so
the label spend and collection tasking of each cycle are traceable to the failures and gaps that justified
them."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import FlywheelCycle
from services.flywheel.controller import adaptive_cycle


async def run_and_record(db: AsyncSession, regressions: list[dict], odd_gaps: list[dict],
                         total_label_budget: int, total_fleet_samples: int,
                         safety_slices: list[str] | None = None, safety_floor: int = 0) -> dict:
    """Run an adaptive cycle and persist it. Returns the plan plus the recorded cycle_id."""
    plan = adaptive_cycle(regressions, odd_gaps, total_label_budget, total_fleet_samples,
                          safety_slices, safety_floor)
    row = FlywheelCycle(
        label_budget=total_label_budget,
        signals={"regressions": regressions, "odd_gaps": odd_gaps, "safety_slices": safety_slices or []},
        allocation=plan["allocation"],
        collection_tasks=plan["collection_tasks"],
        rationale=plan["rationale"],
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return {"cycle_id": str(row.cycle_id), **plan}


async def recent_cycles(db: AsyncSession, limit: int = 20) -> dict:
    """The adaptive-cycle history, newest first."""
    rows = (await db.execute(
        select(FlywheelCycle).order_by(FlywheelCycle.created_at.desc()).limit(min(max(limit, 1), 100)))
    ).scalars().all()
    return {"cycles": [
        {"cycle_id": str(r.cycle_id), "label_budget": r.label_budget, "allocation": r.allocation,
         "collection_tasks": r.collection_tasks, "rationale": r.rationale,
         "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]}
