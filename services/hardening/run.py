"""Hardening persistence (M19): evaluate a plane's SLO and record it to the observability ledger."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import PlaneSLO
from services.hardening.slo import evaluate_slos


async def record_slo(db: AsyncSession, plane: str, measurements: dict, window_s: float = 0.0) -> dict:
    """Evaluate and persist a plane SLO tick."""
    result = evaluate_slos(plane, measurements)
    row = PlaneSLO(plane=plane, window_s=window_s, met=result["met"], metrics=measurements,
                   breaches=result["breaches"])
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return {"slo_id": str(row.slo_id), **result}


async def slo_board(db: AsyncSession, limit: int = 50) -> dict:
    """The latest SLO evaluation per plane, for the observability board."""
    rows = (await db.execute(
        select(PlaneSLO).order_by(PlaneSLO.created_at.desc()).limit(min(max(limit, 1), 200)))).scalars().all()
    latest: dict[str, dict] = {}
    for r in rows:
        if r.plane in latest:
            continue
        latest[r.plane] = {"plane": r.plane, "met": r.met, "breaches": r.breaches, "metrics": r.metrics,
                           "created_at": r.created_at.isoformat() if r.created_at else None}
    return {"planes": list(latest.values()),
            "all_met": all(p["met"] for p in latest.values()) if latest else True}
