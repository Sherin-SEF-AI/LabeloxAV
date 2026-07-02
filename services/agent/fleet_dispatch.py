"""The Fleet Dispatch agent: fuses the coverage gaps, the fleet's geography, and the weather forecast into
daily per-vehicle collection orders, closing the acquisition loop.

The labeling agents close the loop on data the platform already has; this one closes the loop on data it
lacks, and only a platform that owns the fleet can act on it. It reads what the corpus is starved of
(coverage gaps), which vehicles and cities the fleet operates, and (optionally) the forecast, and proposes
ranked orders like "Vehicle 7: BLR 18:00-22:00, rain forecast, starved of night-rain data". It proposes;
a human dispatches.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import delete, distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import CollectionOrder
from db.models import Session as DbSession

log = get_logger("agent.fleet_dispatch")

_KIND = "fleet_dispatch"

_WINDOWS = {"night": "18:00-22:00", "dusk": "17:00-18:30", "dawn": "05:30-06:30", "day": "10:00-16:00"}


def _parse_gap(gap: str) -> tuple[str, str, str] | None:
    """Turn a coverage-gap string into (gap_kind, value, human target). Returns None for gaps that are not
    collectable by driving (e.g. an unlabeled class with no imagery is a labeling task, not a collection one)."""
    m = re.match(r"(weather|time_of_day|road_type)=(\w+) thin", gap)
    if m:
        kind, val = m.group(1), m.group(2)
        noun = {"weather": "conditions", "time_of_day": "driving", "road_type": "roads"}[kind]
        return kind, val, f"{val} {noun}"
    m = re.match(r"density=(\w+) thin", gap)
    if m:
        return "density", m.group(1), f"{m.group(1)}-density traffic"
    return None


def _window_for(kind: str, value: str) -> str:
    if kind == "time_of_day":
        return _WINDOWS.get(value, _WINDOWS["day"])
    if kind == "weather" and value in ("rain", "fog"):
        return "any (watch forecast)"
    return _WINDOWS["day"]


def _priority(kind: str, value: str, forecast: str) -> float:
    base = {"weather": 0.8, "time_of_day": 0.7, "road_type": 0.5, "density": 0.5}.get(kind, 0.4)
    if kind == "weather" and forecast == value:      # collect rain data when rain is actually forecast
        base += 0.4
    return round(base, 3)


async def plan_collection(db: AsyncSession, *, created_by: str | None = None, max_orders: int = 12) -> dict:
    """Generate ranked collection orders from the current gaps + fleet + forecast. Supersedes prior proposals."""
    from services.agent import weather
    from services.agent.coverage import analyze_coverage

    cov = await analyze_coverage(db)
    gaps = cov.get("gaps", [])
    vehicles = [v for (v,) in (await db.execute(select(distinct(DbSession.vehicle_id)))).all() if v]
    cities = [c for (c,) in (await db.execute(select(distinct(DbSession.city)).where(DbSession.city.isnot(None)))).all()]
    if not vehicles:
        vehicles = ["unassigned"]

    proposals: list[CollectionOrder] = []
    i = 0
    for gap in gaps:
        parsed = _parse_gap(gap)
        if parsed is None:
            continue
        kind, value, target = parsed
        city = cities[i % len(cities)] if cities else None
        fc = (await weather.forecast(city))["condition"] if kind == "weather" else "n/a"
        vehicle = vehicles[i % len(vehicles)]
        proposals.append(CollectionOrder(
            vehicle_id=vehicle, city=city, area=None, window=_window_for(kind, value), target=target,
            gap_kind=kind, forecast=fc, priority=_priority(kind, value, fc), status="proposed",
            created_by=created_by or _KIND))
        i += 1

    await db.execute(delete(CollectionOrder).where(CollectionOrder.status == "proposed"))
    for o in sorted(proposals, key=lambda x: x.priority, reverse=True)[:max_orders]:
        db.add(o)
    await db.commit()
    log.info("fleet.plan", gaps=len(gaps), orders=min(len(proposals), max_orders), vehicles=len(vehicles))
    return {"gaps": len(gaps), "orders": min(len(proposals), max_orders), "vehicles": len(vehicles)}


async def list_orders(db: AsyncSession, status: str = "proposed", limit: int = 50) -> list[dict]:
    rows = (await db.execute(select(CollectionOrder).where(CollectionOrder.status == status)
                             .order_by(CollectionOrder.priority.desc()).limit(limit))).scalars().all()
    return [{"order_id": str(o.order_id), "vehicle_id": o.vehicle_id, "city": o.city, "window": o.window,
             "target": o.target, "gap_kind": o.gap_kind, "forecast": o.forecast,
             "priority": o.priority, "status": o.status,
             "summary": _summary(o)} for o in rows]


def _summary(o: CollectionOrder) -> str:
    loc = " ".join(x for x in [o.city, o.area] if x) or "any area"
    fc = f", {o.forecast} forecast" if o.forecast and o.forecast not in ("n/a", "unknown") else ""
    return f"{o.vehicle_id}: {loc} {o.window}{fc}, starved of {o.target}"


async def set_status(db: AsyncSession, order_id: uuid.UUID, status: str) -> dict:
    o = await db.get(CollectionOrder, order_id)
    if o is None:
        raise ValueError("order not found")
    o.status = status
    await db.commit()
    return {"order_id": str(order_id), "status": status}
