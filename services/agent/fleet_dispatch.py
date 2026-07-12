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

from sqlalchemy import delete, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import CollectionOrder, FlywheelCycle
from db.models import Session as DbSession

log = get_logger("agent.fleet_dispatch")

_KIND = "fleet_dispatch"
_FLYWHEEL_KIND = "flywheel_cycle"   # collection orders sourced from a flywheel cycle's collection tasks

_WINDOWS = {"night": "18:00-22:00", "dusk": "17:00-18:30", "dawn": "05:30-06:30", "day": "10:00-16:00"}

_TOD = {"day", "night", "dusk", "dawn"}
_WEATHER = {"clear", "overcast", "rain", "fog", "snow"}


def _classify_cell(cell: str) -> tuple[str, str, str | None]:
    """A flywheel collection-task cell is either a scene cell ('night_rain') or a class ('elephant'). Returns
    (kind, a, b): ('scene', time_of_day, weather) or ('class', class_name, None)."""
    parts = cell.rsplit("_", 1)
    if len(parts) == 2 and parts[0] in _TOD and parts[1] in _WEATHER:
        return "scene", parts[0], parts[1]
    return "class", cell, None


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


async def plan_from_flywheel(db: AsyncSession, *, cycle_id: uuid.UUID | None = None,
                             created_by: str | None = None, max_orders: int = 24) -> dict:
    """Turn a flywheel cycle's collection tasks into ranked collection orders. Where plan_collection parses
    coverage strings and skips classes, this consumes the flywheel's real signals: scene cells (night_rain,
    day_fog) become windowed, forecast-aware drives, and empty safety classes (elephant, camel, buffalo)
    become class-collection orders, because a species with no imagery is found by driving, not by labeling.
    Each order carries the target sample count the cycle computed. Supersedes prior flywheel-sourced orders
    only, so it coexists with plan_collection."""
    from services.agent import weather

    cycle = (await db.get(FlywheelCycle, cycle_id)) if cycle_id else (await db.execute(
        select(FlywheelCycle).order_by(FlywheelCycle.created_at.desc()).limit(1))).scalars().first()
    if cycle is None:
        raise ValueError("no flywheel cycle on record; run one first")

    tasks = sorted(cycle.collection_tasks or [], key=lambda t: t.get("priority", 0.0), reverse=True)[:max_orders]
    vehicles = [v for (v,) in (await db.execute(select(distinct(DbSession.vehicle_id)))).all() if v] or ["unassigned"]
    # the fleet's dominant operating city (most sessions), so orders default to where the fleet actually drives
    city_row = (await db.execute(
        select(DbSession.city, func.count()).where(DbSession.city.isnot(None))
        .group_by(DbSession.city).order_by(func.count().desc()).limit(1))).first()
    default_city = city_row[0] if city_row else None

    proposals: list[CollectionOrder] = []
    for i, t in enumerate(tasks):
        kind, a, b = _classify_cell(t.get("cell", ""))
        need = t.get("target_count")
        need_str = f" (need {need:,})" if need else ""
        vehicle = vehicles[i % len(vehicles)]
        if kind == "scene":
            tod, wx = a, b
            window = _WINDOWS.get(tod, _WINDOWS["day"])
            fc = (await weather.forecast(default_city))["condition"] if wx in ("rain", "fog") else "n/a"
            proposals.append(CollectionOrder(
                vehicle_id=vehicle, city=default_city, area=None, window=window,
                target=f"{tod} {wx} driving{need_str}", gap_kind="scene", forecast=fc,
                priority=round(float(t.get("priority", 0.0)), 4), status="proposed",
                created_by=created_by or _FLYWHEEL_KIND))
        else:
            proposals.append(CollectionOrder(
                vehicle_id=vehicle, city=default_city, area=None, window=_WINDOWS["day"],
                target=f"{a} sightings{need_str}", gap_kind="class", forecast="n/a",
                priority=round(float(t.get("priority", 0.0)), 4), status="proposed",
                created_by=created_by or _FLYWHEEL_KIND))

    # supersede only prior flywheel-sourced proposals (coexist with plan_collection's orders)
    await db.execute(delete(CollectionOrder).where(
        CollectionOrder.status == "proposed", CollectionOrder.created_by == (created_by or _FLYWHEEL_KIND)))
    for o in proposals:
        db.add(o)
    await db.commit()
    log.info("fleet.plan_flywheel", cycle=str(cycle.cycle_id), orders=len(proposals))
    return {"cycle_id": str(cycle.cycle_id), "orders": len(proposals),
            "scene": sum(1 for o in proposals if o.gap_kind == "scene"),
            "classes": sum(1 for o in proposals if o.gap_kind == "class")}


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
