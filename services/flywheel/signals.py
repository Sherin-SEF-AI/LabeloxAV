"""Real corpus signals for the adaptive flywheel (M17 wired to live data). The controller consumes VERDYX
safety regressions and SIEVYX ODD coverage gaps; this collects both from the actual corpus instead of typed
input, so a cycle steers the label budget and collection at what the fleet is genuinely missing.

Two real signals:
- class starvation: a safety-critical class (VRU, animal, two-wheeler by the ontology l1) whose share of the
  labeled corpus is below a floor is a standing labeling demand. cattle, cyclist, and the long-tail VRU are the
  classes the safety gate blocks on, so they surface here as regressions the controller routes to labeling.
- ODD coverage gaps: scene cells (time-of-day x weather) underrepresented against a target operational design
  domain become collection tasks. Night, rain, fog, and dusk are the starved cells the daytime-heavy fleet
  needs collected, not labeled.

The pure functions (starvation_demands, scene_fleet_counts) are testable without a database; the async
functions run the real queries."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, Object, OntologyClass
from services.sievyx.odd import coverage_gaps

# the ontology l1 groups that carry a safety obligation; vru + animal are the most protected (a miss is a
# potential collision with a person or animal), two-wheeler riders are safety-relevant.
SAFETY_L1 = {"vru", "animal", "two_wheeler"}
MOST_PROTECTED_L1 = {"vru", "animal"}

# a target India ODD: the daytime-clear majority is capped so night, rain, fog, and dusk earn real share. The
# fractions need not sum to 1; coverage_gaps normalizes them.
DEFAULT_TARGET_ODD = {
    "day_clear": 0.28, "day_overcast": 0.14, "day_rain": 0.07, "day_fog": 0.03,
    "night_clear": 0.10, "night_overcast": 0.07, "night_rain": 0.09, "night_fog": 0.03,
    "dusk_clear": 0.05, "dusk_overcast": 0.03, "dawn_clear": 0.05, "dawn_overcast": 0.03,
}


def starvation_demands(class_counts: list[dict], total: int, min_share: float = 0.003) -> list[dict]:
    """Turn per-class object counts into labeling demands for the safety-critical classes below their floor.

    Each class_count is {name, l1, class_id, count}. A safety class whose share of the corpus is under
    ``min_share`` is starved; its deficit (how far below the floor) becomes a regression the controller routes
    to labeling, with the VRU/animal classes marked protected so the allocator floors them. Returns the demands
    worst-starved first; a class already at or above its floor produces nothing."""
    demands = []
    for c in class_counts:
        if c["l1"] not in SAFETY_L1:
            continue
        share = (c["count"] / total) if total else 0.0
        deficit = min_share - share
        if deficit <= 0:
            continue
        demands.append({
            "slice": c["name"], "class_id": c["class_id"], "l1": c["l1"], "count": c["count"],
            "share": round(share, 6), "delta": -round(deficit, 6),
            "protected": c["l1"] in MOST_PROTECTED_L1,
        })
    demands.sort(key=lambda d: d["delta"])   # most negative (most starved) first
    return demands


def scene_fleet_counts(scene_rows: list[dict]) -> dict[str, int]:
    """Aggregate per-frame scene tags into ODD cell counts keyed ``{time_of_day}_{weather}``. A frame missing
    either axis is skipped (an untagged frame is not evidence of coverage)."""
    counts: dict[str, int] = {}
    for r in scene_rows:
        tod, weather = r.get("time_of_day"), r.get("weather")
        if not tod or not weather:
            continue
        cell = f"{tod}_{weather}"
        counts[cell] = counts.get(cell, 0) + 1
    return counts


async def class_starvation(db: AsyncSession, min_share: float = 0.003) -> dict:
    """Real class-starvation demands from the object table joined to the ontology."""
    rows = (await db.execute(
        select(OntologyClass.id, OntologyClass.name, OntologyClass.l1, func.count(Object.object_id))
        .join(Object, Object.class_id == OntologyClass.id, isouter=True)
        .where(OntologyClass.l1.in_(SAFETY_L1))
        .group_by(OntologyClass.id, OntologyClass.name, OntologyClass.l1))).all()
    total = (await db.execute(select(func.count(Object.object_id)))).scalar_one()
    class_counts = [{"class_id": r[0], "name": r[1], "l1": r[2], "count": r[3]} for r in rows]
    demands = starvation_demands(class_counts, total, min_share)
    return {"total_objects": total, "min_share": min_share, "n_starved": len(demands), "demands": demands}


async def odd_gaps(db: AsyncSession, target_odd: dict[str, float] | None = None, min_gap: float = 0.02) -> dict:
    """Real ODD coverage gaps from the frame scene tags."""
    rows = (await db.execute(
        select(Frame.scene).where(Frame.scene.isnot(None)))).scalars().all()
    scene_rows = [{"time_of_day": (s or {}).get("time_of_day"), "weather": (s or {}).get("weather")}
                  for s in rows]
    fleet_counts = scene_fleet_counts(scene_rows)
    return coverage_gaps(fleet_counts, target_odd or DEFAULT_TARGET_ODD, min_gap)


async def corpus_totals(db: AsyncSession) -> dict:
    """Corpus size, used as the controller's fleet-sample base for collection target counts."""
    frames = (await db.execute(select(func.count(Frame.frame_id)))).scalar_one()
    objects = (await db.execute(select(func.count(Object.object_id)))).scalar_one()
    return {"frames": frames, "objects": objects}
