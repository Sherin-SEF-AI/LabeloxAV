"""Coverage-gap analyzer: profile the labeled corpus so the thin spots are obvious. It reports the class
balance (which ontology classes are rare or missing), the scene-axis coverage (weather / time-of-day /
road-type / density cells that are under-represented against what an AV dataset should span), and the
geographic spread, then names the concrete gaps to fill next. Read-only; pairs with the scenario miner and
active-learning selector, which actually go find candidates for the gaps it flags.
"""

from __future__ import annotations

from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object
from db.models import Session as DbSession

log = get_logger("agent.coverage")

# What an Indian-road AV dataset should span on each scene axis (missing/thin values are gaps).
_EXPECTED = {
    "weather": ["clear", "rain", "fog", "overcast"],
    "time_of_day": ["day", "night", "dusk", "dawn"],
    "road_type": ["urban", "highway", "residential", "rural"],
    "density": ["sparse", "moderate", "dense"],
}
_THIN_FRAC = 0.05   # an axis value below this share of scene-tagged frames is under-covered


async def analyze_coverage(db: AsyncSession, *, rare_object_floor: int = 50) -> dict:
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    # class balance over machine objects
    rows = (await db.execute(
        select(Object.class_id, func.count()).where(Object.source != "human").group_by(Object.class_id))).all()
    counts = {int(cid): int(n) for cid, n in rows}
    class_dist = []
    for c in onto.classes:
        class_dist.append({"class": c.name, "l1": c.l1, "count": counts.get(c.id, 0)})
    class_dist.sort(key=lambda d: d["count"])
    values = [d["count"] for d in class_dist]
    median = sorted(values)[len(values) // 2] if values else 0
    missing = [d["class"] for d in class_dist if d["count"] == 0]
    rare = [d["class"] for d in class_dist if 0 < d["count"] < max(rare_object_floor, int(0.1 * median))]

    # scene-axis coverage
    scenes = (await db.execute(select(Frame.scene).where(Frame.scene.isnot(None)))).scalars().all()
    n_scene = len(scenes)
    axes: dict[str, Counter] = {a: Counter() for a in _EXPECTED}
    for s in scenes:
        for a in _EXPECTED:
            v = (s or {}).get(a)
            if v:
                axes[a][v] += 1
    scene_report: dict[str, dict] = {}
    scene_gaps: list[str] = []
    for a, expected in _EXPECTED.items():
        dist = {v: axes[a].get(v, 0) for v in expected}
        for extra in axes[a]:
            dist.setdefault(extra, axes[a][extra])
        scene_report[a] = dist
        for v in expected:
            share = (dist.get(v, 0) / n_scene) if n_scene else 0.0
            if dist.get(v, 0) == 0:
                scene_gaps.append(f"no {a}={v} frames")
            elif share < _THIN_FRAC:
                scene_gaps.append(f"{a}={v} thin ({dist[v]} frames, {share * 100:.1f}%)")

    # geography
    geo = {(city or "unknown"): int(n) for city, n in
           (await db.execute(select(DbSession.city, func.count()).group_by(DbSession.city))).all()}

    gaps = []
    if missing:
        gaps.append(f"{len(missing)} ontology classes have no labels (e.g. {', '.join(missing[:5])})")
    if rare:
        gaps.append(f"{len(rare)} classes are under-represented (< {max(rare_object_floor, int(0.1 * median))} objects: {', '.join(rare[:5])})")
    gaps += scene_gaps

    log.info("agent.coverage", scene_frames=n_scene, missing=len(missing), rare=len(rare), scene_gaps=len(scene_gaps))
    return {"class_balance": {"median": median, "rarest": class_dist[:10], "missing": missing, "rare": rare},
            "scene_coverage": scene_report, "scene_frames": n_scene, "geo": geo, "gaps": gaps}
