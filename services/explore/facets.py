"""Faceted counts for the explorer sidebar.

Every facet is computed under the CURRENT predicate, minus its own clause. That "drop your own clause" rule is
what makes a facet sidebar usable: if the class facet were computed with the class filter applied, selecting
"autorickshaw" would leave the class list showing only autorickshaw and you could never widen or switch to a
second class without clearing the filter first. Excluding the facet's own clause means each bar always answers
"how many would I get if I picked this instead", which is the question a curator is actually asking.

Counts come from the same SQL predicate the selection uses (services/explore/query.py), so the number on a bar
is exactly the number of rows a lasso or an export over that filter would return.
"""

from __future__ import annotations

from sqlalchemy import func, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object
from db.models import Session as DbSession
from services.explore.query import frame_clauses, object_clauses, object_select

log = get_logger("explore_facets")

# Confidence buckets, chosen to straddle the gate thresholds so a curator can see the review band directly.
_CONF_BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]

_SCENE_AXES = ("weather", "time_of_day", "road_type", "density")


def _without(pred: dict, *keys: str) -> dict:
    return {k: v for k, v in (pred or {}).items() if k not in keys}


async def _counts_by(db: AsyncSession, pred: dict, column, *, join_session: bool = False) -> list[dict]:
    """GROUP BY one column under the predicate. Returns [{value, count}] descending."""
    stmt = (select(column, func.count()).select_from(Object)
            .join(Frame, Frame.frame_id == Object.frame_id))
    if join_session or pred.get("cities"):
        stmt = stmt.join(DbSession, DbSession.session_id == Frame.session_id)
    for c in object_clauses(pred) + frame_clauses(pred):
        stmt = stmt.where(c)
    rows = (await db.execute(stmt.group_by(column).order_by(func.count().desc()).limit(200))).all()
    return [{"value": v, "count": int(n)} for v, n in rows if v is not None]


async def object_facets(db: AsyncSession, pred: dict | None = None) -> dict:
    """The full facet set for the object-level explorer, each computed with its own clause dropped."""
    pred = pred or {}
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    # class: drop class_names so every class stays selectable
    cls_rows = await _counts_by(db, _without(pred, "class_names"), Object.class_id)
    classes = []
    for r in cls_rows:
        try:
            classes.append({"value": onto.by_id(int(r["value"])).name, "class_id": int(r["value"]),
                            "count": r["count"]})
        except Exception:  # noqa: BLE001 - a class id with no ontology entry is skipped, not fatal
            continue

    states = await _counts_by(db, _without(pred, "states"), Object.state)
    sources = await _counts_by(db, _without(pred, "sources"), Object.source)
    cities = await _counts_by(db, _without(pred, "cities"), DbSession.city, join_session=True)

    scene: dict[str, list[dict]] = {}
    for axis in _SCENE_AXES:
        scene[axis] = await _counts_by(db, _without(pred, axis), Frame.scene[axis].astext)

    # confidence histogram: one count per bucket, each under the predicate minus the conf clauses
    conf_pred = _without(pred, "min_conf", "max_conf")
    conf: list[dict] = []
    for lo, hi in _CONF_BUCKETS:
        stmt = object_select({**conf_pred}, func.count()).where(Object.conf >= lo, Object.conf < hi)
        n = (await db.execute(stmt)).scalar_one()
        conf.append({"value": f"{lo:.1f}-{min(hi, 1.0):.1f}", "lo": lo, "hi": hi, "count": int(n)})

    tags = await _tag_counts(db, _without(pred, "tags"), Object.tags)

    total = (await db.execute(object_select(pred, func.count()))).scalar_one()
    return {"total": int(total), "classes": classes, "states": states, "sources": sources,
            "cities": cities, "scene": scene, "conf": conf, "tags": tags}


async def _tag_counts(db: AsyncSession, pred: dict, column) -> list[dict]:
    """Count each distinct tag under the predicate. jsonb_array_elements_text unnests the tag array so a row
    with three tags contributes to all three counts."""
    elem = func.jsonb_array_elements_text(column).table_valued("value").alias("t")
    stmt = (select(elem.c.value, func.count())
            .select_from(Object)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .join(elem, true()))
    if pred.get("cities"):
        stmt = stmt.join(DbSession, DbSession.session_id == Frame.session_id)
    for c in object_clauses(pred) + frame_clauses(pred):
        stmt = stmt.where(c)
    rows = (await db.execute(stmt.group_by(elem.c.value).order_by(func.count().desc()).limit(100))).all()
    return [{"value": v, "count": int(n)} for v, n in rows]
