"""Bulk curation tagging.

Tagging is the explorer's action verb: lasso a cluster, mark it, then use that mark as a filter, an export
slice, or a work queue. Tags are deliberately free-form and orthogonal to the ontology. `object.attrs` is the
typed, per-class attribute schema and `frame.scene` is model-derived; neither should absorb "I want to come
back to these", which is what a tag is.

Applied set-wise and idempotently in SQL so tagging 20k lassoed objects is one statement, not 20k round trips.
"""

from __future__ import annotations

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object
from services.explore.query import frame_select, object_select

log = get_logger("explore_tags")

MAX_TAG_LEN = 64


def normalize_tags(tags: list[str]) -> list[str]:
    """Lowercase, trim, drop empties and duplicates, and bound the length. Keeps the tag namespace from
    fragmenting into 'Night', 'night ' and 'night'."""
    out: list[str] = []
    for t in tags or []:
        s = str(t).strip().lower()[:MAX_TAG_LEN]
        if s and s not in out:
            out.append(s)
    return out


async def apply_tags(db: AsyncSession, *, level: str, pred: dict, add: list[str] | None = None,
                     remove: list[str] | None = None) -> dict:
    """Add and/or remove tags across every row matching the predicate.

    level: "object" or "frame". Returns the number of rows touched. Add is a set-union and remove is a
    set-difference, so re-running the same call is a no-op rather than producing duplicates."""
    if level not in ("object", "frame"):
        raise ValueError("level must be object or frame")
    add_n, rem_n = normalize_tags(add or []), normalize_tags(remove or [])
    if not add_n and not rem_n:
        return {"matched": 0, "added": [], "removed": [], "detail": "no tags supplied"}

    model = Object if level == "object" else Frame
    id_col = Object.object_id if level == "object" else Frame.frame_id

    # Resolve the predicate to concrete ids first. Doing the id resolution and the write as two steps keeps
    # the UPDATE free of the frame/session joins, which UPDATE ... FROM would otherwise need.
    sel = object_select(pred, Object.object_id) if level == "object" else frame_select(pred, Frame.frame_id)
    ids = (await db.execute(sel)).scalars().all()
    if not ids:
        return {"matched": 0, "added": add_n, "removed": rem_n}

    # Union then difference, done in jsonb so the whole selection is one statement.
    await db.execute(
        update(model).where(id_col.in_(ids)).values(tags=_tag_expr(model, add_n, rem_n)))
    await db.commit()
    log.info("explore.tags_applied", level=level, matched=len(ids), add=add_n, remove=rem_n)
    return {"matched": len(ids), "added": add_n, "removed": rem_n}


def _tag_expr(model, add: list[str], remove: list[str]):
    """Build the new tags value: dedup(existing + add) minus remove, entirely in SQL."""
    import sqlalchemy as sa

    base = sa.func.coalesce(model.tags, sa.text("'[]'::jsonb"))
    if add:
        add_json = sa.func.to_jsonb(sa.cast(sa.literal(add), sa.ARRAY(sa.Text)))
        base = base.op("||")(add_json)
    elem = sa.func.jsonb_array_elements_text(base).table_valued("value").alias("e")
    q = sa.select(sa.func.coalesce(sa.func.jsonb_agg(sa.distinct(elem.c.value)), sa.text("'[]'::jsonb")))
    if remove:
        q = q.where(elem.c.value.notin_(remove))
    return q.scalar_subquery()


async def tag_vocabulary(db: AsyncSession, level: str = "object", limit: int = 200) -> list[dict]:
    """Every tag in use at this level with its count, for autocomplete and the facet rail."""
    import sqlalchemy as sa

    model = Object if level == "object" else Frame
    elem = sa.func.jsonb_array_elements_text(model.tags).table_valued("value").alias("t")
    rows = (await db.execute(
        sa.select(elem.c.value, sa.func.count()).select_from(model).join(elem, sa.true())
        .group_by(elem.c.value).order_by(sa.func.count().desc()).limit(limit)
    )).all()
    return [{"tag": v, "count": int(n)} for v, n in rows]
