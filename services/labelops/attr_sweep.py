"""One attribute at a time, over a grid of crops, with the track as the unit wherever the attribute allows.

The write path for attributes has been complete and reversible for a long time: `apply_review_batch` with
`action="set_attrs"` validates against the ontology, merges, re-derives, bumps the lock version, writes a
Review row and lands in one revertible run. It had exactly one caller, the modal that opens after somebody
has already made a correction. Nothing anywhere asked the prior question, which is *where the attributes
are missing*, so there was no way to go and fill them.

Measured over the corpus, that gap is most of the attribute schema:

    attribute            in scope      set      missing   unit
    occlusion              578,436      829      577,607   frame
    load_type              282,061        0      282,061   track
    cargo_secured          282,061        0      282,061   track
    occupant_count         139,613        0      139,613   track
    helmet                 187,638    1,024      186,614   track

Measured 2026-09-01 by `coverage()` in this module, so the figures are the ones it reports rather than a
separate count taken a different way.

`load_type`, `cargo_secured` and `occupant_count` are exactly the axes the India work added, and every one
of them is empty. `triple_riding` is derived from `occupant_count`, so it is empty too and will stay empty
until somebody answers the question this module exists to ask.

**The track is the unit for anything that describes the object.** What a truck is carrying does not change
between frames, so asking about the same truck fifty times is not fifty answers, it is one answer and
forty-nine chances to disagree with it. `AttributeDef.track_constant` says which attributes those are, and
the queue offers one representative crop per track, the largest box on it, with the member count attached.
That is the same leverage tube review gets, and it is large: over the real corpus the first eight
`load_type` crops this queue offers cover 2,199 objects between them.

**A sweep fills; it does not overwrite.** Members that already carry a value keep it unless the caller
explicitly asks otherwise, so re-running a sweep is safe and a track-wide answer can never silently
replace a value somebody set deliberately on one frame.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from sqlalchemy import Float, and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object

log = get_logger("attr_sweep")

# A queue page. Big enough that the sprite sheet is worth building and small enough that the annotator is
# looking at one screenful; the sheet endpoint caps at 400.
DEFAULT_LIMIT = 60
MAX_LIMIT = 200

# Rejected objects are not work: nobody needs to know what a box that is not there was carrying.
_LIVE_STATES = ("review", "auto_accept", "accepted", "annotate", "submitted")

_SAFE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def classes_in_scope(onto: Any, attr: str) -> list[int]:
    """Class ids the attribute applies to, via the l1 scope plus per-class extras."""
    out = []
    for c in onto.classes:
        applies = onto.attrs_for_class(c.id)
        if applies is None or attr in applies:
            out.append(c.id)
    return out


def _missing(attr: str):
    """SQL for "this object does not carry the attribute".

    `~Object.attrs.has_key(k)` alone is NULL when `attrs` itself is NULL, and a NULL predicate excludes the
    row, so the plain negation silently drops every object with no attributes at all.

    Measured today: `object.attrs` is a nullable column and 0 of 578,436 rows are actually NULL, because
    the ORM default fills `{}`. So this is a guard against a shape the schema permits, not a fix for a
    live defect. It is here because the same predicate on a genuinely NULL column has already cost this
    repo a sweep: `Frame.scene` is NULL on most frames, and a rarity pass spelled this way reported
    `remaining: 0` after scoring 1,780 of 41,752.
    """
    return or_(Object.attrs.is_(None), ~Object.attrs.has_key(attr))


async def coverage(db: AsyncSession, onto: Any, *, session_id: UUID | None = None) -> dict:
    """Per attribute: how many objects it applies to, how many carry it, how many are missing it.

    One table scan with an aggregate per attribute rather than one query per attribute. Measured over
    578,436 objects: 1.2s for all 44, against 2.2s for a single attribute asked on its own, because the
    cost is the scan and not the arithmetic.
    """
    keys = [k for k, d in onto.attributes.items() if not d.derived_from]
    bad = [k for k in keys if not _SAFE_KEY.match(k)]
    if bad:
        # The names are interpolated into SQL below. They come from a governed YAML, so this cannot fire
        # today; it is here so that it cannot start firing quietly if that stops being true.
        raise ValueError(f"attribute names unusable as SQL identifiers: {bad}")

    cols = ",\n  ".join(f"count(*) filter (where o.attrs ? '{k}') as \"{k}\"" for k in keys)
    where = "o.state <> 'rejected'"
    params: dict[str, Any] = {}
    join = ""
    if session_id is not None:
        join = "join frame f on f.frame_id = o.frame_id"
        where += " and f.session_id = :sid"
        params["sid"] = str(session_id)
    sql = text(f"select o.class_id, count(*) as n,\n  {cols}\nfrom object o {join}\n"
               f"where {where}\ngroup by o.class_id")
    rows = (await db.execute(sql, params)).mappings().all()

    out: dict[str, dict] = {}
    for k in keys:
        d = onto.attributes[k]
        out[k] = {"attribute": k, "type": d.type, "values": d.values, "range": list(d.range) if d.range else None,
                  "track_constant": d.track_constant, "in_scope": 0, "set": 0, "missing": 0, "classes": []}
    for r in rows:
        try:
            applies = onto.attrs_for_class(r["class_id"])
            cname = onto.by_id(r["class_id"]).name
        except KeyError:
            # A class id in the data that the loaded ontology does not know. Counted nowhere rather than
            # attributed to the wrong attribute; the drift itself is a different problem with its own check.
            continue
        for k in keys:
            if applies is not None and k not in applies:
                continue
            miss = int(r["n"]) - int(r[k])
            e = out[k]
            e["in_scope"] += int(r["n"])
            e["set"] += int(r[k])
            e["missing"] += miss
            if miss:
                e["classes"].append({"class_id": r["class_id"], "class_name": cname, "missing": miss})
    for e in out.values():
        e["classes"].sort(key=lambda c: -c["missing"])
        e["classes"] = e["classes"][:12]
    return {"attributes": sorted(out.values(), key=lambda e: -e["missing"]),
            "session_id": str(session_id) if session_id else None}


def _area():
    """Box area in SQL. Postgres arrays are 1-indexed, so bbox is [x1, y1, x2, y2] at 1..4."""
    return ((Object.bbox[3] - Object.bbox[1]) * (Object.bbox[4] - Object.bbox[2])).cast(Float)


async def sweep_queue(db: AsyncSession, onto: Any, *, attr: str, class_name: str | None = None,
                      session_id: UUID | None = None, limit: int = DEFAULT_LIMIT,
                      unit: str = "auto") -> dict:
    """A page of work for one attribute: crops to look at, and what each answer will cover.

    `unit="auto"` follows the attribute: track for one that describes the object, object for one that
    describes the moment. A caller can force either, because a per-frame attribute on a stationary car is
    genuinely constant and an annotator who knows that should not be stopped from saying so.
    """
    if attr not in onto.attributes:
        raise ValueError(f"unknown attribute '{attr}'")
    spec = onto.attributes[attr]
    if spec.derived_from:
        raise ValueError(f"'{attr}' is derived from '{spec.derived_from}' and is never written directly")
    limit = max(1, min(int(limit), MAX_LIMIT))

    if unit == "auto":
        unit = "track" if spec.track_constant else "object"
    if unit not in ("track", "object"):
        raise ValueError(f"unit must be 'track' or 'object', not '{unit}'")

    cids = classes_in_scope(onto, attr)
    if class_name is not None:
        if not onto.has_name(class_name):
            raise ValueError(f"unknown class '{class_name}'")
        cid = onto.by_name(class_name).id
        if cid not in cids:
            return {"attribute": attr, "unit": unit, "items": [], "remaining": 0,
                    "reason": f"'{attr}' does not apply to '{class_name}'"}
        cids = [cid]

    where = [Object.class_id.in_(cids), Object.state.in_(_LIVE_STATES), _missing(attr)]
    if session_id is not None:
        where.append(Frame.session_id == session_id)

    def base(*cols):
        q = select(*cols).select_from(Object).join(Frame, Frame.frame_id == Object.frame_id)
        return q.where(and_(*where))

    remaining = (await db.execute(base(func.count()))).scalar_one()

    cols = (Object.object_id, Object.frame_id, Object.track_id, Object.class_id, Object.bbox,
            Object.state, Object.source, Object.conf, Object.attrs, Frame.session_id, Frame.cam_id)
    if unit == "track":
        # One representative per track, the largest box on it, because a crop you cannot make out is not a
        # question anybody can answer. DISTINCT ON must order by its own key first, so the presentation
        # order is applied in an outer select.
        inner = base(*cols, _area().label("area")).distinct(Object.track_id).where(
            Object.track_id.is_not(None)).order_by(Object.track_id, _area().desc()).subquery()
        q = select(inner).order_by(inner.c.area.desc()).limit(limit)
        rows = (await db.execute(q)).mappings().all()
        tids = [r["track_id"] for r in rows]
        # What each answer will cover: the members still missing the attribute, which is what the apply
        # path will actually write to. Counting all members instead would promise more than it delivers.
        covers: dict[Any, int] = {}
        if tids:
            cq = (select(Object.track_id, func.count())
                  .where(Object.track_id.in_(tids), Object.state.in_(_LIVE_STATES), _missing(attr))
                  .group_by(Object.track_id))
            covers = {t: n for t, n in (await db.execute(cq)).all()}
        # Objects with no track at all cannot be swept track-wise; they are reported so the caller can
        # come back for them as objects rather than being told the queue is empty.
        untracked = (await db.execute(base(func.count()).where(Object.track_id.is_(None)))).scalar_one()
    else:
        # Ordered by area for the same reason, then by id so the page is stable across identical areas.
        q = base(*cols, _area().label("area")).order_by(_area().desc(), Object.object_id).limit(limit)
        rows = (await db.execute(q)).mappings().all()
        covers, untracked = {}, 0

    items = []
    for r in rows:
        tid = r["track_id"]
        items.append({
            "object_id": str(r["object_id"]), "frame_id": str(r["frame_id"]),
            "track_id": str(tid) if tid else None,
            "class_id": r["class_id"], "class_name": onto.by_id(r["class_id"]).name,
            "bbox": [float(v) for v in r["bbox"]], "state": r["state"], "source": r["source"],
            "conf": r["conf"], "session_id": str(r["session_id"]), "cam_id": r["cam_id"],
            # What is already on the object, so a grid can show that a neighbouring attribute is answered
            # and this one is not.
            "attrs": dict(r["attrs"] or {}),
            "covers": int(covers.get(tid, 1)) if unit == "track" else 1,
            "crop_url": f"/api/objects/{r['object_id']}/crop",
        })
    return {"attribute": attr, "type": spec.type, "values": spec.values,
            "range": list(spec.range) if spec.range else None,
            "track_constant": spec.track_constant, "unit": unit,
            "class_name": class_name, "remaining": int(remaining),
            "untracked": int(untracked), "items": items}


async def expand_targets(db: AsyncSession, *, attr: str, unit: str, ids: list[str],
                         overwrite: bool = False) -> list[Object]:
    """The objects an answer lands on.

    For `unit="track"` this is every live member of each track, which is the point: one answer, one value
    across the track, no per-frame disagreement for an exporter to have to resolve.

    `overwrite=False` restricts it to members that do not already carry the attribute, so a sweep fills
    holes and never replaces an answer somebody gave deliberately on one frame. A caller correcting a
    wrong track-wide answer asks for `overwrite=True` and is doing so on purpose.
    """
    if not ids:
        return []
    uuids = [UUID(i) for i in ids]
    q = select(Object).where(Object.state.in_(_LIVE_STATES))
    q = q.where(Object.track_id.in_(uuids)) if unit == "track" else q.where(Object.object_id.in_(uuids))
    if not overwrite:
        q = q.where(_missing(attr))
    return list((await db.execute(q)).scalars().all())
