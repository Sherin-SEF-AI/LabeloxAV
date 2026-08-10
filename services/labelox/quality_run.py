"""Writing the label-quality table that nothing has ever written.

`annotation_quality` is read by the data lake export and surfaced in the workspace as "which labels to
trust". It holds zero rows, and has since the migration created it. The scoring lives in quality.py and is
pure and tested; there was simply never a caller to persist what it computes.

That is worth more than a missing feature. Reviewers were shown a trust signal that is uniformly absent, and
an absent signal reads as "nothing wrong here" rather than as "nobody looked". It is the same failure mode as
the analytics page reporting an empty corpus and the benchmarks naming artifacts that do not exist: silence
presented as an all-clear.

Scoring the whole table is a scan of 570,505 objects, so this runs in batches over a cursor and is safe to
stop and resume. It is idempotent by object id: a rerun after a relabel overwrites the score rather than
accumulating a second opinion, which matters because objects here are relabelled often.

Agreement is only supplied where more than one annotator actually touched an object. On this corpus that is
zero objects: 557 reviews exist and not one object has been seen by two different reviewers. So agreement is
null everywhere, and the run reports that count rather than letting it pass unremarked, because
`annotation_quality` treats a missing agreement as neutral and a reader would otherwise never learn the term
had contributed nothing. Passing 1.0 instead would score "one person said so, unchallenged" identically to
"four people independently agreed", which is the flattering reading of thin evidence this codebase keeps
having to remove.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AnnotationQuality, Frame, Object, Review
from services.labelox.quality import annotation_flags, annotation_quality

log = get_logger("labelox.quality_run")

BATCH = 2000


# Review actions that leave the label as the reviewer found it. Everything else (reclassify, reclassify_track,
# adjust_geometry) is a correction, which is a disagreement with whoever labelled it first.
_CONFIRMING_ACTIONS = {"confirm", "accept", "approve"}


async def _agreement_by_object(db: AsyncSession, object_ids: list[UUID]) -> dict[UUID, float]:
    """Agreement across reviewers, for objects that more than one person actually judged.

    The Review trail is what this system has instead of parallel annotation passes: two reviewers who touch
    the same object and disagree about what to do with it are the closest thing to two annotators
    disagreeing. A single review is not agreement, so those objects get no agreement term at all rather than
    a default of 1.0.

    On the current corpus this returns nothing for every batch, because no object has been reviewed by two
    different people. That is a fact about the corpus and not a bug here, and `score_corpus` reports it.
    """
    if not object_ids:
        return {}
    rows = (await db.execute(
        select(Review.object_id, Review.reviewer, Review.action)
        .where(Review.object_id.in_(object_ids)))).all()
    by_obj: dict[UUID, dict[str, str]] = defaultdict(dict)
    for oid, reviewer, action in rows:
        if reviewer:
            by_obj[oid][str(reviewer)] = str(action or "")
    out: dict[UUID, float] = {}
    for oid, per_reviewer in by_obj.items():
        if len(per_reviewer) < 2:
            continue
        acts = list(per_reviewer.values())
        confirming = sum(1 for a in acts if a in _CONFIRMING_ACTIONS)
        out[oid] = round(max(confirming, len(acts) - confirming) / len(acts), 4)
    return out


async def score_batch(db: AsyncSession, rows: Sequence[Any]) -> int:
    """Score and upsert one batch of (Object, frame width, frame height)."""
    if not rows:
        return 0
    ids = [o.object_id for o, _w, _h in rows]
    agree = await _agreement_by_object(db, ids)

    payload = []
    for obj, w, h in rows:
        # The scorer's own defaults are 1920x1080; passing the frame's real size is what makes off_screen
        # mean anything on a corpus that is not all one resolution.
        as_dict = {"bbox": list(obj.bbox) if obj.bbox else None, "class_id": obj.class_id,
                   "source": obj.source, "conf": obj.conf}
        flags = annotation_flags(as_dict, img_w=float(w or 1920), img_h=float(h or 1080))
        scored = annotation_quality(as_dict, agreement=agree.get(obj.object_id))
        payload.append({"object_id": obj.object_id, "quality": scored["quality"],
                        "agreement": scored["agreement"], "flags": flags, "audit_verdict": None})

    stmt = insert(AnnotationQuality).values(payload)
    # Idempotent by object: a rerun after a relabel replaces the verdict rather than leaving a stale one
    # beside it. There is exactly one current quality for an object, by definition.
    await db.execute(stmt.on_conflict_do_update(
        index_elements=[AnnotationQuality.object_id],
        set_={"quality": stmt.excluded.quality, "agreement": stmt.excluded.agreement,
              "flags": stmt.excluded.flags}))
    await db.commit()
    return len(payload)


async def score_corpus(db: AsyncSession, *, limit: int | None = None, batch: int = BATCH,
                       only_missing: bool = False) -> dict:
    """Score every object into `annotation_quality`, in resumable batches.

    `only_missing` skips objects already scored, which is what makes a second run after an ingest cheap
    instead of a full rescan.
    """
    q = (select(Object, Frame.width, Frame.height)
         .join(Frame, Frame.frame_id == Object.frame_id)
         .order_by(Object.object_id))
    if only_missing:
        q = q.outerjoin(AnnotationQuality, AnnotationQuality.object_id == Object.object_id).where(
            AnnotationQuality.object_id.is_(None))

    total = int((await db.execute(
        select(func.count()).select_from(q.subquery()))).scalar() or 0)
    target = min(total, limit) if limit else total

    # Keyset pagination, not OFFSET. Under only_missing the predicate is "has no quality row yet", and this
    # loop writes exactly those rows, so the result set shrinks underneath a growing offset and the walk
    # skips whatever slid past the cursor. The first full run stopped at 286,000 of 570,505 that way and
    # reported success. Advancing by the last object id read is stable whether or not the predicate moves,
    # and it is faster on a table this size besides.
    done = 0
    last_id: UUID | None = None
    while done < target:
        page = q if last_id is None else q.where(Object.object_id > last_id)
        rows = (await db.execute(page.limit(min(batch, target - done)))).all()
        if not rows:
            break
        last_id = rows[-1][0].object_id
        done += await score_batch(db, rows)
        log.info("quality.scored", done=done, target=target)

    scored = (await db.execute(select(func.count()).select_from(AnnotationQuality))).scalar() or 0
    flagged = (await db.execute(
        select(func.count()).select_from(AnnotationQuality)
        .where(func.jsonb_array_length(AnnotationQuality.flags) > 0))).scalar() or 0
    with_agreement = (await db.execute(
        select(func.count()).select_from(AnnotationQuality)
        .where(AnnotationQuality.agreement.isnot(None)))).scalar() or 0
    return {"considered": target, "scored_this_run": done, "rows_total": int(scored),
            "rows_with_flags": int(flagged),
            # Surfaced rather than left to be discovered. Zero here means the agreement term contributed
            # nothing to any score, and a quality number nobody knows is single-sourced is a number that
            # will be over-trusted.
            "rows_with_agreement": int(with_agreement),
            "detail": (f"{int(with_agreement)} of {int(scored)} objects were judged by more than one "
                       "reviewer, so agreement is unmeasured for the rest and their quality rests on "
                       "geometry and confidence alone")}
