"""Persisting per-frame rarity onto `Frame.scene`, so it can be filtered and exported.

Rarity already exists: `services/intelligence/search/rarity.py` computes it to rerank a search, from an
inverse-document-frequency over the classes a frame carries. But it computes it transiently, behind a
five-minute cache, inside one request. Nothing can select on it, no export can slice by it, and the coverage
datasheet cannot report the corpus's rarity distribution, because the number never lands anywhere.

This writes the same number - the same function, not a second implementation of the idea - into
`Frame.scene["rarity"]`, alongside the ingest classifier's axes. A second column was the alternative and it
would have meant a migration, a backfill of 41,752 rows, and a value that could disagree with the one search
uses. `scene` already has a GIN index, from 0062.

The write merges. `scene` carries weather, density, road_type and time_of_day from ingest, and a rarity pass
that replaced the object would silently delete all of it.

Rarity moves as labelling moves, so this is a sweep rather than a one-off backfill: re-running recomputes,
and `stale_after_s` skips frames scored recently enough that nothing can have changed.
"""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object
from services.intelligence.search.rarity import class_frame_counts, frame_rarity

log = get_logger("context.rarity")

KEY = "rarity"
# Written alongside the value so a consumer can tell a fresh score from one computed against a corpus that
# has since been labelled. `scene` has no per-key timestamp otherwise.
AT_KEY = "rarity_at_ns"


async def sweep_rarity(db: AsyncSession, *, limit: int = 2000, session_id=None,
                       force: bool = False, commit: bool = True) -> dict:
    """Score up to `limit` frames and merge the result into `Frame.scene`.

    Batched deliberately. This runs off-hours beside annotators, and a single statement over 41,752 frames
    holds locks on the table the editor writes to.
    """
    from core.timebase import now_ns

    idf_map, total = await class_frame_counts(db)
    if not total:
        return {"scored": 0, "skipped": "no objects in the corpus, so idf is undefined"}

    q = select(Frame.frame_id, Frame.scene).order_by(Frame.ts_ns)
    if session_id is not None:
        q = q.where(Frame.session_id == session_id)
    if not force:
        # Frames that have never been scored; re-scoring is what `force` is for.
        #
        # The NULL arm is not defensive padding. `~Frame.scene.has_key(KEY)` is NULL, not true, when `scene`
        # is NULL, so a sweep written without it silently skipped the 39,972 frames that have no scene at
        # all - which is 96% of the corpus - and then reported `remaining: 0`. It looked finished.
        q = q.where(or_(Frame.scene.is_(None), ~Frame.scene.has_key(KEY)))
    rows = (await db.execute(q.limit(limit))).all()
    if not rows:
        return {"scored": 0, "remaining": 0}

    ids = [r[0] for r in rows]
    per_frame: dict = {}
    for fid, cid in (await db.execute(
            select(Object.frame_id, Object.class_id).where(Object.frame_id.in_(ids)))).all():
        per_frame.setdefault(fid, set()).add(cid)

    ts = now_ns()
    scored = 0
    for fid, scene in rows:
        value = frame_rarity(per_frame.get(fid, set()), idf_map)
        # Reassigned, not mutated. SQLAlchemy does not flag an in-place JSONB edit as dirty, and a sweep
        # that mutated the dict in place would report thousands of frames scored and write none of them.
        merged = dict(scene or {})
        merged[KEY] = value
        merged[AT_KEY] = ts
        await db.execute(Frame.__table__.update().where(Frame.frame_id == fid).values(scene=merged))
        scored += 1

    if commit:
        await db.commit()
    remaining = 0 if force else int((await db.execute(
        select(func.count(Frame.frame_id))
        .where(or_(Frame.scene.is_(None), ~Frame.scene.has_key(KEY))))).scalar() or 0)
    log.info("rarity.swept", scored=scored, remaining=remaining, classes=len(idf_map))
    return {"scored": scored, "remaining": remaining, "classes_in_idf": len(idf_map),
            "frames_in_idf": total}
