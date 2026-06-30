"""Milestone G: split / merge track-identity correction. Two re-identification errors the tracker makes:
a single physical object fragmented across two track_ids (an ID switch through an occlusion) is fixed by
MERGING, and two physical objects collapsed onto one track_id is fixed by SPLITTING the track at a frame.
Every reassigned object writes a Review row, so an identity edit is auditable and reversible like any other
review action. The pure helpers (overlap, partition, span) are separated from the DB writes so the routing
logic is tested without infra.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("reid")


def tracks_overlap(ts_a, ts_b) -> bool:
    """Two tracks of the same physical object must be temporally disjoint. A shared frame timestamp means
    both tracks have a box on the same frame, so they coexist and are NOT the same object: a merge should be
    refused unless forced."""
    return bool(set(ts_a) & set(ts_b))


def partition_at(items: list, at_ts_ns: int) -> tuple[list, list]:
    """Split objects [{..., ts_ns}] at a boundary into (before < at, at and after >= at)."""
    before = [o for o in items if o["ts_ns"] < at_ts_ns]
    after = [o for o in items if o["ts_ns"] >= at_ts_ns]
    return before, after


def span_union(a: tuple, b: tuple) -> tuple:
    """The time span covering both tracks: (min first, max last)."""
    return (min(a[0], b[0]), max(a[1], b[1]))


async def merge_tracks(into_id, from_id, user_name: str = "annotator", *, force: bool = False) -> dict:
    """Merge from_id into into_id: every object on from_id is reassigned to into_id and from_id is removed.
    Refuses if the two tracks share a frame (they coexist, so they are not the same object) unless force."""
    from sqlalchemy import select

    from core.timebase import now_ns
    from db.models import Frame, Object, Review, Track
    from db.session import get_sessionmaker
    if into_id == from_id:
        return {"error": "cannot merge a track into itself"}
    async with get_sessionmaker()() as db:
        into = await db.get(Track, into_id)
        frm = await db.get(Track, from_id)
        if into is None or frm is None:
            return {"error": "track not found"}
        if into.session_id != frm.session_id:
            return {"error": "tracks are in different sessions"}

        async def _ts(tid):
            return {int(t) for t in (await db.execute(
                select(Frame.ts_ns).join(Object, Object.frame_id == Frame.frame_id)
                .where(Object.track_id == tid))).scalars().all()}
        if not force and tracks_overlap(await _ts(into_id), await _ts(from_id)):
            return {"conflict": True, "reason": "tracks overlap in time (likely not the same object)"}

        objs = (await db.execute(select(Object).where(Object.track_id == from_id))).scalars().all()
        for o in objs:
            db.add(Review(object_id=o.object_id, reviewer=user_name, action="merge_track",
                          before={"track_id": str(from_id)}, after={"track_id": str(into_id)}, ts_ns=now_ns()))
            o.track_id = into_id
        into.first_ts_ns, into.last_ts_ns = span_union(
            (into.first_ts_ns, into.last_ts_ns), (frm.first_ts_ns, frm.last_ts_ns))
        await db.delete(frm)
        await db.commit()
        moved = len(objs)
    log.info("reid.merge", into=str(into_id), removed=str(from_id), moved=moved)
    return {"merged": str(from_id), "into": str(into_id), "moved": moved}


async def split_track(track_id, at_ts_ns: int, user_name: str = "annotator") -> dict:
    """Split track_id at at_ts_ns: objects at or after the boundary move to a new track (same class/session);
    objects before stay. Refuses a boundary that leaves either side empty."""
    from sqlalchemy import select

    from core.timebase import now_ns
    from db.models import Frame, Object, Review, Track
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        track = await db.get(Track, track_id)
        if track is None:
            return {"error": "track not found"}
        rows = (await db.execute(
            select(Object, Frame.ts_ns).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Object.track_id == track_id).order_by(Frame.ts_ns))).all()
        before = [(o, int(ts)) for o, ts in rows if int(ts) < at_ts_ns]
        after = [(o, int(ts)) for o, ts in rows if int(ts) >= at_ts_ns]
        if not before or not after:
            return {"error": "split point leaves one side empty; choose a timestamp inside the track"}
        new_track = Track(session_id=track.session_id, class_id=track.class_id,
                          first_ts_ns=after[0][1], last_ts_ns=after[-1][1], trajectory=None,
                          tracker_version=track.tracker_version)
        db.add(new_track)
        await db.flush()
        new_id = new_track.track_id
        for o, _ in after:
            db.add(Review(object_id=o.object_id, reviewer=user_name, action="split_track",
                          before={"track_id": str(track_id)}, after={"track_id": str(new_id)}, ts_ns=now_ns()))
            o.track_id = new_id
        track.last_ts_ns = before[-1][1]
        await db.commit()
        new_id_s = str(new_id)
    log.info("reid.split", original=str(track_id), new=new_id_s, moved=len(after), kept=len(before))
    return {"original": str(track_id), "new_track": new_id_s, "moved": len(after), "kept": len(before)}
