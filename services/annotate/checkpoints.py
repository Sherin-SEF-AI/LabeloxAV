"""Save a frame's annotations under a name, list them, and get one back.

Undo is not a save. It is capped at a hundred steps, it lives in one browser tab, and it dies with a
refresh. An annotator who works a dense junction for an hour, tries a different reading of it, and wants the
first one back currently has nothing to ask.

Restoring is the part that needs care, because it destroys the present to recover the past. So a restore
takes its own checkpoint of what it is about to replace, marked `auto`, before it touches anything. Undoing a
restore is then another restore rather than an apology, and the guarantee is symmetric: nothing this module
does can lose a state that existed.

Objects are matched by id on the way back in. One still present is updated in place so its version and its
review history survive; one the snapshot has and the frame no longer does is recreated; one the frame has and
the snapshot does not is deleted. Recreating rather than resurrecting means a restored object gets a new id,
which is honest: the row a person deleted is gone and this is a new row that looks like it.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime

from core.logging import get_logger

log = get_logger("checkpoints")

# Fields that make an object what it is. Everything else on the row (timestamps, the id, the version) either
# belongs to the row rather than the annotation or is re-established by the restore itself.
SNAPSHOT_FIELDS = ("object_id", "track_id", "class_id", "bbox", "mask_uri", "mask_encoding",
                   "attrs", "conf", "source", "provenance", "state", "rot_deg", "cuboid_3d",
                   "keypoints", "polyline")


def _to_snapshot(obj) -> dict:
    out: dict = {}
    for f in SNAPSHOT_FIELDS:
        v = getattr(obj, f, None)
        out[f] = str(v) if isinstance(v, _uuid.UUID) else v
    return out


async def create_checkpoint(db, frame_id, *, name: str, note: str | None = None,
                            created_by: str | None = None, auto: bool = False) -> dict:
    """Save the frame's objects exactly as they stand."""
    from sqlalchemy import select

    from db.models import AnnotationCheckpoint, Frame, Object

    fid = frame_id if isinstance(frame_id, _uuid.UUID) else _uuid.UUID(str(frame_id))
    frame = await db.get(Frame, fid)
    if frame is None:
        raise ValueError("frame not found")

    objs = (await db.execute(select(Object).where(Object.frame_id == fid))).scalars().all()
    snapshot = [_to_snapshot(o) for o in objs]

    row = AnnotationCheckpoint(
        frame_id=fid, session_id=frame.session_id, name=name.strip()[:120] or "unnamed",
        note=note, objects=snapshot, object_count=len(snapshot),
        created_by=created_by, auto=auto)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("checkpoint.created", frame=str(fid), name=row.name, objects=len(snapshot), auto=auto)
    return _row(row)


def _row(c, *, with_objects: bool = False) -> dict:
    out = {
        "checkpoint_id": str(c.checkpoint_id), "frame_id": str(c.frame_id),
        "session_id": str(c.session_id) if c.session_id else None,
        "name": c.name, "note": c.note, "object_count": c.object_count,
        "created_by": c.created_by, "auto": c.auto,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }
    if with_objects:
        out["objects"] = c.objects
    return out


async def list_checkpoints(db, frame_id, *, include_auto: bool = True, limit: int = 50) -> dict:
    """A frame's checkpoints, newest first.

    Automatic ones are included by default and flagged rather than hidden. They are the states restores
    replaced, which is exactly what somebody looking for a lost state is after.
    """
    from sqlalchemy import select

    from db.models import AnnotationCheckpoint

    fid = frame_id if isinstance(frame_id, _uuid.UUID) else _uuid.UUID(str(frame_id))
    q = select(AnnotationCheckpoint).where(AnnotationCheckpoint.frame_id == fid)
    if not include_auto:
        q = q.where(AnnotationCheckpoint.auto.is_(False))
    rows = (await db.execute(
        q.order_by(AnnotationCheckpoint.created_at.desc()).limit(limit))).scalars().all()
    return {"frame_id": str(fid), "count": len(rows), "checkpoints": [_row(c) for c in rows]}


async def get_checkpoint(db, checkpoint_id) -> dict | None:
    from db.models import AnnotationCheckpoint

    cid = checkpoint_id if isinstance(checkpoint_id, _uuid.UUID) else _uuid.UUID(str(checkpoint_id))
    c = await db.get(AnnotationCheckpoint, cid)
    return _row(c, with_objects=True) if c else None


async def delete_checkpoint(db, checkpoint_id) -> bool:
    from db.models import AnnotationCheckpoint

    cid = checkpoint_id if isinstance(checkpoint_id, _uuid.UUID) else _uuid.UUID(str(checkpoint_id))
    c = await db.get(AnnotationCheckpoint, cid)
    if c is None:
        return False
    await db.delete(c)
    await db.commit()
    return True


async def restore_checkpoint(db, checkpoint_id, *, created_by: str | None = None) -> dict:
    """Put the frame back to a saved state.

    Takes a checkpoint of the present first. A restore is the one operation here that destroys something, and
    it should not be possible to lose work to the feature whose purpose is not losing work.
    """
    from sqlalchemy import select

    from db.models import AnnotationCheckpoint, Object

    cid = checkpoint_id if isinstance(checkpoint_id, _uuid.UUID) else _uuid.UUID(str(checkpoint_id))
    c = await db.get(AnnotationCheckpoint, cid)
    if c is None:
        raise ValueError("checkpoint not found")

    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    before = await create_checkpoint(
        db, c.frame_id, name=f"before restoring {c.name}"[:120],
        note=f"taken automatically at {stamp} so the replaced state is recoverable",
        created_by=created_by, auto=True)

    current = (await db.execute(select(Object).where(Object.frame_id == c.frame_id))).scalars().all()
    by_id = {str(o.object_id): o for o in current}
    wanted = {str(s.get("object_id")): s for s in (c.objects or []) if s.get("object_id")}

    # track_id is a foreign key, and a track that existed when the checkpoint was taken may have been merged
    # away or deleted since. Restoring the stale reference would fail the whole restore on a detail nobody
    # cares about, so a reference that no longer resolves is dropped and counted rather than fatal.
    from db.models import Track

    referenced = {str(s.get("track_id")) for s in wanted.values() if s.get("track_id")}
    alive: set[str] = set()
    if referenced:
        alive = {str(t) for t in (await db.execute(
            select(Track.track_id).where(
                Track.track_id.in_([_uuid.UUID(t) for t in referenced])))).scalars().all()}
    dropped_tracks = len(referenced - alive)

    updated = created = deleted = 0
    for oid, snap in wanted.items():
        row = by_id.get(oid)
        if row is None:
            # The object was deleted since the checkpoint. It comes back as a new row rather than under its
            # old id: the row a person deleted is gone, and pretending otherwise would resurrect an id that
            # reviews and tracks may already have moved past.
            db.add(Object(frame_id=c.frame_id, **_restorable(snap, alive)))
            created += 1
            continue
        for k, v in _restorable(snap, alive).items():
            setattr(row, k, v)
        row.version = (row.version or 1) + 1
        updated += 1

    for oid, row in by_id.items():
        if oid not in wanted:
            await db.delete(row)
            deleted += 1

    await db.commit()
    log.info("checkpoint.restored", checkpoint=str(cid), frame=str(c.frame_id),
             updated=updated, created=created, deleted=deleted)
    return {"restored": _row(c), "updated": updated, "created": created, "deleted": deleted,
            "dropped_stale_tracks": dropped_tracks,
            # The caller shows this: the state that was replaced is a checkpoint of its own, so a restore
            # taken by mistake is one click from being undone.
            "undo_with": before["checkpoint_id"]}


def _restorable(snap: dict, live_tracks: set[str]) -> dict:
    """The snapshot fields that can be written onto an Object row.

    object_id and track_id are handled apart from the rest: the first identifies the row rather than
    describing it, and the second is a foreign key whose target may since have gone.
    """
    out = {}
    for f in SNAPSHOT_FIELDS:
        if f == "object_id":
            continue
        v = snap.get(f)
        if f == "track_id":
            out[f] = _uuid.UUID(v) if isinstance(v, str) and v in live_tracks else None
            continue
        out[f] = v
    return out
