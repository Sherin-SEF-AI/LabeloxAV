"""Track (tracklet) endpoints for the track editor: view a track across frames, relabel the whole
track at once (the common fix for class flips / ID switches), or delete a junk track. One action here
corrects every frame, which is the point of tracklet-level review."""

from __future__ import annotations

from collections import Counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.timebase import now_ns
from db.models import Frame, Object, Review
from services.api.deps import RelabelTrackIn, current_user, db_session
from services.autolabel.ontology import get_ontology

router = APIRouter()


@router.get("/tracks/{track_id}")
async def get_track(track_id: str, db: AsyncSession = Depends(db_session)):
    onto = get_ontology()
    rows = (await db.execute(
        select(Object, Frame.ts_ns, Frame.frame_id)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Object.track_id == UUID(track_id)).order_by(Frame.ts_ns)
    )).all()
    if not rows:
        raise HTTPException(404, "track not found or has no objects")
    items = [
        {
            "object_id": str(o.object_id), "frame_id": str(fid), "ts_ns": ts,
            "class_id": o.class_id, "class_name": onto.by_id(o.class_id).name,
            "bbox": list(o.bbox), "state": o.state, "conf": o.conf,
            "source": o.source, "is_keyframe": o.is_keyframe, "interp_source": o.interp_source,
            "crop_url": f"/api/objects/{o.object_id}/crop",
        }
        for o, ts, fid in rows
    ]
    classes = Counter(i["class_name"] for i in items)
    return {
        "track_id": track_id, "n_frames": len(items), "classes": dict(classes),
        "dominant": classes.most_common(1)[0][0], "flips": len(classes) > 1, "items": items,
    }


@router.post("/tracks/{track_id}/relabel")
async def relabel_track(track_id: str, payload: RelabelTrackIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Set one class on every object in the track (and confirm them as human). Fixes a class flip in
    one action instead of N per-frame edits."""
    onto = get_ontology()
    if not onto.has_name(payload.class_name):
        raise HTTPException(400, f"unknown class '{payload.class_name}'")
    cid = onto.by_name(payload.class_name).id
    rows = (await db.execute(select(Object).where(Object.track_id == UUID(track_id)))).scalars().all()
    if not rows:
        raise HTTPException(404, "track not found or has no objects")
    for o in rows:
        before = {"class_id": o.class_id, "bbox": list(o.bbox), "attrs": dict(o.attrs or {}), "state": o.state}
        o.class_id = cid
        o.source = "human"
        o.state = payload.state
        db.add(Review(object_id=o.object_id, reviewer=user.name if user else "annotator",
                      user_id=user.user_id if user else None, action="reclassify_track",
                      before=before, after={"class_id": cid, "bbox": list(o.bbox), "attrs": dict(o.attrs or {}), "state": o.state},
                      time_spent_ms=0, ts_ns=now_ns()))
    await db.commit()
    return {"track_id": track_id, "relabeled": len(rows), "class_name": payload.class_name}


@router.delete("/tracks/{track_id}")
async def delete_track(track_id: str, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(Object).where(Object.track_id == UUID(track_id)))).scalars().all()
    if not rows:
        raise HTTPException(404, "track not found or has no objects")
    for o in rows:
        await db.delete(o)
    await db.commit()
    return {"deleted_track": track_id, "n_objects": len(rows)}


@router.post("/tracks/retrack")
async def retrack(session_id: str, db: AsyncSession = Depends(db_session)):
    """Re-run BoT-SORT + DINOv3 tracking over a session: stable track_ids across occlusions/re-entries."""
    from uuid import UUID as _UUID

    from services.autolabel.track.assign import retrack_session

    return await retrack_session(_UUID(session_id))


@router.post("/tracks/{track_id}/interpolate-keyframed")
async def interpolate_keyframed(track_id: str, method: str = "linear", db: AsyncSession = Depends(db_session)):
    """M2.5: fill frames between human keyframes with linear/cubic boxes, marked source=interpolated."""
    from uuid import UUID as _UUID

    from services.temporal.interpolate import interpolate_track_keyframed

    return await interpolate_track_keyframed(_UUID(track_id), method)


@router.post("/objects/{object_id}/keyframe")
async def set_keyframe(object_id: str, value: bool = True, db: AsyncSession = Depends(db_session)):
    from uuid import UUID as _UUID

    from services.temporal.keyframes import mark_keyframe

    return await mark_keyframe(_UUID(object_id), value)


@router.post("/objects/{object_id}/reinterpolate")
async def reinterpolate(object_id: str, method: str = "linear", db: AsyncSession = Depends(db_session)):
    """M2.5 edit-propagation: fixing a keyframe re-interpolates only its adjacent segments."""
    from uuid import UUID as _UUID

    from services.temporal.keyframes import reinterpolate_segment

    return await reinterpolate_segment(_UUID(object_id), method)


@router.post("/tracks/{track_id}/interpolate")
async def interpolate(track_id: str, db: AsyncSession = Depends(db_session)):
    """Fill the gaps between this track's keyframes with linearly-interpolated boxes (no drift)."""
    from services.intelligence.propagate import interpolate_track

    return await interpolate_track(UUID(track_id))
