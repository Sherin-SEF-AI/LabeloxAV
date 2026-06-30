"""Track (tracklet) endpoints for the track editor: view a track across frames, relabel the whole
track at once (the common fix for class flips / ID switches), or delete a junk track. One action here
corrects every frame, which is the point of tracklet-level review."""

from __future__ import annotations

from collections import Counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.timebase import now_ns
from db.models import Frame, Object, Review
from services.api.deps import RelabelTrackIn, current_user, db_session
from services.autolabel.ontology import get_ontology

router = APIRouter()


@router.get("/tracks/{track_id}")
async def get_track(track_id: UUID, db: AsyncSession = Depends(db_session)):
    onto = get_ontology()
    rows = (await db.execute(
        select(Object, Frame.ts_ns, Frame.frame_id)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Object.track_id == track_id).order_by(Frame.ts_ns)
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
        "track_id": str(track_id), "n_frames": len(items), "classes": dict(classes),
        "dominant": classes.most_common(1)[0][0], "flips": len(classes) > 1, "items": items,
    }


@router.post("/tracks/{track_id}/relabel")
async def relabel_track(track_id: UUID, payload: RelabelTrackIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Set one class on every object in the track (and confirm them as human). Fixes a class flip in
    one action instead of N per-frame edits."""
    onto = get_ontology()
    if not onto.has_name(payload.class_name):
        raise HTTPException(400, f"unknown class '{payload.class_name}'")
    cid = onto.by_name(payload.class_name).id
    rows = (await db.execute(select(Object).where(Object.track_id == track_id))).scalars().all()
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
    return {"track_id": str(track_id), "relabeled": len(rows), "class_name": payload.class_name}


class MergeTrackIn(BaseModel):
    from_track_id: UUID
    force: bool = False


class SplitTrackIn(BaseModel):
    at_ts_ns: int


@router.post("/tracks/{track_id}/merge")
async def merge_track_ep(track_id: UUID, body: MergeTrackIn, db: AsyncSession = Depends(db_session),
                         user=Depends(current_user)):
    """Milestone G re-ID: merge a fragmented track (from_track_id) into this one. Refuses with 409 if the two
    tracks share a frame (they coexist, so they are not the same object) unless force is set."""
    from services.temporal.reid import merge_tracks

    res = await merge_tracks(track_id, body.from_track_id, user.name if user else "annotator", force=body.force)
    if res.get("error"):
        raise HTTPException(400, res["error"])
    if res.get("conflict"):
        raise HTTPException(409, res)
    return res


@router.post("/tracks/{track_id}/split")
async def split_track_ep(track_id: UUID, body: SplitTrackIn, db: AsyncSession = Depends(db_session),
                         user=Depends(current_user)):
    """Milestone G re-ID: split a track that collapsed two objects, at a frame timestamp. Objects at or after
    the boundary move to a new track; objects before stay."""
    from services.temporal.reid import split_track

    res = await split_track(track_id, body.at_ts_ns, user.name if user else "annotator")
    if res.get("error"):
        raise HTTPException(400, res["error"])
    return res


@router.delete("/tracks/{track_id}")
async def delete_track(track_id: UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(Object).where(Object.track_id == track_id))).scalars().all()
    if not rows:
        raise HTTPException(404, "track not found or has no objects")
    for o in rows:
        await db.delete(o)
    await db.commit()
    return {"deleted_track": str(track_id), "n_objects": len(rows)}


@router.post("/tracks/retrack")
async def retrack(session_id: str, db: AsyncSession = Depends(db_session)):
    """Re-run BoT-SORT + DINOv3 tracking over a session: stable track_ids across occlusions/re-entries."""
    from uuid import UUID as _UUID

    from services.autolabel.track.assign import retrack_session

    return await retrack_session(_UUID(session_id))


@router.post("/tracks/{track_id}/interpolate-keyframed")
async def interpolate_keyframed(track_id: UUID, method: str = "linear", db: AsyncSession = Depends(db_session)):
    """M2.5: fill frames between human keyframes with linear/cubic boxes, marked source=interpolated."""
    from services.temporal.interpolate import interpolate_track_keyframed

    return await interpolate_track_keyframed(track_id, method)


@router.post("/tracks/{track_id}/smooth")
async def smooth_track_path(track_id: UUID, window: int = 5, db: AsyncSession = Depends(db_session)):
    """M-4D.2: smooth the track's motion path, shifting each box to its low-pass-filtered centroid (jitter
    and velocity discontinuities removed) while keeping box sizes and the true endpoints."""
    from services.temporal.trajectory import smooth_track

    return await smooth_track(track_id, window)


@router.get("/tracks/{track_id}/attribute-timeline")
async def attribute_timeline(track_id: UUID, key: str, db: AsyncSession = Depends(db_session)):
    """M-4D / Milestone G: the transition timeline of one attribute across a track (e.g. signal_state,
    brake, indicator), as contiguous value segments so the change points are explicit."""
    from services.temporal.attributes import track_attribute_timeline

    return await track_attribute_timeline(track_id, key)


@router.get("/tracks/{track_id}/seg4d-consistency")
async def seg4d_consistency(track_id: UUID, window: int = 2, db: AsyncSession = Depends(db_session)):
    """Milestone G 4D semantic seg: temporal consistency of the track's per-frame class, with the count of
    isolated flickers a temporal majority filter would correct (proposed, not auto-applied)."""
    from services.temporal.seg4d import track_class_consistency

    return await track_class_consistency(track_id, window)


@router.post("/objects/{object_id}/keyframe")
async def set_keyframe(object_id: UUID, value: bool = True, db: AsyncSession = Depends(db_session)):
    from services.temporal.keyframes import mark_keyframe

    return await mark_keyframe(object_id, value)


@router.post("/objects/{object_id}/reinterpolate")
async def reinterpolate(object_id: UUID, method: str = "linear", db: AsyncSession = Depends(db_session)):
    """M2.5 edit-propagation: fixing a keyframe re-interpolates only its adjacent segments."""
    from services.temporal.keyframes import reinterpolate_segment

    return await reinterpolate_segment(object_id, method)


@router.post("/tracks/{track_id}/interpolate")
async def interpolate(track_id: UUID, db: AsyncSession = Depends(db_session)):
    """Fill the gaps between this track's keyframes with linearly-interpolated boxes (no drift)."""
    from services.intelligence.propagate import interpolate_track

    return await interpolate_track(track_id)
