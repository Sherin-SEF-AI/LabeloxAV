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

from db.models import Frame, Object, Track
from services.api.deps import MAX_TRACK_RELABEL_OBJECTS, RelabelTrackIn, current_user, db_session
from services.autolabel.ontology import get_ontology
from services.review_apply import apply_review_batch
from services.review_batch import record_batch
from services.review_policy import ReviewStateError

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
    track = await db.get(Track, track_id)
    return {
        "track_id": str(track_id), "n_frames": len(items), "classes": dict(classes),
        "dominant": classes.most_common(1)[0][0], "flips": len(classes) > 1, "items": items,
        "intents": (track.intents if track else []) or [],  # M-F.2 track-level intents
    }


# M-F.2 behavior/intent annotation


@router.get("/intent/vocab")
async def intent_vocab():
    """The governed closed intent vocabularies (VRU + vehicle) and which each assist path may propose."""
    from services.intelligence.intent import vocab

    return vocab()


@router.post("/tracks/{track_id}/intent/propose")
async def intent_propose(track_id: UUID):
    """Propose trajectory-derived intents for a track (geometric: cut_in, hard_brake, crossing, ...)."""
    from services.intelligence.intent import propose_track

    res = await propose_track(track_id)
    if "error" in res:
        raise HTTPException(404, res["error"])
    return res


@router.post("/tracks/{track_id}/intent/vlm")
async def intent_vlm(track_id: UUID):
    """Propose a VLM-derived intent for a VRU track (looking_at_vehicle, hesitating, jaywalking)."""
    from services.intelligence.intent import propose_vlm

    res = await propose_vlm(track_id)
    if "error" in res:
        raise HTTPException(404, res["error"])
    return res


@router.post("/intent/propose-session")
async def intent_propose_session(session_id: UUID):
    """Trajectory-derived intent proposals across a whole session."""
    from services.intelligence.intent import propose_session

    return await propose_session(session_id)


class IntentSetIn(BaseModel):
    intent: str
    kind: str
    status: str = "confirmed"


@router.post("/tracks/{track_id}/intent/set", dependencies=[Depends(current_user)])
async def intent_set(track_id: UUID, body: IntentSetIn):
    """Human confirms or sets a track's intent from the closed vocabulary (or 'unknown' to clear)."""
    from services.intelligence.intent import set_intent

    res = await set_intent(track_id, body.intent, body.kind, body.status)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.post("/tracks/{track_id}/relabel")
async def relabel_track(track_id: UUID, payload: RelabelTrackIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Set one class on every object in the track. Fixes a class flip in one action instead of N per-frame
    edits.

    This is what a class correction in the frame editor now fans out through, and the reason it had to be
    hardened first. Measured before the change: 413 tracks carried an unambiguous human class, and of the
    44,097 objects on them only 5,798 had it. The median track is 93 frames and the median number of frames
    a person actually touched is 1, so 86.9% of every correction ever made was sitting on one frame.

    The per-object work is services/review_apply.py, shared with bulk review, which is what gives this path
    the optimistic lock, the role clamp, attribute revalidation and a revertible run it previously had none
    of.
    """
    onto = get_ontology()
    if not onto.has_name(payload.class_name):
        raise HTTPException(400, f"unknown class '{payload.class_name}'")
    cid = onto.by_name(payload.class_name).id
    rows = (await db.execute(select(Object).where(Object.track_id == track_id))).scalars().all()
    if not rows:
        raise HTTPException(404, "track not found or has no objects")
    if len(rows) > MAX_TRACK_RELABEL_OBJECTS:
        raise HTTPException(409, f"track holds {len(rows)} objects, above the {MAX_TRACK_RELABEL_OBJECTS} "
                                 "limit; it is more likely mis-linked than real, so fix the track first")

    # The frame the human edited is already saved by the editor's own review call, with source=human and
    # its own state. Touching it again here would bump the version the editor is holding and make its next
    # save 409 against its own propagation.
    origin = str(payload.origin_object_id) if payload.origin_object_id else None
    targets = [o for o in rows if str(o.object_id) != origin]

    reviewer = user.name if user is not None else payload.reviewer
    uid = user.user_id if user is not None else None
    try:
        res = await apply_review_batch(
            db, targets, action="reclassify_track", onto=onto, class_id=cid,
            requested_state=payload.state, role=getattr(user, "role", None),
            # NOT "human". These frames were never looked at; 92 of 93 claiming human authorship would make
            # every consumer that filters on it read machine output as ground truth, and would make the rows
            # un-self-healable, since source == "human" is this repo's "an agent must not touch this" flag.
            source="propagated",
            reviewer=reviewer, uid=uid, time_spent_ms=payload.time_spent_ms,
            provenance_extra={"track_relabel": True, "propagated_from": origin} if origin else {"track_relabel": True},
            skip_human=True, guard_class_move=not payload.force, revalidate_attrs=True)
    except ReviewStateError as exc:
        raise HTTPException(400, str(exc)) from exc

    if res.refused and not res.n:
        # Nothing was written, so say why rather than reporting a successful relabel of zero objects.
        raise HTTPException(409, {"refused": res.refused[0]["reason"], "objects": len(res.refused),
                                  "hint": "a reviewer can override with force"})

    # The track's own class, which nothing kept in sync. services/intelligence/propagate.py fills
    # interpolated gaps from it, so leaving it stale re-injects the pre-relabel class into new boxes, and
    # services/temporal/tracklet.py names the class from it while GET /tracks/{id} derives it from the
    # objects, so the two views disagreed. Measured: 2,019 tracks had already drifted.
    track = await db.get(Track, track_id)
    track_class_from = int(track.class_id) if track is not None else None
    if track is not None:
        track.class_id = cid

    run_id = await record_batch(
        db, res.changes, created_by=reviewer, commit=False,
        policy={"action": "reclassify_track", "class_name": payload.class_name,
                "track_id": str(track_id), "track_class_from": track_class_from,
                "track_class_to": cid}) if res.changes else None
    await db.commit()

    # Surfaced, not blocked. 9,139 tracks carry a re-identification event, which is where the tracker lost
    # the object and picked it up again, and so where a propagated label could have jumped to a different
    # physical object. The caller is told the count and given an undo rather than being stopped, because
    # the alternative refuses the fix on most of the corpus.
    switches = 0
    if track is not None and isinstance(track.id_switch_flags, dict):
        switches = len(track.id_switch_flags.get("events") or [])

    return {"track_id": str(track_id), "relabeled": res.n, "class_name": payload.class_name,
            "state": res.new_state, "clamped": res.clamped, "run_id": run_id,
            "skipped_stale": res.stale, "skipped_human": res.skipped_human,
            "refused": res.refused, "attrs_dropped": res.attrs_dropped,
            "id_switch_events": switches}


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
async def interpolate(track_id: UUID, method: str = "cubic", anchor_policy: str = "detection",
                      db: AsyncSession = Depends(db_session)):
    """Fill this track's gaps, skipping holes whose bracketing anchors are not plausibly one object.

    Anchors default to detections rather than human keyframes: only 179 of 11,406 tracks carry two
    human-verified boxes, so the keyframe policy fills nothing on a machine-labelled track. The response
    reports refusals by reason, which is the part worth reading - a track that fills nothing because its
    endpoints teleport is a tracking problem, not an interpolation one.
    """
    from services.temporal.interpolate import interpolate_track_keyframed

    return await interpolate_track_keyframed(track_id, method, anchor_policy=anchor_policy)
