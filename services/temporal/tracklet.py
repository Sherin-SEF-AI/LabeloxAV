"""Tracklet-first editing: draw once, correct where it drifts, and let the rest follow.

The editor is frame-first. An annotator draws a box, moves to the next frame, draws it again, and again,
which for a vehicle crossing a junction over sixty frames is sixty boxes describing one object. Every piece
needed to stop doing that already existed and none of them were joined: SAM propagation forwards, optical
flow, keyframe interpolation, track-level attributes, the filmstrip.

The model here is the one every video annotator converges on, because it matches how the error behaves:

- **A track is defined by its keyframes.** Everything between them is derived, and derived geometry is
  marked as such so nobody mistakes an interpolation for an observation.
- **Correcting a frame makes it a keyframe.** That is the whole interaction: scrub until the box is wrong,
  fix it, and the segments either side re-derive from the corrected point.
- **Propagation runs in both directions.** An object is usually noticed mid-track, not at its first frame,
  and a forward-only propagator makes the annotator scrub backwards and start again.

The costly thing is `derive`, which rewrites the non-keyframe boxes. It is idempotent and reports what it
changed, so it can be run after every correction without the editor having to reason about staleness.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("tracklet")

# Interpolation methods, and when each is right. Linear is correct for a box moving steadily and wrong
# through a turn; a cubic spline follows the turn and overshoots when keyframes are sparse, which looks
# like the box leading the object.
METHODS = ("linear", "cubic")


class TrackletError(Exception):
    """A tracklet operation refused."""


@dataclass
class Sample:
    frame_id: str
    object_id: str
    ts_ns: int
    bbox: list[float]
    is_keyframe: bool
    source: str
    state: str


async def load_tracklet(db: AsyncSession, track_id: str) -> dict:
    """One track as an ordered timeline, with its keyframes marked."""
    from db.models import Frame, Object, Track
    from services.autolabel.ontology import get_ontology

    track = await db.get(Track, uuid.UUID(track_id))
    if track is None:
        raise TrackletError(f"track {track_id} not found")

    rows = (await db.execute(
        select(Object, Frame.ts_ns, Frame.frame_id)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Object.track_id == track.track_id)
        .order_by(Frame.ts_ns))).all()

    onto = get_ontology()
    samples = [Sample(frame_id=str(fid), object_id=str(o.object_id), ts_ns=int(ts),
                      bbox=[float(v) for v in o.bbox], is_keyframe=bool(o.is_keyframe),
                      source=o.source, state=o.state)
               for o, ts, fid in rows]
    keys = [s for s in samples if s.is_keyframe]
    return {
        "track_id": track_id,
        "class_id": track.class_id,
        "class_name": onto.by_id(track.class_id).name,
        "first_ts_ns": int(track.first_ts_ns), "last_ts_ns": int(track.last_ts_ns),
        "length": len(samples), "keyframes": len(keys),
        # The number an annotator is actually optimising: how many frames one correction covers.
        "frames_per_keyframe": round(len(samples) / len(keys), 2) if keys else None,
        "samples": [s.__dict__ for s in samples],
        "intents": list(track.intents or []),
    }


async def set_keyframe(db: AsyncSession, object_id: str, *, bbox: list[float] | None = None,
                       is_keyframe: bool = True, user_name: str | None = None) -> dict:
    """Mark a frame as observed, optionally correcting its box.

    The core interaction. Correcting a frame necessarily makes it a keyframe: a corrected box that stayed
    derived would be overwritten by the next derive, which is the single most infuriating thing a video
    annotator can experience.
    """
    from db.models import Object

    obj = await db.get(Object, uuid.UUID(object_id))
    if obj is None:
        raise TrackletError("object not found")
    if obj.track_id is None:
        raise TrackletError("that object is not on a track, so it has nothing to be a keyframe of")

    if bbox is not None:
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise TrackletError("bbox must be [x1, y1, x2, y2] with positive width and height")
        obj.bbox = [float(v) for v in bbox]
        obj.source = "human"
        obj.state = "accepted"
        # A corrected frame is an observation, whatever the caller passed.
        is_keyframe = True
        obj.version = int(obj.version or 1) + 1

    obj.is_keyframe = bool(is_keyframe)
    obj.interp_source = None if is_keyframe else obj.interp_source
    await db.commit()

    from services.activity import record_activity

    await record_activity(db, user_name=user_name, verb="reviewed", subject_type="object",
                          subject_id=object_id, summary="set a tracklet keyframe",
                          href=f"/frame/{obj.frame_id}")
    return {"object_id": object_id, "track_id": str(obj.track_id),
            "is_keyframe": obj.is_keyframe, "bbox": list(obj.bbox)}


async def derive(db: AsyncSession, track_id: str, *, method: str = "linear",
                 overwrite_human: bool = False) -> dict:
    """Rewrite every non-keyframe box from the keyframes around it.

    Idempotent, so the editor can call it after every correction without tracking staleness. Human-edited
    frames are never overwritten unless asked: a box somebody drew is an observation even when they forgot
    to mark it, and silently replacing it would discard work.
    """
    if method not in METHODS:
        raise TrackletError(f"method must be one of {METHODS}")

    from db.models import Frame, Object

    rows = (await db.execute(
        select(Object, Frame.ts_ns)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Object.track_id == uuid.UUID(track_id))
        .order_by(Frame.ts_ns))).all()
    if not rows:
        raise TrackletError("that track has no objects")

    keys = [(int(ts), np.asarray(o.bbox, dtype=np.float64)) for o, ts in rows if o.is_keyframe]
    if len(keys) < 2:
        # One keyframe defines a position, not a trajectory. Refusing is better than propagating a
        # constant box down the whole track and calling it interpolation.
        return {"track_id": track_id, "updated": 0, "keyframes": len(keys),
                "detail": "a track needs at least two keyframes before anything can be derived"}

    kf_ts = [k[0] for k in keys]
    kf_box = np.stack([k[1] for k in keys])

    from services.temporal.interpolate import build_box_interpolator

    # Returns (box_at, actual_method): the builder falls back to linear when a cubic is not possible
    # (fewer than three keyframes, or no SciPy), and reporting the method it really used matters, because
    # a caller that asked for cubic and silently got linear would attribute the difference to the data.
    interp, actual_method = build_box_interpolator(kf_ts, kf_box, method)

    updated = skipped_human = outside = 0
    for obj, ts in rows:
        if obj.is_keyframe:
            continue
        if obj.source == "human" and not overwrite_human:
            skipped_human += 1
            continue
        if int(ts) < kf_ts[0] or int(ts) > kf_ts[-1]:
            # Outside the keyframed span. Extrapolation past the last observation is where an interpolator
            # invents a box in the middle of nothing, so it is left alone.
            outside += 1
            continue
        box = interp(float(ts))
        obj.bbox = [float(v) for v in box]
        obj.source = "interpolated"
        obj.interp_source = actual_method
        updated += 1

    await db.commit()
    log.info("tracklet.derived", track=track_id[:8], updated=updated, keyframes=len(keys),
             method=actual_method)
    return {"track_id": track_id, "method": actual_method, "method_requested": method,
            "keyframes": len(keys),
            "updated": updated,
            # Both reported, because a derive that updated nothing is either correct or a sign the track is
            # entirely human-edited, and the two look identical from the count alone.
            "skipped_human": skipped_human, "outside_keyframe_span": outside}


async def propagate(db: AsyncSession, object_id: str, *, direction: str = "both",
                    frames: int = 12, refine: bool = True) -> dict:
    """Carry one box outwards along its track, forwards, backwards, or both.

    Both by default. An object is noticed mid-track far more often than at its first frame, and a
    forward-only propagator makes the annotator scrub back and start again, which is the interaction this
    is meant to remove.
    """
    if direction not in ("forward", "backward", "both"):
        raise TrackletError("direction must be forward, backward or both")

    from services.temporal.sam_propagate import sam_propagate_object

    out: dict = {"object_id": object_id, "direction": direction}
    total = 0
    for step in (["forward", "backward"] if direction == "both" else [direction]):
        try:
            res = await sam_propagate_object(uuid.UUID(object_id), frames, step, refine)
            out[step] = res
            total += int((res or {}).get("created") or (res or {}).get("propagated") or 0)
        except Exception as exc:  # noqa: BLE001
            # One direction failing must not lose the other: propagating backwards from the first frame of
            # a session has nowhere to go, and that is ordinary rather than an error.
            out[step] = {"error": f"{type(exc).__name__}: {exc}"}
    out["created"] = total
    return out


async def set_track_attributes(db: AsyncSession, track_id: str, attrs: dict, *,
                               user_name: str | None = None) -> dict:
    """Set an attribute once for a whole track rather than once per frame.

    A vehicle's colour does not change between frames. Setting it per frame is sixty edits that can
    disagree with each other, and a disagreement in a track-constant attribute is a data defect the
    exporter has no way to resolve.
    """
    from db.models import Object, Track

    track = await db.get(Track, uuid.UUID(track_id))
    if track is None:
        raise TrackletError("track not found")

    objs = (await db.execute(
        select(Object).where(Object.track_id == track.track_id))).scalars().all()
    for obj in objs:
        obj.attrs = {**(obj.attrs or {}), **attrs}
    await db.commit()

    from services.activity import record_activity

    await record_activity(db, user_name=user_name, verb="reviewed", subject_type="track",
                          subject_id=track_id,
                          summary=f"set {', '.join(attrs)} on {len(objs)} frames",
                          href=f"/track/{track_id}")
    log.info("tracklet.attributes_set", track=track_id[:8], objects=len(objs), keys=list(attrs))
    return {"track_id": track_id, "objects": len(objs), "attrs": attrs}


async def suggest_keyframes(db: AsyncSession, track_id: str, *, budget: int = 8) -> dict:
    """Which frames are worth correcting, given the keyframes already set.

    Ranked by how far the current derived box is from what the track actually does, approximated by the
    curvature of the trajectory: a box moving in a straight line needs no keyframe in the middle however
    long it runs, and one going round a corner needs one at the corner. This is where an annotator's next
    correction buys the most frames.
    """
    from db.models import Frame, Object

    rows = (await db.execute(
        select(Object, Frame.ts_ns)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Object.track_id == uuid.UUID(track_id))
        .order_by(Frame.ts_ns))).all()
    if len(rows) < 3:
        return {"track_id": track_id, "suggestions": [],
                "detail": "too short to need intermediate keyframes"}

    centres = []
    for obj, ts in rows:
        x1, y1, x2, y2 = (float(v) for v in obj.bbox)
        centres.append((int(ts), (x1 + x2) / 2, (y1 + y2) / 2, obj))

    scored = []
    for i in range(1, len(centres) - 1):
        _, px, py, _ = centres[i - 1]
        ts, cx, cy, obj = centres[i]
        _, nx, ny, _ = centres[i + 1]
        # Second difference: how far the middle point sits from the straight line between its neighbours.
        # Zero on a straight run at any speed, large at a corner or an occlusion recovery.
        curvature = float(np.hypot((px + nx) / 2 - cx, (py + ny) / 2 - cy))
        if obj.is_keyframe:
            continue
        scored.append({"object_id": str(obj.object_id), "frame_id": str(obj.frame_id),
                       "ts_ns": ts, "curvature": round(curvature, 2)})

    scored.sort(key=lambda s: -s["curvature"])
    return {"track_id": track_id, "length": len(rows),
            "suggestions": scored[:max(1, budget)],
            "detail": "ranked by trajectory curvature; a straight run needs no intermediate keyframe"}


async def tracklet_stats(db: AsyncSession, session_id: str | None = None) -> dict:
    """How much the tracklet workflow is actually saving, corpus-wide."""
    from db.models import Frame, Object

    stmt = (select(func.count(Object.object_id),
                   func.count(Object.object_id).filter(Object.is_keyframe.is_(True)),
                   func.count(func.distinct(Object.track_id)))
            .select_from(Object)
            .where(Object.track_id.isnot(None)))
    if session_id:
        stmt = stmt.join(Frame, Object.frame_id == Frame.frame_id).where(
            Frame.session_id == uuid.UUID(session_id))
    total, keys, tracks = (await db.execute(stmt)).one()
    total, keys, tracks = int(total or 0), int(keys or 0), int(tracks or 0)
    return {"tracked_objects": total, "keyframes": keys, "tracks": tracks,
            "derived": max(0, total - keys),
            # The headline: one correction covering this many frames is the whole argument for the mode.
            "frames_per_keyframe": round(total / keys, 2) if keys else None}
