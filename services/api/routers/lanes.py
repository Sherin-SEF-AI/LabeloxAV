"""Lane annotation endpoints (M2.1): propose (CLRerNet on pod / classical local), list, create/update/
delete human lanes, and propagate a frame's lanes forward by optical flow (source=propagated)."""

from __future__ import annotations

import json
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Lane
from services.api.deps import db_session
from services.autolabel.lane.curves import fit_control_points, mark_ego, propagate_control_points
from services.autolabel.lane.detect import model_tag, propose_lanes
from services.autolabel.lane.linetype import MODEL_VERSION as LINETYPE_VERSION
from services.autolabel.lane.linetype import classify_lane
from services.autolabel.lane.plausible import filter_proposals

log = get_logger("lanes")
router = APIRouter()


class LaneIn(BaseModel):
    control_points: list
    lane_type: str = "solid"
    is_ego: bool = False
    # How strongly the paint supported the type, when the caller has actually measured it. Left out by a
    # human editor, and that absence is meaningful: see the note in update_lane.
    marking_conf: float | None = None


def _decode(store, uri):
    return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)


async def _drivable_classes(db, frame_id) -> dict | None:
    """The frame's drivable polygons, or None if nobody has segmented it.

    None and an empty mask are different and the plausibility test treats them differently: no mask is
    missing evidence, not evidence that the road is missing.
    """
    from db.models import DrivableMask

    dm = await db.get(DrivableMask, frame_id)
    if dm is None or not dm.mask_uri:
        return None
    try:
        return json.loads(get_object_store().get_bytes(dm.mask_uri)).get("classes") or None
    except Exception as exc:  # noqa: BLE001
        log.warning("lanes.drivable_unreadable", frame=str(frame_id), error=str(exc))
        return None


def _row(lane: Lane) -> dict:
    return {"lane_id": str(lane.lane_id), "frame_id": str(lane.frame_id),
            "track_ref": str(lane.track_ref) if lane.track_ref else None,
            "control_points": lane.control_points, "lane_type": lane.lane_type,
            "marking_conf": lane.marking_conf,
            # Null confidence means the type was never measured, which is a different claim from a type
            # measured and found uncertain, and consumers that gate on solid need to tell them apart.
            "measured": lane.marking_conf is not None,
            "is_ego": lane.is_ego, "source": lane.source, "model_version": lane.model_version}


@router.get("/frames/{frame_id}/lanes")
async def list_lanes(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(Lane).where(Lane.frame_id == frame_id))).scalars().all()
    return [_row(lane) for lane in rows]


class BevProjectIn(BaseModel):
    # Points in BEV pixels, as drawn on the warp.
    points: list
    lane_type: str = "solid"
    is_ego: bool = False


async def _bev_context(db: AsyncSession, frame_id: UUID):
    from services.calibration.resolve import resolve_calibration
    from services.hdmap.bev_view import view_for

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    if not frame.cam_id:
        raise HTTPException(400, "this frame has no camera, so there is nothing to calibrate against")
    cal = await resolve_calibration(frame.session_id, frame.cam_id, frame.width, frame.height)
    return frame, cal, view_for(cal)


@router.get("/frames/{frame_id}/bev")
async def frame_bev(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    """The road under this frame, flattened, plus the metric extent it covers.

    A forward camera runs the road to a vanishing point, so the far half of a lane is a handful of pixels,
    parallel lanes converge, and a curve and a lane change look alike. Flattening removes all three.

    The far bound is `ipm_max_range_m`, the codebase's own limit on where a flat-road lift stops meaning
    anything, rather than a number picked to fill the picture. `calibration` travels with the response
    because a nominal calibration and a measured one produce views that look identical and mean different
    things, and an annotator drawing metric lanes should know which they have.
    """
    import base64

    frame, cal, view = await _bev_context(db, frame_id)
    img = _decode(get_object_store(), frame.img_uri)
    if img is None:
        raise HTTPException(404, "frame image unavailable")

    from services.hdmap.bev_view import render

    bev = render(img, cal, view)
    ok, buf = cv2.imencode(".jpg", bev, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "failed to encode the bird's-eye view")
    return {
        "frame_id": str(frame_id),
        "view": view.as_dict(),
        "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
        "calibration": {"source": cal.source, "quality": round(float(cal.quality), 3),
                        "model": cal.model, "cam_id": cal.cam_id},
        # Said plainly rather than left for the annotator to infer from a blurry far edge.
        "caveat": ("the warp assumes a flat road, and a pixel near the far edge is worth many metres"
                   + ("; this camera has no measured calibration, so the metric scale is nominal"
                      if cal.source == "nominal" else "")),
    }


@router.get("/frames/{frame_id}/lanes/bev")
async def lanes_in_bev(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    """This frame's existing lanes, in BEV pixels, so they can be drawn on the warp.

    A control point above the horizon has no ground intersection and is dropped rather than clamped: it
    was never on the road plane, and a clamped point is a coordinate nobody drew.
    """
    from services.hdmap.bev_view import image_to_bev

    _frame, cal, view = await _bev_context(db, frame_id)
    rows = (await db.execute(select(Lane).where(Lane.frame_id == frame_id))).scalars().all()
    out = []
    for lane in rows:
        pts = image_to_bev(lane.control_points or [], cal, view)
        out.append({**_row(lane), "bev_points": pts,
                    "dropped": max(0, len(lane.control_points or []) - len(pts))})
    return {"frame_id": str(frame_id), "view": view.as_dict(), "lanes": out}


@router.post("/frames/{frame_id}/lanes/bev")
async def create_lane_from_bev(frame_id: UUID, body: BevProjectIn, db: AsyncSession = Depends(db_session)):
    """Create a lane from a line drawn on the bird's-eye view.

    This is the half of the mathematics that did not exist. `ipm_pixel_to_vehicle` has lifted a pixel to
    the ground since the georeferencing work; the inverse appeared nowhere, and without it a BEV can be
    looked at and not drawn on.

    The lane is stored in image space like every other lane, so nothing downstream needs to know it was
    drawn on a warp. `marking_conf` is left null: a line drawn on a flattened image is an assertion about
    where the lane runs, not a measurement of the paint.
    """
    from services.hdmap.bev_view import bev_to_image

    frame, cal, view = await _bev_context(db, frame_id)
    pts = bev_to_image(body.points or [], cal, view)
    if len(pts) < 2:
        raise HTTPException(400, {"reason": "fewer than two of those points land on the road ahead of "
                                            "the camera, so they do not describe a lane",
                                  "given": len(body.points or []), "projected": len(pts)})
    lane = Lane(frame_id=frame.frame_id, session_id=frame.session_id, control_points=pts,
                lane_type=body.lane_type, is_ego=body.is_ego, source="human")
    db.add(lane)
    await db.flush()
    r = _row(lane)
    await db.commit()
    return r


@router.post("/frames/{frame_id}/lanes/propose")
async def propose(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    img = _decode(get_object_store(), frame.img_uri)
    if img is None:
        raise HTTPException(500, "could not decode frame image")
    await db.execute(delete(Lane).where(Lane.frame_id == frame.frame_id, Lane.source == "proposed"))
    cps = [fit_control_points(p) for p in propose_lanes(img)]

    # A lane sits on the road. The detector finds bright linear structure, and a dashcam frame is full of
    # bright linear structure that is not a lane: a striped hoarding, a flyover girder, a kerb, an awning.
    # On one real frame that produced six "lanes", every one an edge of a striped wall, drawn as diagonals
    # across the sky. The drivable mask already knows where the road is, so it is asked.
    surface = await _drivable_classes(db, frame.frame_id)
    kept, rejected = filter_proposals(cps, surface, frame.width or 0)
    if rejected:
        log.info("lanes.rejected_off_surface", frame=str(frame.frame_id), n=len(rejected),
                 kept=len(kept))
    cps = [c for c, _ev in kept]

    ego = mark_ego(cps, frame.width, frame.height)
    created = []
    for i, cp in enumerate(cps):
        # Typed from the paint on the way in. This used to be the literal "solid" for every lane the system
        # had ever proposed, which is what left 4,548 of 4,558 lanes claiming a type nobody measured.
        lane_type, conf, evidence = classify_lane(img, cp, frame_width=frame.width)
        lane = Lane(frame_id=frame.frame_id, session_id=frame.session_id, control_points=cp,
                    lane_type=lane_type, is_ego=(i == ego), source="proposed",
                    marking_conf=conf, provenance={"linetype": evidence.as_dict()},
                    model_version=f"{model_tag()}+{LINETYPE_VERSION}")
        db.add(lane)
        created.append(lane)
    await db.flush()
    out = [_row(lane) for lane in created]
    await db.commit()
    return {"proposed": len(out), "lanes": out, "model": model_tag(),
            # Reported rather than dropped quietly. A proposer that silently halves its own output is one
            # nobody can debug, and if this rejects everything the drivable mask is the thing to look at.
            "rejected_off_surface": len(rejected),
            "rejected_detail": [ev for _c, ev in rejected][:5]}


@router.post("/frames/{frame_id}/lanes")
async def create_lane(frame_id: UUID, body: LaneIn, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    lane = Lane(frame_id=frame.frame_id, session_id=frame.session_id, control_points=body.control_points,
                lane_type=body.lane_type, is_ego=body.is_ego, source="human",
                marking_conf=body.marking_conf)
    db.add(lane)
    await db.flush()
    r = _row(lane)
    await db.commit()
    return r


@router.put("/lanes/{lane_id}")
async def update_lane(lane_id: UUID, body: LaneIn, db: AsyncSession = Depends(db_session)):
    lane = await db.get(Lane, lane_id)
    if lane is None:
        raise HTTPException(404, "lane not found")
    lane.control_points, lane.lane_type, lane.is_ego, lane.source = body.control_points, body.lane_type, body.is_ego, "human"
    lane.model_version = None  # a human now owns this lane; drop the stale proposing-model tag so provenance is not "human - clrernet"
    # And the same for the paint confidence, for the same reason.
    #
    # `marking_conf` is documented as "how strongly the paint supported the type", written by
    # classify_lane. A person retyping a lane has not measured the paint; they have asserted the type.
    # Leaving the machine's number behind attaches a measurement to a claim it was not made about, and a
    # consumer gating on "solid with confidence above 0.8" would then be reading the old type's evidence.
    # Null is the honest value: `_row` already distinguishes it as "never measured".
    lane.marking_conf = body.marking_conf
    await db.commit()
    return _row(lane)


@router.delete("/lanes/{lane_id}")
async def delete_lane(lane_id: UUID, db: AsyncSession = Depends(db_session)):
    lane = await db.get(Lane, lane_id)
    if lane is None:
        raise HTTPException(404, "lane not found")
    await db.delete(lane)
    await db.commit()
    return {"deleted": str(lane_id)}


@router.post("/frames/{frame_id}/lanes/propagate")
async def propagate(frame_id: UUID, frames: int = 8, db: AsyncSession = Depends(db_session)):
    """Carry this frame's lanes forward via optical flow; the annotator only fixes keyframes."""
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    store = get_object_store()
    lanes = (await db.execute(select(Lane).where(Lane.frame_id == frame.frame_id))).scalars().all()
    if not lanes:
        return {"created": 0, "reason": "no lanes on the source frame"}
    nexts = (await db.execute(
        select(Frame).where(Frame.session_id == frame.session_id, Frame.ts_ns > frame.ts_ns)
        .order_by(Frame.ts_ns).limit(frames))).scalars().all()
    prev_gray = cv2.cvtColor(_decode(store, frame.img_uri), cv2.COLOR_BGR2GRAY)
    cur = {lane.lane_id: lane.control_points for lane in lanes}
    meta = {lane.lane_id: (lane.lane_type, lane.is_ego, lane.track_ref or lane.lane_id) for lane in lanes}
    created = 0
    for nf in nexts:
        nimg = _decode(store, nf.img_uri)
        if nimg is None:
            break
        cur_gray = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
        for lid in list(cur):
            ncp = propagate_control_points(prev_gray, cur_gray, cur[lid])
            if ncp is None:
                continue
            lt, ego, ref = meta[lid]
            # Typed against the frame it landed on, not the one it came from: a line that is solid where it
            # was drawn may be dashed two seconds later, and inheriting the keyframe's type propagates a
            # claim about paint nobody looked at.
            ptype, pconf, pev = classify_lane(nimg, ncp, frame_width=nf.width)
            db.add(Lane(frame_id=nf.frame_id, session_id=frame.session_id, control_points=ncp,
                        lane_type=(ptype if pconf > 0 else lt), is_ego=ego, source="propagated",
                        track_ref=ref, marking_conf=pconf,
                        provenance={"linetype": pev.as_dict(), "inherited_type": lt},
                        model_version=f"optical-flow+{LINETYPE_VERSION}"))
            cur[lid] = ncp
            created += 1
        prev_gray = cur_gray
    await db.commit()
    return {"created": created, "to_frames": len(nexts)}
