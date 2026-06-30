"""M-4D.3: SAM-propagated mask + box propagation, the temporal labor multiplier. The interp_source value
'sam_propagated' was declared but never produced; this produces it. From a keyframe object, propagate the
box BOTH forward and backward along the clip with Lucas-Kanade optical flow + a RANSAC similarity fit, then
refine each propagated box into a mask with a SAM box prompt (the resident image SAM), so the mask follows
the object, not just the box. Propagated objects are threaded onto the source object's track and routed to
review (state=annotate), never auto-accepted.

Honest scope: this is flow tracking plus per-frame SAM refinement, not the SAM2 memory-bank video model. It
drifts on long runs (the flow does), so it is anchored at the keyframe and seeds human review; the true
SAM2 video predictor is a documented follow-on that would replace the flow step.
"""

from __future__ import annotations

import json
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Object, Track
from db.session import get_sessionmaker
from services.intelligence.propagate import _clip_box, _gray

log = get_logger("sam_propagate")

_LK = {"winSize": (21, 21), "maxLevel": 3,
       "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)}


def flow_box(prev_gray: np.ndarray, cur_gray: np.ndarray, box: list[float]) -> list[float] | None:
    """Move a box from prev_gray to cur_gray via LK optical flow on its interior features plus a RANSAC
    similarity fit. None when it cannot be tracked (too small, too few features, no transform)."""
    h, w = cur_gray.shape
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    roi = prev_gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.01, minDistance=4)
    if pts is None or len(pts) < 4:
        return None
    pts = pts.reshape(-1, 2) + np.float32([x1, y1])
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **_LK)
    st = st.reshape(-1).astype(bool)
    if st.sum() < 4:
        return None
    m, _ = cv2.estimateAffinePartial2D(pts[st], nxt.reshape(-1, 2)[st], method=cv2.RANSAC)
    if m is None:
        return None
    corners = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
    moved = cv2.transform(corners.reshape(-1, 1, 2), m).reshape(-1, 2)
    nb = _clip_box([moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()], w, h)
    if nb[2] - nb[0] < 4 or nb[3] - nb[1] < 4:
        return None
    # reject a blown-up or collapsed track (e.g. flow onto a featureless frame): a real object's box does not
    # jump more than ~4x in area between adjacent frames.
    old_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    if not (0.25 <= (nb[2] - nb[0]) * (nb[3] - nb[1]) / old_area <= 4.0):
        return None
    return nb


def flow_track(grays: list, box0: list[float]) -> list[list[float] | None]:
    """Track a box across a sequence of gray frames (grays[0] is the anchor). Returns the propagated box for
    each of grays[1:], stopping (the rest None) once tracking fails."""
    out: list[list[float] | None] = []
    box, prev = list(box0), grays[0]
    for cur in grays[1:]:
        nb = flow_box(prev, cur, box) if (prev is not None and cur is not None) else None
        out.append(nb)
        if nb is None:
            break
        box, prev = nb, cur
    return out


def _sam_refine(image_bgr, box: list[float]) -> tuple[list[float], list[list[float]]] | None:
    """Best-effort SAM box-prompt refinement: returns (tightened_box, polygons) or None if SAM yields nothing."""
    try:
        from services.api.sam_service import segment as sam_segment
        res = sam_segment(image_bgr, box=box)
    except Exception as exc:  # noqa: BLE001  SAM/GPU unavailable must not abort propagation
        log.warning("sam_propagate.refine_failed", error=str(exc))
        return None
    polys = res.get("polygons") or []
    if not polys:
        return None
    return (res.get("bbox") or box), polys


async def sam_propagate_object(object_id: UUID, n_frames: int = 12, direction: str = "both",
                               refine: bool = True) -> dict:
    """Propagate a keyframe object's box (and SAM mask) forward and/or backward along the clip. direction is
    'forward' | 'backward' | 'both'. Returns the created object ids per direction."""
    store = get_object_store()
    async with get_sessionmaker()() as db:
        obj = await db.get(Object, object_id)
        if obj is None:
            return {"created": 0, "reason": "object not found"}
        anchor = await db.get(Frame, obj.frame_id)
        dirs = ["forward", "backward"] if direction == "both" else [direction]

        track_id = obj.track_id
        if track_id is None:
            tr = Track(session_id=anchor.session_id, class_id=obj.class_id,
                       first_ts_ns=anchor.ts_ns, last_ts_ns=anchor.ts_ns)
            db.add(tr)
            await db.flush()
            track_id = tr.track_id
            obj.track_id = track_id

        created: dict[str, list[str]] = {}
        for d in dirs:
            if d == "forward":
                frames = (await db.execute(select(Frame).where(
                    Frame.session_id == anchor.session_id, Frame.ts_ns > anchor.ts_ns)
                    .order_by(Frame.ts_ns.asc()).limit(n_frames))).scalars().all()
            else:
                frames = list(reversed((await db.execute(select(Frame).where(
                    Frame.session_id == anchor.session_id, Frame.ts_ns < anchor.ts_ns)
                    .order_by(Frame.ts_ns.desc()).limit(n_frames))).scalars().all()))
            if not frames:
                created[d] = []
                continue

            grays = [_gray(store, anchor.img_uri)] + [_gray(store, f.img_uri) for f in frames]
            boxes = flow_track(grays, list(obj.bbox))
            made: list[str] = []
            for fr, box in zip(frames, boxes, strict=False):
                if box is None:
                    break
                src = "flow_propagated"
                mask_uri = None
                if refine:
                    img = cv2.imdecode(np.frombuffer(store.get_bytes(fr.img_uri), np.uint8), cv2.IMREAD_COLOR)
                    ref = _sam_refine(img, box) if img is not None else None
                    if ref is not None:
                        box, polys = ref
                        src = "sam_propagated"
                        key = f"masks/{anchor.session_id}/{fr.frame_id}/prop_{object_id}.json"
                        mask_uri = store.put_bytes(key, json.dumps(
                            {"polygons": polys, "width": fr.width, "height": fr.height}).encode(),
                            "application/json")
                o = Object(frame_id=fr.frame_id, track_id=track_id, class_id=obj.class_id,
                           bbox=[float(v) for v in box], conf=0.5, source="propagated", state="annotate",
                           interp_source=src, mask_uri=mask_uri,
                           provenance={"propagated_from": str(object_id), "method": src, "direction": d})
                db.add(o)
                await db.flush()
                made.append(str(o.object_id))
            created[d] = made

        tr = await db.get(Track, track_id)
        if tr is not None and (created.get("forward") or created.get("backward")):
            tr.first_ts_ns = min(tr.first_ts_ns or anchor.ts_ns, anchor.ts_ns)
            tr.last_ts_ns = max(tr.last_ts_ns or anchor.ts_ns, anchor.ts_ns)
        await db.commit()

    total = sum(len(v) for v in created.values())
    log.info("sam_propagate.done", object=str(object_id), created=total, track=str(track_id))
    return {"created": total, "track_id": str(track_id), "by_direction": created, "refined": refine}
