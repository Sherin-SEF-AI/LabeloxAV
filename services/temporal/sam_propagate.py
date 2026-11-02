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
from services.temporal.flow import flow_track
from services.temporal.flow import gray as _gray

log = get_logger("sam_propagate")

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
                # The anchor's camera only; see services/intelligence/propagate.py.
                frames = (await db.execute(select(Frame).where(
                    Frame.session_id == anchor.session_id, Frame.cam_id == anchor.cam_id,
                    Frame.ts_ns > anchor.ts_ns)
                    .order_by(Frame.ts_ns.asc()).limit(n_frames))).scalars().all()
            else:
                # Backward: walk from the anchor to progressively earlier frames (nearest-first), so optical
                # flow sees small step-to-step motion. Reversing to ascending order made the first flow step
                # jump from the anchor all the way to the farthest frame, breaking the track.
                frames = (await db.execute(select(Frame).where(
                    Frame.session_id == anchor.session_id, Frame.cam_id == anchor.cam_id,
                    Frame.ts_ns < anchor.ts_ns)
                    .order_by(Frame.ts_ns.desc()).limit(n_frames))).scalars().all()
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
