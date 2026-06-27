"""Object detail, frame image proxy, and SAM click-to-segment."""

from __future__ import annotations

import json
import uuid
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, Object, Review
from services.api.deps import CreateObjectIn, MaskIn, ObjectDetail, SegmentIn, current_user, db_session
from services.autolabel.ontology import get_ontology

router = APIRouter()


def _mask_polygons(mask_uri: str | None) -> list[list[float]]:
    if not mask_uri:
        return []
    store = get_object_store()
    try:
        return json.loads(store.get_bytes(mask_uri)).get("polygons", [])
    except Exception:
        return []


def _mask_key(session_id, frame_id, object_id) -> str:
    return f"masks/{session_id}/{frame_id}/{object_id}.json"


def _write_mask(store, session_id, frame_id, object_id, polygons, width, height) -> str:
    # Same polygon-JSON shape services/autolabel/persist.py writes, so the read path is identical.
    payload = {"encoding": "polygon", "polygons": polygons, "height": height, "width": width}
    return store.put_bytes(_mask_key(session_id, frame_id, object_id),
                           json.dumps(payload).encode(), "application/json")


def _detail(obj: Object, frame: Frame, onto) -> ObjectDetail:
    return ObjectDetail(
        object_id=str(obj.object_id),
        frame_id=str(obj.frame_id),
        session_id=str(frame.session_id),
        track_id=str(obj.track_id) if obj.track_id else None,
        ts_ns=frame.ts_ns,
        cam_id=frame.cam_id,
        image_url=f"/api/frames/{frame.frame_id}/image",
        width=frame.width,
        height=frame.height,
        class_id=obj.class_id,
        class_name=onto.by_id(obj.class_id).name,
        bbox=list(obj.bbox),
        mask_polygons=_mask_polygons(obj.mask_uri),
        attrs=obj.attrs or {},
        conf=obj.conf,
        state=obj.state,
        source=obj.source,
        provenance=obj.provenance or {},
    )


@router.get("/objects/{object_id}", response_model=ObjectDetail)
async def get_object(object_id: str, db: AsyncSession = Depends(db_session)):
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, obj.frame_id)
    return _detail(obj, frame, get_ontology())


@router.get("/frames/{frame_id}/objects")
async def frame_objects(frame_id: str, db: AsyncSession = Depends(db_session)):
    from sqlalchemy import select

    onto = get_ontology()
    rows = (await db.execute(select(Object).where(Object.frame_id == UUID(frame_id)))).scalars().all()
    return [
        {
            "object_id": str(o.object_id),
            "track_id": str(o.track_id) if o.track_id else None,
            "class_id": o.class_id,
            "class_name": onto.by_id(o.class_id).name,
            "bbox": list(o.bbox),
            "conf": o.conf,
            "state": o.state,
            "mask_polygons": _mask_polygons(o.mask_uri),
        }
        for o in rows
    ]


@router.get("/frames/{frame_id}")
async def get_frame(frame_id: str, db: AsyncSession = Depends(db_session)):
    """Frame meta for the editor: dimensions, image url, object count, and prev/next frame in the
    session (by ts_ns) for keyboard frame navigation."""
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    prev = (await db.execute(
        select(Frame.frame_id).where(Frame.session_id == frame.session_id, Frame.ts_ns < frame.ts_ns)
        .order_by(Frame.ts_ns.desc()).limit(1))).scalar_one_or_none()
    nxt = (await db.execute(
        select(Frame.frame_id).where(Frame.session_id == frame.session_id, Frame.ts_ns > frame.ts_ns)
        .order_by(Frame.ts_ns.asc()).limit(1))).scalar_one_or_none()
    n = (await db.execute(select(func.count()).select_from(Object).where(Object.frame_id == frame.frame_id))).scalar_one()
    return {
        "frame_id": str(frame.frame_id), "session_id": str(frame.session_id),
        "width": frame.width, "height": frame.height, "ts_ns": frame.ts_ns, "cam_id": frame.cam_id,
        "image_url": f"/api/frames/{frame.frame_id}/image", "n_objects": int(n),
        "prev_frame_id": str(prev) if prev else None, "next_frame_id": str(nxt) if nxt else None,
    }


@router.post("/frames/{frame_id}/objects", response_model=ObjectDetail)
async def create_object(frame_id: str, payload: CreateObjectIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Create a human-drawn object on a frame (source=human, state=accepted). Optional mask."""
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    onto = get_ontology()
    if not onto.has_name(payload.class_name):
        raise HTTPException(400, f"unknown class '{payload.class_name}'")
    if len(payload.bbox) != 4:
        raise HTTPException(400, "bbox must be [x1,y1,x2,y2]")
    if payload.attrs:
        errors = onto.validate_attrs(payload.attrs)
        if errors:
            raise HTTPException(400, {"attr_errors": errors})

    oid = uuid.uuid4()
    mask_uri = mask_encoding = None
    if payload.mask_polygons:
        mask_uri = _write_mask(get_object_store(), frame.session_id, frame.frame_id, oid,
                               payload.mask_polygons, frame.width, frame.height)
        mask_encoding = "polygon"
    obj = Object(
        object_id=oid, frame_id=frame.frame_id, class_id=onto.by_name(payload.class_name).id,
        bbox=payload.bbox, mask_uri=mask_uri, mask_encoding=mask_encoding, attrs=payload.attrs or {},
        conf=1.0, source="human", state=payload.state, provenance={"created_by": "human-annotation"},
    )
    db.add(obj)
    db.add(Review(object_id=oid, reviewer=user.name if user else "anon", user_id=user.user_id if user else None,
                  action="create", before=None,
                  after={"class_id": obj.class_id, "bbox": list(obj.bbox), "attrs": obj.attrs, "state": obj.state},
                  time_spent_ms=0, ts_ns=now_ns()))
    await db.commit()
    return _detail(obj, frame, onto)


@router.put("/objects/{object_id}/mask")
async def update_mask(object_id: str, payload: MaskIn, db: AsyncSession = Depends(db_session)):
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, obj.frame_id)
    obj.mask_uri = _write_mask(get_object_store(), frame.session_id, frame.frame_id, obj.object_id,
                               payload.polygons, payload.width or frame.width, payload.height or frame.height)
    obj.mask_encoding = "polygon"
    await db.commit()
    return {"object_id": str(obj.object_id), "mask_polygons": payload.polygons}


@router.delete("/objects/{object_id}")
async def delete_object(object_id: str, db: AsyncSession = Depends(db_session)):
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    await db.delete(obj)  # Review rows cascade; the mask blob is left (harmless, content-addressed path)
    await db.commit()
    return {"deleted": object_id}


@router.post("/objects/{object_id}/propagate")
async def propagate_object(object_id: str, frames: int = 12, db: AsyncSession = Depends(db_session)):
    """Label once, carry forward: optical-flow propagate this object's box across the next `frames`
    frames as an annotate-state track the human confirms. Yields the GPU to training is moot (CPU)."""
    from services.intelligence.propagate import propagate_forward

    return await propagate_forward(UUID(object_id), frames)


@router.get("/frames/{frame_id}/image")
async def frame_image(frame_id: str, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    data = get_object_store().get_bytes(frame.img_uri)
    return Response(content=data, media_type="image/jpeg")


@router.get("/objects/{object_id}/crop")
async def object_crop(object_id: str, pad: float = 0.15, db: AsyncSession = Depends(db_session)):
    """A JPEG crop of the object's bbox (with padding) for the track timeline thumbnails."""
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, obj.frame_id)
    buf = np.frombuffer(get_object_store().get_bytes(frame.img_uri), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "failed to decode frame image")
    h, w = img.shape[:2]
    x1, y1, x2, y2 = obj.bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = img
    ok, out = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=out.tobytes(), media_type="image/jpeg")


@router.post("/segment")
async def segment(payload: SegmentIn, db: AsyncSession = Depends(db_session)):
    from sqlalchemy import select

    from db.models import TrainingJob
    from services.api.sam_service import segment as run_segment

    # Single-GPU discipline: interactive segmentation yields to an active training job. Loading SAM
    # on top of a running train would OOM and KILL the multi-hour job, so refuse cleanly (no GPU touch).
    # Box-level review (accept/reject/reclassify) needs no GPU and still works.
    running = (await db.execute(
        select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1)
    )).first()
    if running is not None:
        raise HTTPException(503, "GPU reserved for an active training job. Interactive segmentation is "
                                 "paused until it finishes; box review (accept/reject/reclassify) still works.")

    frame = await db.get(Frame, UUID(payload.frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    buf = np.frombuffer(get_object_store().get_bytes(frame.img_uri), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "failed to decode frame image")
    try:
        return run_segment(img, points=payload.points, labels=payload.labels, box=payload.box)
    except Exception as exc:  # noqa: BLE001
        # On a single GPU, a running training job can consume all VRAM. Surface that cleanly (503)
        # instead of an unhandled 500 so the UI can show a friendly "GPU busy" notice. Box-level
        # review (accept/reject/reclassify) does not need the GPU and still works.
        name = type(exc).__name__
        if "OutOfMemory" in name or "GpuCapacity" in name or "CUDA" in str(exc):
            raise HTTPException(503, "GPU busy (a training job is using the GPU). Interactive "
                                     "segmentation is unavailable until it finishes; box review still works.")
        raise
