"""Object detail, frame image proxy, and SAM click-to-segment."""

from __future__ import annotations

import json
import uuid
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, Object, ObjectRelationship, Review
from services.api.deps import (
    CreateObjectIn,
    MaskIn,
    ObjectDetail,
    RelateIn,
    SegmentIn,
    current_user,
    db_session,
    require_role,
)
from services.autolabel.ontology import get_ontology

router = APIRouter()

# Directed relationship kinds the editor offers (the India case is rider_of on a two-wheeler).
_RELATION_KINDS = {"rider_of", "towed_by", "part_of", "member_of", "occludes"}


@router.post("/objects/{object_id}/relate", dependencies=[Depends(require_role("reviewer"))])
async def relate_object(object_id: str, payload: RelateIn, db: AsyncSession = Depends(db_session)):
    """Create a directed relationship from this object to another on the same frame."""
    if payload.kind not in _RELATION_KINDS:
        raise HTTPException(400, f"unknown relation kind '{payload.kind}'")
    if object_id == payload.to_object_id:
        raise HTTPException(400, "cannot relate an object to itself")
    frm = await db.get(Object, UUID(object_id))
    to = await db.get(Object, UUID(payload.to_object_id))
    if frm is None or to is None:
        raise HTTPException(404, "object not found")
    rel = ObjectRelationship(from_object_id=frm.object_id, to_object_id=to.object_id,
                             frame_id=frm.frame_id, kind=payload.kind)
    db.add(rel)
    await db.commit()
    return {"relationship_id": str(rel.relationship_id), "from_object_id": object_id,
            "to_object_id": payload.to_object_id, "kind": payload.kind}


@router.delete("/relationships/{relationship_id}", dependencies=[Depends(require_role("reviewer"))])
async def delete_relationship(relationship_id: str, db: AsyncSession = Depends(db_session)):
    rel = await db.get(ObjectRelationship, UUID(relationship_id))
    if rel is not None:
        await db.delete(rel)
        await db.commit()
    return {"deleted": relationship_id}


@router.get("/frames/{frame_id}/relationships")
async def frame_relationships(frame_id: str, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(ObjectRelationship)
            .where(ObjectRelationship.frame_id == UUID(frame_id)))).scalars().all()
    return [{"relationship_id": str(r.relationship_id), "from_object_id": str(r.from_object_id),
             "to_object_id": str(r.to_object_id), "kind": r.kind} for r in rows]


@router.get("/frames/{frame_id}/cuboids")
async def frame_cuboids(frame_id: str, db: AsyncSession = Depends(db_session)):
    """Project every cuboid_3d on the frame onto the camera image, so the 3D box is visible (and editable)
    in the 2D editor. Uses the configured rig + nominal intrinsics, so it works without LiDAR calibration."""
    from services.lidar.boxes import project_cuboid

    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    rows = (await db.execute(select(Object).where(
        Object.frame_id == frame.frame_id, Object.cuboid_3d.isnot(None)))).scalars().all()
    out = []
    for o in rows:
        c = o.cuboid_3d or {}
        center, size, yaw = c.get("center"), c.get("size"), float(c.get("yaw", 0.0))
        if not center or not size:
            continue
        dims = [size[1], size[0], size[2]]  # cuboid_3d size is [w,l,h]; project_cuboid wants [length,width,height]
        proj = project_cuboid(center, dims, yaw, frame.cam_id, frame.width, frame.height)
        out.append({"object_id": str(o.object_id), "corners_uv": proj["corners_uv"], "edges": proj["edges"],
                    "any_in_image": proj["any_in_image"]})
    return out


@router.get("/frames/{frame_id}/lift_ground")
async def lift_ground(frame_id: str, u: float, v: float, db: AsyncSession = Depends(db_session)):
    """The ego ground point (z=0) a pixel sees, for placing a cuboid on the road from an image click."""
    from services.lidar.project import camera_ray_to_ego

    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    ray = camera_ray_to_ego(u, v, frame.cam_id, frame.width, frame.height)
    o, dvec = ray["origin"], ray["direction"]
    # A pixel above the horizon (or a ray parallel to the road) simply has no ground point. That is a
    # normal answer for a valid query, not a client error, so return ego=null with a reason instead of a
    # 400 the browser logs on every hover/click near the skyline.
    if abs(float(dvec[2])) < 1e-6:
        return {"ego": None, "reason": "ray is parallel to the ground"}
    t = -float(o[2]) / float(dvec[2])
    if t <= 0:
        return {"ego": None, "reason": "pixel is above the horizon (no ground ahead)"}
    return {"ego": [round(float(o[0] + t * dvec[0]), 3), round(float(o[1] + t * dvec[1]), 3), 0.0]}


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
        version=obj.version,
        rot_deg=obj.rot_deg or 0.0,
        keypoints=obj.keypoints,
        polyline=obj.polyline,
        cuboid_3d=obj.cuboid_3d,
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
            "version": o.version,
            "rot_deg": o.rot_deg or 0.0,
            "keypoints": o.keypoints,
            "polyline": o.polyline,
            "cuboid_3d": o.cuboid_3d,
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
    # The dominant annotation source on this frame, so the editor can say plainly whether these labels are
    # imported from a public dataset (Mapillary / IDD / BDD) or produced in-app.
    src_rows = (await db.execute(
        select(Object.source, func.count()).where(Object.frame_id == frame.frame_id)
        .group_by(Object.source).order_by(func.count().desc()))).all()
    annotation_source = src_rows[0][0] if src_rows else None
    import_format = None
    if annotation_source == "imported":
        prov = (await db.execute(select(Object.provenance).where(
            Object.frame_id == frame.frame_id, Object.source == "imported").limit(1))).scalar()
        import_format = (prov or {}).get("import_format")
    return {
        "frame_id": str(frame.frame_id), "session_id": str(frame.session_id),
        "width": frame.width, "height": frame.height, "ts_ns": frame.ts_ns, "cam_id": frame.cam_id,
        "image_url": f"/api/frames/{frame.frame_id}/image", "n_objects": int(n),
        "annotation_source": annotation_source, "import_format": import_format,
        "prev_frame_id": str(prev) if prev else None, "next_frame_id": str(nxt) if nxt else None,
        "is_lidar": bool(frame.lidar), "lidar_points": (frame.lidar or {}).get("n_points"),
        "lidar_res": ((frame.lidar or {}).get("bev") or {}).get("res"),  # metres per pixel, for the ruler
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
        errors = onto.validate_attrs(payload.attrs, onto.by_name(payload.class_name).id)
        if errors:
            raise HTTPException(400, {"attr_errors": errors})

    # Idempotency: if this frame already carries an object for the client's idem_key, return it rather
    # than creating a duplicate (a retried or raced autosave from the editor).
    if payload.idem_key:
        existing = (await db.execute(
            select(Object).where(Object.frame_id == frame.frame_id,
                                  Object.provenance["idem_key"].astext == payload.idem_key))).scalars().first()
        if existing is not None:
            return _detail(existing, frame, onto)

    oid = uuid.uuid4()
    mask_uri = mask_encoding = None
    if payload.mask_polygons:
        mask_uri = _write_mask(get_object_store(), frame.session_id, frame.frame_id, oid,
                               payload.mask_polygons, frame.width, frame.height)
        mask_encoding = "polygon"
    obj = Object(
        object_id=oid, frame_id=frame.frame_id, class_id=onto.by_name(payload.class_name).id,
        bbox=payload.bbox, mask_uri=mask_uri, mask_encoding=mask_encoding, attrs=payload.attrs or {},
        conf=1.0, source="human", state=payload.state, rot_deg=payload.rot_deg, keypoints=payload.keypoints,
        polyline=payload.polyline, cuboid_3d=payload.cuboid_3d,
        provenance={"created_by": "human-annotation", "idem_key": payload.idem_key},
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


@router.post("/objects/{object_id}/sam_propagate", dependencies=[Depends(require_role("annotator"))])
async def sam_propagate(object_id: str, frames: int = 12, direction: str = "both", refine: bool = True,
                        db: AsyncSession = Depends(db_session)):
    """Label once, carry both ways: propagate this keyframe object's box forward AND backward with optical
    flow, refining each into a mask with a SAM box prompt (interp_source=sam_propagated). Routed to review."""
    from services.temporal.sam_propagate import sam_propagate_object

    return await sam_propagate_object(UUID(object_id), frames, direction, refine)


@router.get("/frames/{frame_id}/image")
async def frame_image(frame_id: str, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    try:
        data = get_object_store().get_bytes(frame.img_uri)
    except Exception as exc:  # noqa: BLE001  (missing/unreadable blob -> 404, never a 500 that breaks the editor)
        raise HTTPException(404, "frame image unavailable") from exc
    return Response(content=data, media_type="image/jpeg")


@router.get("/objects/{object_id}/crop")
async def object_crop(object_id: str, pad: float = 0.15, db: AsyncSession = Depends(db_session)):
    """A JPEG crop of the object's bbox (with padding) for the track timeline thumbnails."""
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, obj.frame_id)
    try:
        buf = np.frombuffer(get_object_store().get_bytes(frame.img_uri), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "frame image unavailable") from exc
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(404, "failed to decode frame image")
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
        return run_segment(img, points=payload.points, labels=payload.labels, box=payload.box, precise=payload.precise)
    except Exception as exc:  # noqa: BLE001
        # On a single GPU, a running training job can consume all VRAM. Surface that cleanly (503)
        # instead of an unhandled 500 so the UI can show a friendly "GPU busy" notice. Box-level
        # review (accept/reject/reclassify) does not need the GPU and still works.
        name = type(exc).__name__
        if "OutOfMemory" in name or "GpuCapacity" in name or "CUDA" in str(exc):
            raise HTTPException(503, "GPU busy (a training job is using the GPU). Interactive "
                                     "segmentation is unavailable until it finishes; box review still works.")
        raise


class ClassifyIn(BaseModel):
    frame_id: str
    box: list[float]                     # [x1, y1, x2, y2] in image pixels


@router.post("/objects/classify")
async def classify_object(payload: ClassifyIn, db: AsyncSession = Depends(db_session)):
    """Zero-shot: what class is the object in this box? Crops the region and scores it against the ontology
    with SigLIP 2, so a SAM box or wand click can auto-detect the class instead of the annotator picking it.
    Returns the top-k class suggestions with confidence; the first is the auto-assigned class."""
    frame = await db.get(Frame, UUID(payload.frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    if len(payload.box) != 4:
        raise HTTPException(400, "box must be [x1,y1,x2,y2]")
    img = cv2.imdecode(np.frombuffer(get_object_store().get_bytes(frame.img_uri), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "failed to decode frame image")
    from services.autolabel.classify_crop import classify_crop
    from services.autolabel.paths.path_c_qwen3vl import crop_object
    crop = crop_object(img, tuple(payload.box), 0.08)
    if crop is None or crop.size == 0:
        raise HTTPException(400, "empty crop")
    try:
        preds = classify_crop(crop)
    except Exception as exc:  # noqa: BLE001
        if "CUDA" in str(exc) or "OutOfMemory" in type(exc).__name__:
            raise HTTPException(503, "GPU busy; auto-classify unavailable right now") from exc
        raise
    return {"predictions": preds}
