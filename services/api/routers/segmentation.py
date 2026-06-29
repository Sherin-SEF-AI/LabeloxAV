"""Dense full-frame segmentation endpoints: run the auto segmentation (semantic or panoptic), fetch the
metadata, and proxy the colored display overlay. Recovered rasters are stored in the object store; this
router holds the create/read and the overlay image proxy the editor draws.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.storage import get_object_store
from db.models import Frame, FrameSegmentation, Object
from services.api.deps import db_session, require_role
from services.autolabel.ontology import get_ontology

router = APIRouter()


def _summary(row: FrameSegmentation) -> dict:
    return {"found": True, "kind": row.kind, "coverage": row.coverage, "segments": row.segments,
            "source": row.source, "model_version": row.model_version,
            "has_overlay": bool(row.overlay_uri)}


@router.post("/frames/{frame_id}/segment", dependencies=[Depends(require_role("annotator"))])
async def auto_segment(frame_id: str, kind: str = "semantic", db: AsyncSession = Depends(db_session)):
    """Run SAM-everything + VLM to produce the dense raster for the frame. Expensive (GPU); on demand."""
    if kind not in ("semantic", "panoptic"):
        raise HTTPException(400, "kind must be semantic or panoptic")
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    from services.recall.backends import load_image_bgr
    from services.segment2d.semantic import segment_frame

    settings, onto, store = get_settings(), get_ontology(), get_object_store()
    store.ensure_bucket()
    try:
        img = load_image_bgr(store, frame.img_uri)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "frame image unavailable") from exc

    res = segment_frame(img, frame.frame_id, frame.session_id, store, onto, settings, kind=kind)

    if kind == "panoptic":
        # link each panoptic instance to an existing Object by mask/box IoU, so ids stay consistent
        await _link_instances(db, frame, res, store)

    await db.execute(delete(FrameSegmentation).where(
        FrameSegmentation.frame_id == frame.frame_id, FrameSegmentation.kind == kind))
    row = FrameSegmentation(
        frame_id=frame.frame_id, kind=kind, labels_uri=res["labels_uri"], instance_uri=res["instance_uri"],
        overlay_uri=res["overlay_uri"], coverage=res["coverage"], segments=res["segments"], source="proposed",
        model_version=res["model_version"], ontology_version=onto.version)
    db.add(row)
    await db.commit()
    return {"kind": kind, "coverage": res["coverage"], "n_instances": res["n_instances"],
            "segments": res["segments"]}


async def _link_instances(db: AsyncSession, frame: Frame, res: dict, store) -> None:
    """Map panoptic instance ids to existing Object ids when their boxes overlap, so a thing-instance and
    its 2D object share identity. Best-effort by bbox IoU against the frame's objects."""
    import numpy as np

    objs = (await db.execute(select(Object).where(Object.frame_id == frame.frame_id))).scalars().all()
    if not objs or not res["segments"]:
        return
    inst = _load_npz(store, res["instance_uri"]) if res["instance_uri"] else None
    if inst is None:
        return
    for sid, seg in res["segments"].items():
        ys, xs = np.where(inst == int(sid))
        if xs.size == 0:
            continue
        ibox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        best, best_iou = None, 0.3
        for o in objs:
            iou = _iou(ibox, list(o.bbox))
            if iou > best_iou:
                best, best_iou = o, iou
        if best is not None:
            seg["object_id"] = str(best.object_id)


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _load_npz(store, uri: str):
    import io

    import numpy as np

    try:
        return np.load(io.BytesIO(store.get_bytes(uri)))["arr"]
    except Exception:  # noqa: BLE001
        return None


@router.get("/frames/{frame_id}/segment")
async def get_segment(frame_id: str, kind: str = "semantic", db: AsyncSession = Depends(db_session)):
    row = (await db.execute(select(FrameSegmentation).where(
        FrameSegmentation.frame_id == UUID(frame_id), FrameSegmentation.kind == kind))).scalars().first()
    return _summary(row) if row else {"found": False}


@router.get("/frames/{frame_id}/segment/labelids.png")
async def segment_labelids(frame_id: str, kind: str = "semantic", db: AsyncSession = Depends(db_session)):
    """The Cityscapes-style labelIds export: a single-channel PNG whose pixel value is the class id."""
    import cv2

    row = (await db.execute(select(FrameSegmentation).where(
        FrameSegmentation.frame_id == UUID(frame_id), FrameSegmentation.kind == kind))).scalars().first()
    if row is None:
        raise HTTPException(404, "no segmentation")
    labels = _load_npz(get_object_store(), row.labels_uri)
    if labels is None:
        raise HTTPException(404, "labels unavailable")
    ok, buf = cv2.imencode(".png", labels.astype("uint16"))
    if not ok:
        raise HTTPException(500, "could not encode labelIds")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.get("/frames/{frame_id}/segment/panoptic.png")
async def segment_panoptic_png(frame_id: str, db: AsyncSession = Depends(db_session)):
    """COCO-panoptic id-encoded PNG: each pixel's RGB encodes its segment id (id = R + 256*G + 256^2*B).
    Pair it with the segments map from GET /segment (category id + linked object id per segment)."""
    import cv2
    import numpy as np

    row = (await db.execute(select(FrameSegmentation).where(
        FrameSegmentation.frame_id == UUID(frame_id), FrameSegmentation.kind == "panoptic"))).scalars().first()
    if row is None or not row.instance_uri:
        raise HTTPException(404, "no panoptic segmentation")
    inst = _load_npz(get_object_store(), row.instance_uri)
    if inst is None:
        raise HTTPException(404, "instance raster unavailable")
    inst = inst.astype(np.int64)
    rgb = np.zeros((*inst.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (inst % 256).astype(np.uint8)            # R
    rgb[..., 1] = ((inst // 256) % 256).astype(np.uint8)   # G
    rgb[..., 2] = ((inst // 65536) % 256).astype(np.uint8)  # B
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "could not encode panoptic png")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.get("/frames/{frame_id}/segment/overlay")
async def segment_overlay(frame_id: str, kind: str = "semantic", db: AsyncSession = Depends(db_session)):
    row = (await db.execute(select(FrameSegmentation).where(
        FrameSegmentation.frame_id == UUID(frame_id), FrameSegmentation.kind == kind))).scalars().first()
    if row is None or not row.overlay_uri:
        raise HTTPException(404, "no overlay")
    try:
        data = get_object_store().get_bytes(row.overlay_uri)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "overlay unavailable") from exc
    return Response(content=data, media_type="image/png")
