"""Embedding driver for the Data Intelligence Layer: embed frames (DINOv3 + SigLIP 2) and object crops
(DINOv3) into the pgvector tables, idempotent (skip already-embedded) and batched. Records the model
versions on every row. Used by the backfill script, the frame.ready consumer, and on-demand callers.
"""

from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.embeddings import model_versions
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, FrameEmbedding, Object, ObjectEmbedding
from db.session import get_sessionmaker
from services.autolabel.paths.path_c_qwen3vl import crop_object
from services.intelligence.embed import dinov3, siglip2

log = get_logger("embed_service")


def _decode(store, uri: str):
    # Resilient: synthetic/test frames may point at missing MinIO keys; skip rather than crash backfill.
    try:
        return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


async def embed_frames(session_id: UUID | None = None, limit: int | None = None, only_missing: bool = True) -> dict:
    cfg = get_settings().intel.embed
    store, maker, mv = get_object_store(), get_sessionmaker(), model_versions()
    async with maker() as db:
        stmt = select(Frame.frame_id, Frame.img_uri)
        if session_id is not None:
            stmt = stmt.where(Frame.session_id == session_id)
        if only_missing:
            stmt = stmt.where(Frame.frame_id.notin_(
                select(FrameEmbedding.frame_id).where(FrameEmbedding.dino_vec.isnot(None))))
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).all()

    n = 0
    for i in range(0, len(rows), cfg.batch_size):
        imgs, fids = [], []
        for fid, uri in rows[i:i + cfg.batch_size]:
            img = _decode(store, uri)
            if img is not None:
                imgs.append(img)
                fids.append(fid)
        if not imgs:
            continue
        dvecs, svecs = dinov3.encode_images(imgs), siglip2.encode_images(imgs)
        async with maker() as db:
            for fid, dv, sv in zip(fids, dvecs, svecs):
                await db.merge(FrameEmbedding(frame_id=fid, dino_vec=dv.tolist(), siglip_vec=sv.tolist(), model_versions=mv))
            await db.commit()
        n += len(fids)
        log.info("embed.frames.progress", embedded=n, total=len(rows))
    return {"embedded_frames": n, "model_versions": mv}


async def embed_objects(session_id: UUID | None = None, limit: int | None = None, only_missing: bool = True) -> dict:
    cfg = get_settings().intel.embed
    store, maker, mv = get_object_store(), get_sessionmaker(), model_versions()
    async with maker() as db:
        stmt = (select(Object.object_id, Object.bbox, Frame.img_uri)
                .join(Frame, Frame.frame_id == Object.frame_id)
                .where(Object.state != "rejected").order_by(Object.frame_id))
        if session_id is not None:
            stmt = stmt.where(Frame.session_id == session_id)
        if only_missing:
            stmt = stmt.where(Object.object_id.notin_(select(ObjectEmbedding.object_id)))
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).all()

    n, last_uri, last_img = 0, None, None
    for i in range(0, len(rows), cfg.batch_size):
        crops, oids = [], []
        for oid, bbox, uri in rows[i:i + cfg.batch_size]:
            if uri != last_uri:  # rows are frame-ordered, so decode each frame once
                last_uri, last_img = uri, _decode(store, uri)
            if last_img is None:
                continue
            crops.append(crop_object(last_img, tuple(bbox), cfg.crop_margin))
            oids.append(oid)
        if not crops:
            continue
        dvecs = dinov3.encode_images(crops)
        async with maker() as db:
            for oid, dv in zip(oids, dvecs):
                await db.merge(ObjectEmbedding(object_id=oid, dino_vec=dv.tolist(), model_versions=mv))
            await db.commit()
        n += len(oids)
        log.info("embed.objects.progress", embedded=n, total=len(rows))
    return {"embedded_objects": n, "model_versions": mv}


async def embed_session(session_id: UUID, only_missing: bool = True) -> dict:
    f = await embed_frames(session_id, only_missing=only_missing)
    o = await embed_objects(session_id, only_missing=only_missing)
    return {**f, **o}
