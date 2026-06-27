"""Interactive AI correction: given ONE human correction (a class or attribute change), find
visually-similar objects that still carry the OLD value, so the fix can be previewed and bulk-applied.
Turns the annotator into a reviewer.

Reuses the CLIP object embeddings + cosine search (services/intelligence/embeddings.py). Filtering to the
old class/attr FIRST keeps the cosine over a small candidate set (scale-safe; pgvector/FAISS is the
documented upgrade path). If the source object is not embedded yet, it is embedded on demand.
"""

from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Embedding, Frame, Object
from db.models import Session as DbSession
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_c_qwen3vl import crop_object
from services.intelligence.embeddings import cosine_topk, encode_image, model_tag

log = get_logger("corrections")


async def _source_vector(db: AsyncSession, object_id: UUID) -> np.ndarray | None:
    """The source object's CLIP vector, embedding it on demand if missing."""
    emb = await db.get(Embedding, object_id)
    if emb is not None:
        return np.array(emb.vec, dtype=np.float32)
    obj = await db.get(Object, object_id)
    if obj is None:
        return None
    frame = await db.get(Frame, obj.frame_id)
    img = cv2.imdecode(np.frombuffer(get_object_store().get_bytes(frame.img_uri), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    vec = encode_image(crop_object(img, tuple(obj.bbox), get_settings().models.clip.crop_margin))
    await db.merge(Embedding(object_id=object_id, model=model_tag(), dim=int(vec.shape[0]), vec=vec.tolist()))
    await db.commit()
    return vec


async def _candidate_matrix(db, *, old_class_id, attr_key, old_value, exclude_id, filters):
    """Embeddings of objects that still carry the OLD value, narrowed by metadata filters."""
    stmt = (
        select(Embedding.object_id, Embedding.vec)
        .join(Object, Embedding.object_id == Object.object_id)
        .where(Object.object_id != exclude_id, Object.state != "rejected")
    )
    if old_class_id is not None:
        stmt = stmt.where(Object.class_id == old_class_id)
    if attr_key is not None:
        stmt = stmt.where(Object.attrs[attr_key].astext == str(old_value))

    cam, city = filters.get("cam_id"), filters.get("city")
    if cam or city:
        stmt = stmt.join(Frame, Object.frame_id == Frame.frame_id)
        if cam:
            stmt = stmt.where(Frame.cam_id == cam)
        if city:
            stmt = stmt.join(DbSession, Frame.session_id == DbSession.session_id).where(DbSession.city == city)
    if filters.get("conf_min") is not None:
        stmt = stmt.where(Object.conf >= filters["conf_min"])
    if filters.get("conf_max") is not None:
        stmt = stmt.where(Object.conf <= filters["conf_max"])
    # bbox is xyxy; PG arrays are 1-based -> area = (x2-x1)*(y2-y1).
    area = (Object.bbox[3] - Object.bbox[1]) * (Object.bbox[4] - Object.bbox[2])
    if filters.get("area_min") is not None:
        stmt = stmt.where(area >= filters["area_min"])
    if filters.get("area_max") is not None:
        stmt = stmt.where(area <= filters["area_max"])

    rows = (await db.execute(stmt)).all()
    if not rows:
        return [], np.zeros((0, 1), dtype=np.float32)
    return [r[0] for r in rows], np.array([r[1] for r in rows], dtype=np.float32)


async def correction_candidates(
    db: AsyncSession, object_id: str, *, kind: str, old_class_id=None, attr_key=None,
    old_value=None, new_value=None, filters: dict | None = None, limit: int = 200, threshold: float = 0.82,
) -> dict:
    filters = filters or {}
    q = await _source_vector(db, UUID(object_id))
    if q is None:
        return {"count": 0, "candidates": [], "reason": "source object has no embedding"}
    ids, mat = await _candidate_matrix(
        db, old_class_id=old_class_id, attr_key=attr_key, old_value=old_value,
        exclude_id=UUID(object_id), filters=filters,
    )
    if not ids:
        return {"count": 0, "candidates": []}

    onto = get_ontology()
    out: list[dict] = []
    for i, s in cosine_topk(q, mat, min(limit, len(ids))):
        if s < threshold:
            break
        oid = ids[i]
        obj = await db.get(Object, oid)
        if obj is None:
            continue
        cur = onto.by_id(obj.class_id).name if kind == "class" else (obj.attrs or {}).get(attr_key)
        out.append({
            "object_id": str(oid), "frame_id": str(obj.frame_id),
            "class_name": onto.by_id(obj.class_id).name, "current": cur,
            "conf": obj.conf, "state": obj.state, "score": round(s, 4),
            "crop_url": f"/api/objects/{oid}/crop",
            "already": cur == new_value,  # already at the corrected value (pre-deselect)
        })
    return {"count": len(out), "candidates": out}
