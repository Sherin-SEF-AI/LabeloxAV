"""pgvector ANN helpers for the Data Intelligence Layer. Cosine top-k over the frame/object embedding
tables via the HNSW indexes, with optional session/city/class filters, plus the model-version registry
recorded on every vector (provenance stays one walk). Replaces the brute-force numpy path.

Vectors are L2-normalized, so pgvector cosine distance d gives similarity 1 - d. ORDER BY the
cosine_distance expression uses the HNSW index (vector_cosine_ops).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, FrameEmbedding, Object, ObjectEmbedding
from db.models import Session as DbSession
from services.intelligence.embed import dinov3, siglip2


def model_versions() -> dict:
    """The exact checkpoints currently in use, recorded on every vector written."""
    return {"dino": dinov3.model_tag(), "siglip": siglip2.model_tag()}


async def frame_neighbors(
    db: AsyncSession, query_vec, *, space: str = "dino", k: int = 24,
    exclude_frame_id: UUID | None = None, session_id: UUID | None = None, city: str | None = None,
) -> list[tuple[str, float]]:
    """Top-k frames by cosine to query_vec in the DINOv3 (visual) or SigLIP 2 (semantic) space."""
    col = FrameEmbedding.dino_vec if space == "dino" else FrameEmbedding.siglip_vec
    dist = col.cosine_distance(list(map(float, query_vec))).label("d")
    stmt = select(FrameEmbedding.frame_id, dist).where(col.isnot(None))
    if exclude_frame_id is not None:
        stmt = stmt.where(FrameEmbedding.frame_id != exclude_frame_id)
    if session_id is not None or city is not None:
        stmt = stmt.join(Frame, Frame.frame_id == FrameEmbedding.frame_id)
        if session_id is not None:
            stmt = stmt.where(Frame.session_id == session_id)
        if city is not None:
            stmt = stmt.join(DbSession, DbSession.session_id == Frame.session_id).where(DbSession.city == city)
    rows = (await db.execute(stmt.order_by(dist).limit(k))).all()
    return [(str(fid), 1.0 - float(d)) for fid, d in rows]


async def object_neighbors(
    db: AsyncSession, query_vec, *, k: int = 24,
    exclude_object_id: UUID | None = None, class_id: int | None = None,
) -> list[tuple[str, float]]:
    """Top-k object crops by DINOv3 cosine to query_vec, optionally restricted to one class."""
    dist = ObjectEmbedding.dino_vec.cosine_distance(list(map(float, query_vec))).label("d")
    stmt = select(ObjectEmbedding.object_id, dist)
    if exclude_object_id is not None:
        stmt = stmt.where(ObjectEmbedding.object_id != exclude_object_id)
    if class_id is not None:
        stmt = stmt.join(Object, Object.object_id == ObjectEmbedding.object_id).where(Object.class_id == class_id)
    rows = (await db.execute(stmt.order_by(dist).limit(k))).all()
    return [(str(oid), 1.0 - float(d)) for oid, d in rows]
