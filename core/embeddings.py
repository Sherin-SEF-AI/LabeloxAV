"""pgvector ANN helpers for the Data Intelligence Layer. Cosine top-k over the frame/object embedding
tables via the HNSW indexes, with optional session/city/class filters, plus the model-version registry
recorded on every vector (provenance stays one walk). Replaces the brute-force numpy path.

Vectors are L2-normalized, so pgvector cosine distance d gives similarity 1 - d. ORDER BY the
cosine_distance expression uses the HNSW index (vector_cosine_ops).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, FrameEmbedding, Object, ObjectEmbedding
from db.models import Session as DbSession
from services.intelligence.embed import dinov3, siglip2
from services.intelligence.embed.prep import PREP_TAG

# HNSW is approximate: its recall rises with ef_search (how many candidates it explores). pgvector defaults
# to 40; a labeling tool wants the true nearest neighbour, so widen the search. Set per transaction
# (SET LOCAL) right before each ANN query so it never leaks into unrelated statements.
HNSW_EF_SEARCH = 200


def model_versions() -> dict:
    """The exact checkpoints + crop prep currently in use, recorded on every vector written."""
    return {"dino": dinov3.model_tag(), "siglip": siglip2.model_tag(), "prep": PREP_TAG}


async def _tune_recall(db: AsyncSession) -> None:
    try:
        await db.execute(text(f"SET LOCAL hnsw.ef_search = {int(HNSW_EF_SEARCH)}"))
    except Exception:  # noqa: BLE001 -- non-pg backend or missing GUC: fall back to the default silently
        pass


async def frame_neighbors(
    db: AsyncSession, query_vec, *, space: str = "dino", k: int = 24,
    exclude_frame_id: UUID | None = None, session_id: UUID | None = None, city: str | None = None,
) -> list[tuple[str, float]]:
    """Top-k frames by cosine to query_vec in the DINOv3 (visual) or SigLIP 2 (semantic) space."""
    await _tune_recall(db)
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
    await _tune_recall(db)
    dist = ObjectEmbedding.dino_vec.cosine_distance(list(map(float, query_vec))).label("d")
    stmt = select(ObjectEmbedding.object_id, dist)
    if exclude_object_id is not None:
        stmt = stmt.where(ObjectEmbedding.object_id != exclude_object_id)
    if class_id is not None:
        stmt = stmt.join(Object, Object.object_id == ObjectEmbedding.object_id).where(Object.class_id == class_id)
    rows = (await db.execute(stmt.order_by(dist).limit(k))).all()
    return [(str(oid), 1.0 - float(d)) for oid, d in rows]


async def fused_frame_neighbors(
    db: AsyncSession, dino_vec, siglip_vec, *, w_visual: float = 0.5, k: int = 24,
    exclude_frame_id: UUID | None = None, session_id: UUID | None = None, city: str | None = None,
) -> list[tuple[str, float]]:
    """Top-k frames ranking both spaces together: DINOv3 catches visual look-alikes, SigLIP 2 catches
    semantic/scene matches, and a lot of the best neighbours only rank high in one. Blend the two cosine
    distances (w_visual weights DINOv3) and order by the combined score. Reuses the frame vectors already
    stored, so no re-embed is needed; the combined expression is exact (no single-column HNSW), which is
    fine for a rerank-scale query. Returns (frame_id, fused_similarity in [0,1])."""
    w = max(0.0, min(1.0, float(w_visual)))
    dd = FrameEmbedding.dino_vec.cosine_distance(list(map(float, dino_vec)))
    ds = FrameEmbedding.siglip_vec.cosine_distance(list(map(float, siglip_vec)))
    dist = (dd * w + ds * (1.0 - w)).label("d")
    stmt = select(FrameEmbedding.frame_id, dist).where(
        FrameEmbedding.dino_vec.isnot(None), FrameEmbedding.siglip_vec.isnot(None))
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
