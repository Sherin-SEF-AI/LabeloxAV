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

# ef_search alone is not enough once a query carries a WHERE clause, and the failure is silent.
#
# A plain HNSW scan visits ef_search candidates and stops. The filter is applied to those candidates, so if
# none of them satisfies it the query returns ZERO ROWS rather than searching further. Measured here: a
# search filtered to `hoarding`, a class with 26,216 embedded objects, returned nothing at every ef_search up
# to the 1,000 maximum, because the query object sat inside a cluster of 2,354 near-identical crops that
# filled the entire window. Nothing in the result distinguishes that from "there are genuinely no
# neighbours", which is how a broken find-similar can look like an empty corpus.
#
# Iterative scan is pgvector 0.8's answer: when the filter eats the candidates, it keeps scanning instead of
# giving up. `relaxed_order` rather than `strict_order` because these callers rerank by similarity anyway and
# strict ordering costs a sort for a guarantee nobody here depends on. max_scan_tuples bounds the worst case
# so a filter that matches nothing cannot turn into a full table scan of half a million vectors.
HNSW_ITERATIVE_SCAN = "relaxed_order"
HNSW_MAX_SCAN_TUPLES = 100_000


def model_versions() -> dict:
    """The exact checkpoints + crop prep currently in use, recorded on every vector written."""
    return {"dino": dinov3.model_tag(), "siglip": siglip2.model_tag(), "prep": PREP_TAG}


# Which hnsw.* settings this server actually has, learned once. Asking first rather than setting and
# catching: a failed SET aborts the surrounding transaction in Postgres, so a try/except per statement would
# turn a missing GUC on an older pgvector into "current transaction is aborted" for the query that follows.
_HNSW_GUCS: set[str] | None = None


async def _hnsw_settings(db: AsyncSession) -> set[str]:
    global _HNSW_GUCS
    if _HNSW_GUCS is None:
        try:
            rows = await db.execute(text("SELECT name FROM pg_settings WHERE name LIKE 'hnsw.%'"))
            _HNSW_GUCS = {r[0] for r in rows}
        except Exception:  # noqa: BLE001 -- not Postgres, or pgvector absent
            _HNSW_GUCS = set()
    return _HNSW_GUCS


async def tune_recall(db: AsyncSession) -> None:
    """Widen the ANN search for one transaction, and let it keep looking when a filter empties the window."""
    have = await _hnsw_settings(db)
    wanted = {
        "hnsw.ef_search": str(int(HNSW_EF_SEARCH)),
        "hnsw.iterative_scan": f"'{HNSW_ITERATIVE_SCAN}'",
        "hnsw.max_scan_tuples": str(int(HNSW_MAX_SCAN_TUPLES)),
    }
    for name, value in wanted.items():
        if name in have:
            await db.execute(text(f"SET LOCAL {name} = {value}"))


async def frame_neighbors(
    db: AsyncSession, query_vec, *, space: str = "dino", k: int = 24,
    exclude_frame_id: UUID | None = None, session_id: UUID | None = None, city: str | None = None,
) -> list[tuple[str, float]]:
    """Top-k frames by cosine to query_vec in the DINOv3 (visual) or SigLIP 2 (semantic) space."""
    await tune_recall(db)
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
    await tune_recall(db)
    dist = ObjectEmbedding.dino_vec.cosine_distance(list(map(float, query_vec))).label("d")
    stmt = select(ObjectEmbedding.object_id, dist)
    if exclude_object_id is not None:
        stmt = stmt.where(ObjectEmbedding.object_id != exclude_object_id)
    if class_id is not None:
        stmt = stmt.join(Object, Object.object_id == ObjectEmbedding.object_id).where(Object.class_id == class_id)
    rows = (await db.execute(stmt.order_by(dist).limit(k))).all()
    return [(str(oid), 1.0 - float(d)) for oid, d in rows]


async def object_neighbors_by_text(
    db: AsyncSession, query_text: str, *, k: int = 24, class_id: int | None = None,
    session_id: UUID | None = None, min_sim: float | None = None,
) -> list[tuple[str, float]]:
    """Top-k object crops matching a text query, through the shared SigLIP2 image-text space.

    This is the query the crop plane could not answer before: ObjectEmbedding carried only a DINOv3 vector,
    and DINOv3 has no text tower, so a phrase could retrieve whole frames but never the objects inside them.
    SigLIP2 embeds images and text into one space, so a text vector and a crop vector are directly comparable
    and the search is one ANN query rather than a frame search followed by manual inspection.

    Crops whose siglip_vec has not been backfilled yet are skipped rather than treated as distant: a null is
    an absence of evidence, and ranking it as a poor match would push genuinely unembedded objects to the
    bottom as though they had been considered.
    """
    from services.intelligence.embed import siglip2

    await tune_recall(db)
    vec = _prompt_vector(query_text, siglip2)
    dist = ObjectEmbedding.siglip_vec.cosine_distance(vec).label("d")
    stmt = select(ObjectEmbedding.object_id, dist).where(ObjectEmbedding.siglip_vec.isnot(None))
    if class_id is not None or session_id is not None:
        stmt = stmt.join(Object, Object.object_id == ObjectEmbedding.object_id)
        if class_id is not None:
            stmt = stmt.where(Object.class_id == class_id)
        if session_id is not None:
            stmt = stmt.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    rows = (await db.execute(stmt.order_by(dist).limit(k))).all()
    hits = [(str(oid), 1.0 - float(d)) for oid, d in rows]
    # No similarity floor by default: "the k nearest" should return k. An implicit floor silently returns
    # fewer than asked for, which reads as "nothing else matched" when really the caller was never told.
    # A caller that wants a floor sets one.
    return hits if min_sim is None else [h for h in hits if h[1] >= min_sim]


def _prompt_vector(query_text: str, siglip2) -> list[float]:
    """Encode a query with the caption-style prompt SigLIP was trained on.

    A bare noun phrase sits off the distribution of the captions the model saw, and the standard remedy
    (used already by the crop classifier) is to wrap it in a caption template. Doing it here too keeps
    text-to-object retrieval consistent with text-to-frame and measurably improves ranking on short queries.
    """
    text = query_text.strip()
    prompt = text if text.lower().startswith(("a photo", "a picture")) else f"a photo of {text}"
    return [float(x) for x in siglip2.encode_text(prompt)]


async def object_candidates(
    db: AsyncSession, query_vec, *, k: int = 200, class_id: int | None = None,
    exclude_object_id: UUID | None = None, session_id: UUID | None = None,
    city: str | None = None,
) -> list[dict]:
    """Top-k object candidates with their vector and metadata in one query, for a reranking stage.

    object_neighbors returns only (id, sim); a diversity or same-track filter needs the vector and the
    object's track/class too, and issuing a follow-up query per candidate is what made find-similar feel
    sluggish. This fetches everything the reranker needs at once: the crop vector (to measure candidate to
    candidate similarity for dedup) and the class/track (to filter) alongside the score.
    """
    await tune_recall(db)
    dist = ObjectEmbedding.dino_vec.cosine_distance(list(map(float, query_vec))).label("d")
    stmt = (select(ObjectEmbedding.object_id, dist, ObjectEmbedding.dino_vec,
                   Object.class_id, Object.track_id, Object.frame_id, Object.conf)
            .join(Object, Object.object_id == ObjectEmbedding.object_id))
    if class_id is not None:
        stmt = stmt.where(Object.class_id == class_id)
    if exclude_object_id is not None:
        stmt = stmt.where(ObjectEmbedding.object_id != exclude_object_id)
    if session_id is not None or city is not None:
        stmt = stmt.join(Frame, Frame.frame_id == Object.frame_id)
        if session_id is not None:
            stmt = stmt.where(Frame.session_id == session_id)
        if city is not None:
            stmt = stmt.join(DbSession, DbSession.session_id == Frame.session_id).where(DbSession.city == city)
    rows = (await db.execute(stmt.order_by(dist).limit(k))).all()
    return [{"object_id": str(oid), "sim": 1.0 - float(d), "vec": vec,
             "class_id": cid, "track_id": str(tid) if tid else None,
             "frame_id": str(fid), "conf": float(c) if c is not None else None}
            for oid, d, vec, cid, tid, fid, c in rows]


async def frame_candidates(
    db: AsyncSession, query_vec, *, space: str = "dino", k: int = 200,
    exclude_frame_id: UUID | None = None, session_id: UUID | None = None, city: str | None = None,
) -> list[dict]:
    """Top-k frame candidates with vector + scene metadata, the frame analogue of object_candidates."""
    await tune_recall(db)
    col = FrameEmbedding.dino_vec if space == "dino" else FrameEmbedding.siglip_vec
    dist = col.cosine_distance(list(map(float, query_vec))).label("d")
    stmt = (select(FrameEmbedding.frame_id, dist, col, Frame.session_id, Frame.scene)
            .join(Frame, Frame.frame_id == FrameEmbedding.frame_id).where(col.isnot(None)))
    if exclude_frame_id is not None:
        stmt = stmt.where(FrameEmbedding.frame_id != exclude_frame_id)
    if session_id is not None:
        stmt = stmt.where(Frame.session_id == session_id)
    if city is not None:
        stmt = stmt.join(DbSession, DbSession.session_id == Frame.session_id).where(DbSession.city == city)
    rows = (await db.execute(stmt.order_by(dist).limit(k))).all()
    return [{"frame_id": str(fid), "sim": 1.0 - float(d), "vec": vec,
             "session_id": str(sid), "scene": scene}
            for fid, d, vec, sid, scene in rows]


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
