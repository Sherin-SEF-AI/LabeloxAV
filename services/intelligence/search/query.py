"""Hybrid natural-language frame search (M1.4), on Postgres + pgvector (OpenSearch deferred). A query
like "night rain autorickshaw" is parsed into structured filters (scene.time_of_day=night,
scene.weather=rain, class=autorickshaw) over the frame.scene jsonb + object-class join, then the
candidate set is reranked by a SigLIP 2 text embedding against frame_embedding.siglip_vec via pgvector.

A clean indexer seam is left for OpenSearch: swap the candidate-filter query for an OpenSearch query
without touching the rerank.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, FrameEmbedding, Object
from services.autolabel.ontology import get_ontology
from services.intelligence.scene import SCENE_AXES

log = get_logger("nl_search")

# value -> axis, e.g. "night" -> "time_of_day", "rain" -> "weather", "dense" -> "density"
_SCENE_VALUE_AXIS = {label: axis for axis, pairs in SCENE_AXES.items() for label, _ in pairs}


def parse_query(q: str) -> tuple[dict, list[str]]:
    """Split a query into scene filters {axis: value} and ontology class names. Unmatched words still
    contribute to the semantic rerank (the full query string is always embedded)."""
    onto = get_ontology()
    scene: dict[str, str] = {}
    classes: list[str] = []
    for tok in q.lower().replace(",", " ").split():
        norm = tok.replace("-", "_")
        if tok in _SCENE_VALUE_AXIS:
            scene[_SCENE_VALUE_AXIS[tok]] = tok
        elif onto.has_name(norm):
            classes.append(norm)
    return scene, classes


async def semantic_search(db: AsyncSession, q: str, k: int = 24) -> dict:
    from services.intelligence.embed import siglip2

    onto = get_ontology()
    scene, classes = parse_query(q)

    # structured candidate filter
    cand = select(Frame.frame_id).join(FrameEmbedding, FrameEmbedding.frame_id == Frame.frame_id).where(
        FrameEmbedding.siglip_vec.isnot(None))
    for axis, val in scene.items():
        cand = cand.where(Frame.scene[axis].astext == val)
    if classes:
        cids = [onto.by_name(c).id for c in classes]
        cand = cand.where(Frame.frame_id.in_(select(Object.frame_id).where(Object.class_id.in_(cids))))
    candidate_ids = [r[0] for r in (await db.execute(cand)).all()]

    # OCR text (M2.4): frames whose objects carry recognized road text matching a query word are
    # searchable through Phase 1. Union them into the candidate set (then reranked semantically).
    from sqlalchemy import or_

    words = [w for w in q.lower().replace(",", " ").split() if len(w) >= 3]
    ocr_hit = False
    if words:
        ocr_frames = [r[0] for r in (await db.execute(
            select(Object.frame_id).where(Object.ocr_text.isnot(None),
                                          or_(*[Object.ocr_text.ilike(f"%{w}%") for w in words])))).all()]
        if ocr_frames:
            ocr_hit = True
            candidate_ids = list(set(candidate_ids) | set(ocr_frames))

    # SigLIP 2 text embedding rerank (pgvector)
    qvec = siglip2.encode_text(q).tolist()
    dist = FrameEmbedding.siglip_vec.cosine_distance(qvec).label("d")
    rerank = select(FrameEmbedding.frame_id, dist).where(FrameEmbedding.siglip_vec.isnot(None))
    if scene or classes or ocr_hit:  # only constrain when the parse or OCR matched something
        if not candidate_ids:
            return {"query": q, "filters": scene, "classes": classes, "count": 0, "results": []}
        rerank = rerank.where(FrameEmbedding.frame_id.in_(candidate_ids))
    rows = (await db.execute(rerank.order_by(dist).limit(k))).all()

    results = []
    for fid, d in rows:
        fr = await db.get(Frame, fid)
        if fr is not None:
            results.append({"frame_id": str(fid), "image_url": f"/api/frames/{fid}/image",
                            "scene": fr.scene, "score": round(1.0 - float(d), 4)})
    return {"query": q, "filters": scene, "classes": classes, "count": len(results), "results": results}
