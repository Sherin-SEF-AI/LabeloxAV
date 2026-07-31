"""Semantic search endpoints: compute embeddings, NL object search, and visual find-similar."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.embeddings import fused_frame_neighbors
from db.models import Frame, FrameEmbedding, Object, ObjectEmbedding
from services.api.deps import db_session
from services.autolabel.ontology import get_ontology
from services.intelligence.embeddings import (
    compute_session_embeddings,
    search_objects_by_text,
    similar_objects,
)

router = APIRouter()


class SimilarIn(BaseModel):
    frame_id: str | None = None
    object_id: str | None = None
    image_b64: str | None = None
    mode: str = "visual"  # visual (DINOv3) | semantic (SigLIP 2) | fused (both, blended)
    k: int = 24
    # Reranking controls (services/intelligence/search/similar.py). Defaults give a diverse, thresholded
    # result; a caller can widen or disable each.
    min_sim: float = 0.0          # drop neighbours below this cosine similarity
    diversity: bool = True        # collapse near-identical results (e.g. the same object across frames)
    same_class: bool = False      # object search only: restrict to the query object's class
    exclude_track: bool = True    # object search only: drop other crops of the very object you started from
    city: str | None = None       # restrict to one city
    session_id: str | None = None  # restrict to one session
    # Which plane an uploaded crop searches. `object_id` has always searched objects and `frame_id` frames,
    # because the query itself said which; an uploaded image says nothing, so it silently meant frames.
    # "here is a picture of the thing, find me more of it" is the exemplar query people actually want, and
    # it was unreachable. Defaults to frame so existing callers are unchanged.
    target: str = "frame"         # frame | object


def _decorate_frames(nbrs) -> list[dict]:
    # The reranker already carried scene forward, so no per-row DB round trip is needed.
    return [{"frame_id": c["frame_id"], "image_url": f"/api/frames/{c['frame_id']}/image",
             "scene": c.get("scene"), "score": round(c["sim"], 4)} for c in nbrs]


def _decorate_objects(onto, nbrs) -> list[dict]:
    return [{"object_id": c["object_id"], "frame_id": c["frame_id"],
             "class_name": onto.by_id(c["class_id"]).name, "track_id": c.get("track_id"),
             "crop_url": f"/api/objects/{c['object_id']}/crop", "score": round(c["sim"], 4)}
            for c in nbrs]


@router.post("/search/similar")
async def search_similar(body: SimilarIn, db: AsyncSession = Depends(db_session)):
    """Visual (DINOv3) or semantic (SigLIP 2) neighbours of a frame, an object, or an uploaded image, reranked
    for diversity and a similarity floor (services/intelligence/search/similar.py)."""
    from services.intelligence.search.similar import find_similar_frames, find_similar_objects

    common = {"k": body.k, "min_sim": body.min_sim, "diversity": body.diversity,
              "city": body.city, "session_id": UUID(body.session_id) if body.session_id else None}

    if body.object_id:  # object similarity is DINOv3-visual
        emb = await db.get(ObjectEmbedding, UUID(body.object_id))
        if emb is None:
            return {"kind": "object", "results": [], "reason": "object not embedded yet"}
        class_id = None
        exclude_track = None
        if body.same_class or body.exclude_track:
            ob = await db.get(Object, UUID(body.object_id))
            if ob is not None:
                if body.same_class:
                    class_id = ob.class_id
                if body.exclude_track and ob.track_id is not None:
                    exclude_track = str(ob.track_id)
        nbrs = await find_similar_objects(db, emb.dino_vec, exclude_object_id=UUID(body.object_id),
                                          class_id=class_id, exclude_track_id=exclude_track, **common)
        return {"kind": "object", "mode": "visual", "results": _decorate_objects(get_ontology(), nbrs)}

    fused = body.mode == "fused"
    space = "siglip" if body.mode == "semantic" else "dino"
    if body.frame_id:
        emb = await db.get(FrameEmbedding, UUID(body.frame_id))
        if emb is None:
            return {"kind": "frame", "results": [], "reason": "frame not embedded yet"}
        if fused and emb.dino_vec is not None and emb.siglip_vec is not None:
            # Fused blends both spaces; the exact blended distance is not one of the single-space candidate
            # queries, so keep the dedicated path and decorate its (id, score) tuples.
            nbrs = await fused_frame_neighbors(db, emb.dino_vec, emb.siglip_vec, k=body.k,
                                               exclude_frame_id=UUID(body.frame_id))
            return {"kind": "frame", "mode": "fused",
                    "results": await _decorate_fused_frames(db, nbrs, body.min_sim)}
        qv = emb.siglip_vec if space == "siglip" else emb.dino_vec
        nbrs = await find_similar_frames(db, qv, space=space, exclude_frame_id=UUID(body.frame_id), **common)
    elif body.image_b64:
        import base64

        import cv2
        import numpy as np

        from services.intelligence.embed import dinov3, siglip2

        img = cv2.imdecode(np.frombuffer(base64.b64decode(body.image_b64), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode image")
        if body.target == "object":
            # An uploaded crop against the object plane: the exemplar query. DINOv3 only, because that is
            # the space object crops are indexed in and the one that answers "more things that look like
            # this"; SigLIP2 on a crop would compare a picture against a text-aligned space for no gain.
            # Letterboxed the same way the indexed crops were, so the query sits in the same distribution
            # rather than a stretched version of it.
            from services.intelligence.embed.prep import square_letterbox

            qv = dinov3.encode_image(square_letterbox(img)).tolist()
            nbrs = await find_similar_objects(db, qv, **common)
            return {"kind": "object", "mode": "visual",
                    "results": _decorate_objects(get_ontology(), nbrs)}
        if fused:
            nbrs = await fused_frame_neighbors(db, dinov3.encode_image(img).tolist(),
                                               siglip2.encode_image(img).tolist(), k=body.k)
            return {"kind": "frame", "mode": "fused",
                    "results": await _decorate_fused_frames(db, nbrs, body.min_sim)}
        qv = (siglip2.encode_image(img) if space == "siglip" else dinov3.encode_image(img)).tolist()
        nbrs = await find_similar_frames(db, qv, space=space, **common)
    else:
        raise HTTPException(400, "provide frame_id, object_id, or image_b64")
    return {"kind": "frame", "mode": space, "results": _decorate_frames(nbrs)}


async def _decorate_fused_frames(db, nbrs, min_sim: float) -> list[dict]:
    out = []
    for fid, score in nbrs:
        if score < min_sim:
            continue
        fr = await db.get(Frame, UUID(fid))
        if fr is not None:
            out.append({"frame_id": fid, "image_url": f"/api/frames/{fid}/image",
                        "scene": fr.scene, "score": round(score, 4)})
    return out


@router.post("/embeddings/compute")
async def embeddings_compute(payload: dict):
    session_id = payload["session_id"]
    return await compute_session_embeddings(UUID(session_id), payload.get("limit"))


@router.get("/search/objects")
async def search_objects(
    db: AsyncSession = Depends(db_session),
    q: str = Query(...),
    session_id: str | None = None,
    limit: int = 24,
):
    results = await search_objects_by_text(db, q, limit=limit, session_id=session_id)
    return {"query": q, "count": len(results), "results": results}


@router.get("/objects/{object_id}/similar")
async def objects_similar(object_id: str, db: AsyncSession = Depends(db_session), limit: int = 12):
    return {"object_id": object_id, "results": await similar_objects(db, object_id, limit=limit)}


@router.get("/search/semantic")
async def search_semantic(db: AsyncSession = Depends(db_session), q: str = Query(...), k: int = 24):
    """Natural-language frame search: parse scene/class filters then SigLIP 2 pgvector rerank."""
    from services.intelligence.search.query import semantic_search

    return await semantic_search(db, q, k=k)


@router.post("/scene/classify")
async def scene_classify(session_id: str | None = None, limit: int | None = None):
    """Zero-shot scene tags (weather/time_of_day/road_type/density) for a session, or the whole corpus.
    Cheap: a matmul over the SigLIP 2 frame vectors already in the index, no image decode."""
    from services.intelligence.scene import classify_session

    return await classify_session(UUID(session_id) if session_id else None, limit)
