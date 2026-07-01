"""Semantic search endpoints: compute embeddings, NL object search, and visual find-similar."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.embeddings import frame_neighbors, fused_frame_neighbors, object_neighbors
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


async def _decorate_frames(db, nbrs) -> list[dict]:
    out = []
    for fid, score in nbrs:
        fr = await db.get(Frame, UUID(fid))
        if fr is not None:
            out.append({"frame_id": fid, "image_url": f"/api/frames/{fid}/image",
                        "scene": fr.scene, "score": round(score, 4)})
    return out


async def _decorate_objects(db, onto, nbrs) -> list[dict]:
    out = []
    for oid, score in nbrs:
        ob = await db.get(Object, UUID(oid))
        if ob is not None:
            out.append({"object_id": oid, "frame_id": str(ob.frame_id),
                        "class_name": onto.by_id(ob.class_id).name,
                        "crop_url": f"/api/objects/{oid}/crop", "score": round(score, 4)})
    return out


@router.post("/search/similar")
async def search_similar(body: SimilarIn, db: AsyncSession = Depends(db_session)):
    """Visual (DINOv3) or semantic (SigLIP 2) neighbors of a frame, an object, or an uploaded image."""
    if body.object_id:  # object similarity is DINOv3-visual
        emb = await db.get(ObjectEmbedding, UUID(body.object_id))
        if emb is None:
            return {"kind": "object", "results": [], "reason": "object not embedded yet"}
        nbrs = await object_neighbors(db, emb.dino_vec, k=body.k, exclude_object_id=UUID(body.object_id))
        return {"kind": "object", "mode": "visual", "results": await _decorate_objects(db, get_ontology(), nbrs)}

    fused = body.mode == "fused"
    space = "siglip" if body.mode == "semantic" else "dino"
    if body.frame_id:
        emb = await db.get(FrameEmbedding, UUID(body.frame_id))
        if emb is None:
            return {"kind": "frame", "results": [], "reason": "frame not embedded yet"}
        if fused and emb.dino_vec is not None and emb.siglip_vec is not None:
            nbrs = await fused_frame_neighbors(db, emb.dino_vec, emb.siglip_vec, k=body.k, exclude_frame_id=UUID(body.frame_id))
        else:
            qv = emb.siglip_vec if space == "siglip" else emb.dino_vec
            nbrs = await frame_neighbors(db, qv, space=space, k=body.k, exclude_frame_id=UUID(body.frame_id))
    elif body.image_b64:
        import base64

        import cv2
        import numpy as np

        from services.intelligence.embed import dinov3, siglip2

        img = cv2.imdecode(np.frombuffer(base64.b64decode(body.image_b64), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode image")
        if fused:
            nbrs = await fused_frame_neighbors(db, dinov3.encode_image(img).tolist(), siglip2.encode_image(img).tolist(), k=body.k)
        else:
            qv = (siglip2.encode_image(img) if space == "siglip" else dinov3.encode_image(img)).tolist()
            nbrs = await frame_neighbors(db, qv, space=space, k=body.k)
    else:
        raise HTTPException(400, "provide frame_id, object_id, or image_b64")
    return {"kind": "frame", "mode": body.mode if fused else space, "results": await _decorate_frames(db, nbrs)}


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
