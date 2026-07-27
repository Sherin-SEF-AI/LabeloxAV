"""CLIP/SigLIP embeddings for semantic search (the Vector index plane, pgvector-first seam).

Per-object crop embeddings power: find-similar (visual neighbours for mining/dedup/active-learning
diversity) and natural-language object search (text query embedded into the same space). Scenario
semantic search reuses the actor-object embeddings. Cosine is computed in numpy at P0 scale; the
upgrade is pgvector then Qdrant once embeddings cross tens of millions.
"""

from __future__ import annotations

import threading
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Embedding, Frame, Object
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_c_qwen3vl import crop_object

log = get_logger("embeddings")

_lock = threading.Lock()
_state: dict = {}


def _model():
    if "model" not in _state:
        with _lock:
            if "model" not in _state:
                import clip
                import torch

                dev = get_settings().gpu.device if torch.cuda.is_available() else "cpu"
                name = get_settings().models.clip.model
                model, preprocess = clip.load(name, device=dev)
                model.eval()
                _state.update(model=model, preprocess=preprocess, device=dev, torch=torch, clip=clip)
                log.info("clip.loaded", model=name, device=dev)
    return _state


def model_tag() -> str:
    return "clip-" + get_settings().models.clip.model.lower().replace("/", "").replace("-", "")


def encode_image(image_bgr: np.ndarray) -> np.ndarray:
    s = _model()
    from PIL import Image

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = s["preprocess"](Image.fromarray(rgb)).unsqueeze(0).to(s["device"])
    with s["torch"].no_grad():
        feat = s["model"].encode_image(tensor)
    v = feat.cpu().numpy()[0].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def encode_text(text: str) -> np.ndarray:
    s = _model()
    tok = s["clip"].tokenize([text]).to(s["device"])
    with s["torch"].no_grad():
        feat = s["model"].encode_text(tok)
    v = feat.cpu().numpy()[0].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_topk(query: np.ndarray, mat: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (row_index, score) for the top-k rows of mat by cosine to query (rows pre-normalized)."""
    if mat.shape[0] == 0:
        return []
    sims = mat @ query
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


async def compute_session_embeddings(session_id: UUID, limit: int | None = None) -> dict:
    store = get_object_store()
    from db.session import get_sessionmaker

    maker = get_sessionmaker()
    tag = model_tag()
    margin = get_settings().models.clip.crop_margin
    n = 0
    async with maker() as db:
        stmt = (
            select(Object, Frame.img_uri)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .where(Frame.session_id == session_id, Object.state != "rejected")
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).all()

        # Decode each frame once.
        cache: dict[str, np.ndarray] = {}
        for obj, img_uri in rows:
            if img_uri not in cache:
                buf = np.frombuffer(store.get_bytes(img_uri), dtype=np.uint8)
                cache[img_uri] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cache[img_uri]
            if img is None:
                continue
            crop = crop_object(img, tuple(obj.bbox), margin)
            vec = encode_image(crop)
            await db.merge(Embedding(object_id=obj.object_id, model=tag, dim=int(vec.shape[0]), vec=vec.tolist()))
            n += 1
        await db.commit()
    log.info("embeddings.computed", session_id=str(session_id), n=n)
    return {"session_id": str(session_id), "embedded": n, "model": tag}


async def _load_matrix(db: AsyncSession, session_id: str | None = None):
    stmt = select(Embedding.object_id, Embedding.vec)
    if session_id:
        stmt = (
            select(Embedding.object_id, Embedding.vec)
            .join(Object, Embedding.object_id == Object.object_id)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .where(Frame.session_id == UUID(session_id))
        )
    rows = (await db.execute(stmt)).all()
    if not rows:
        return [], np.zeros((0, 1), dtype=np.float32)
    ids = [r[0] for r in rows]
    mat = np.array([r[1] for r in rows], dtype=np.float32)
    return ids, mat


async def _decorate(db: AsyncSession, ids: list[UUID], scores: dict[UUID, float]) -> list[dict]:
    onto = get_ontology()
    out = []
    for oid in ids:
        obj = await db.get(Object, oid)
        if obj is None:
            continue
        out.append({
            "object_id": str(oid),
            "frame_id": str(obj.frame_id),
            "class_id": obj.class_id,
            "class_name": onto.by_id(obj.class_id).name,
            "conf": obj.conf,
            "state": obj.state,
            "score": round(scores[oid], 4),
            "image_url": f"/api/frames/{obj.frame_id}/image",
        })
    return out


async def search_objects_by_text(db: AsyncSession, query: str, limit: int = 24,
                                 session_id: str | None = None) -> list[dict]:
    """Find object crops matching a phrase, through the indexed SigLIP2 image-text space.

    This used to load the entire legacy CLIP `Embedding` table into memory and cosine it in Python. That table
    is no longer populated by the pipeline (see similar_objects below, which was moved off it for the same
    reason), so the search quietly returned nothing on a live corpus, and even when populated it was a full
    scan rather than an index lookup.

    It now runs as a pgvector ANN query over ObjectEmbedding.siglip_vec, which is the same plane find-similar
    uses and is backfilled by the embedding daemon. Crops not yet embedded are skipped rather than ranked, so
    an unembedded object is absent rather than appearing as a poor match.
    """
    from core.embeddings import object_neighbors_by_text

    hits = await object_neighbors_by_text(
        db, query, k=limit, session_id=UUID(session_id) if session_id else None)
    if not hits:
        return []
    ids = [UUID(oid) for oid, _ in hits]
    scores = {UUID(oid): float(sim) for oid, sim in hits}
    return await _decorate(db, ids, scores)


async def similar_objects(db: AsyncSession, object_id: str, limit: int = 12,
                          diversity: bool = True, min_sim: float = 0.0) -> list[dict]:
    # Find-similar reads the live DINOv3 ObjectEmbedding table via pgvector HNSW (the same source
    # /search/similar uses), not the legacy CLIP `Embedding` table, which the current pipeline no longer
    # populates: an in-Python cosine over that dead table returned nothing. Routed through the reranker so the
    # object-page "similar" strip gets the same diversity dedup as the search page: without it, an object that
    # is tracked across frames returns fifteen copies of itself.
    from db.models import ObjectEmbedding
    from services.intelligence.search.similar import find_similar_objects

    emb = await db.get(ObjectEmbedding, UUID(object_id))
    if emb is None:
        return []
    ob = await db.get(Object, UUID(object_id))
    exclude_track = str(ob.track_id) if ob is not None and ob.track_id is not None else None
    nbrs = await find_similar_objects(db, emb.dino_vec, k=limit, min_sim=min_sim, diversity=diversity,
                                      exclude_object_id=UUID(object_id), exclude_track_id=exclude_track)
    ids = [UUID(c["object_id"]) for c in nbrs]
    scores = {UUID(c["object_id"]): c["sim"] for c in nbrs}
    return await _decorate(db, ids, scores)


async def scenario_embedding(db: AsyncSession, actor_ids: list[str]) -> np.ndarray | None:
    """Mean of a scenario's actor-object embeddings, for semantic scenario ranking. Actor ids are
    track ids; gather each track's object embeddings and average."""
    if not actor_ids:
        return None
    vecs = []
    for aid in actor_ids:
        objs = (await db.execute(select(Object.object_id).where(Object.track_id == UUID(aid)))).scalars().all()
        for oid in objs:
            emb = await db.get(Embedding, oid)
            if emb is not None:
                vecs.append(np.array(emb.vec, dtype=np.float32))
    if not vecs:
        return None
    m = np.mean(vecs, axis=0)
    return m / (np.linalg.norm(m) + 1e-8)


def main() -> None:
    import asyncio
    from uuid import UUID as _UUID

    import click

    from core.logging import setup_logging

    @click.command()
    @click.option("--session", "session_id", required=True)
    @click.option("--limit", type=int, default=None)
    def _cli(session_id: str, limit: int | None) -> None:
        setup_logging(get_settings().log_level)
        click.echo(asyncio.run(compute_session_embeddings(_UUID(session_id), limit)))

    _cli()


if __name__ == "__main__":
    main()
