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
from db.models import Object, ObjectEmbedding
from services.autolabel.ontology import get_ontology

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
    """Embed a session's object crops. Kept under its old name because two routes call it by that name.

    It used to run CLIP and write the legacy `embedding` table, which nothing reads. Two endpoints exposed
    it, including the correction dialog's own "compute embeddings" button, so pressing that button spent GPU
    time filling a dead table while the coverage figure beside it read `object_embedding` and did not move.
    A button that cannot affect the number it offers to fix is worse than no button.

    It now delegates to the real embedder, which writes the DINOv3 and SigLIP2 vectors that find-similar,
    text search and the correction dialog actually read. `_load_matrix` went with it: its only caller was the
    correction search, which no longer loads a whole table into memory to cosine it in Python.
    """
    from services.intelligence.embed.service import embed_objects

    return await embed_objects(session_id=session_id, limit=limit, only_missing=True)


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
    """Mean of a scenario's actor-object crops, for semantic scenario ranking. Actor ids are track ids.

    Reads the SigLIP2 vector rather than the legacy CLIP table. That table was abandoned when the pipeline
    moved to pgvector, and this function was the last thing still reading it: with 39 rows against 567,527
    in `object_embedding`, it returned None for every scenario and the semantic half of the ranking was
    silently zero for all of them.

    SigLIP2 rather than DINOv3 because the caller compares this mean against a text vector, and DINOv3 has
    no text tower. SigLIP2 puts images and text in one space, which is what makes the comparison meaningful
    rather than merely well typed.

    One query per scenario rather than one per object: a scenario with forty actors was forty round trips,
    each fetching a vector to average and throw away.
    """
    if not actor_ids:
        return None
    try:
        track_ids = [UUID(a) for a in actor_ids]
    except (ValueError, AttributeError):
        return None
    rows = (await db.execute(
        select(ObjectEmbedding.siglip_vec)
        .join(Object, Object.object_id == ObjectEmbedding.object_id)
        .where(Object.track_id.in_(track_ids), ObjectEmbedding.siglip_vec.isnot(None)))).scalars().all()
    if not rows:
        return None
    m = np.mean([np.asarray(v, dtype=np.float32) for v in rows], axis=0)
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
