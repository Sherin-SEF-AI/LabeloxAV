"""Active-learning curation over DINOv2 frame embeddings. The core data-engine lever: label the RIGHT
frames, not random ones. Novelty (far from everything = coverage gap), duplicates (near-identical = skip),
and a farthest-point diversity sample (a maximally-varied set to label next)."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import func, select

from core.logging import get_logger
from db.models import Frame, FrameEmbedding
from db.session import get_sessionmaker

log = get_logger("curation")


async def _load_matrix(session_id: str | None) -> tuple[list[str], np.ndarray]:
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = select(FrameEmbedding.frame_id, FrameEmbedding.dino_vec).where(FrameEmbedding.dino_vec.isnot(None))
        if session_id:
            stmt = stmt.join(Frame, Frame.frame_id == FrameEmbedding.frame_id).where(Frame.session_id == UUID(session_id))
        rows = (await db.execute(stmt)).all()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    ids = [str(r[0]) for r in rows]
    mat = np.asarray([r[1] for r in rows], dtype=np.float32)
    return ids, mat


def _farthest_point_sample(mat: np.ndarray, k: int) -> list[int]:
    """Greedy diversity: start from the most central frame, repeatedly add the frame farthest (lowest
    max-similarity) from the chosen set. A maximally-varied subset."""
    n = mat.shape[0]
    k = min(k, n)
    if n == 0:
        return []
    chosen = [int(np.argmax(mat @ mat.mean(axis=0)))]  # most representative seed
    max_sim = mat @ mat[chosen[0]]
    while len(chosen) < k:
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, mat @ mat[nxt])
    return chosen


async def curation_summary(session_id: str | None = None, dup_threshold: float = 0.93, top: int = 40) -> dict:
    ids, mat = await _load_matrix(session_id)
    maker = get_sessionmaker()
    async with maker() as db:
        fstmt = select(func.count()).select_from(Frame)
        if session_id:
            fstmt = fstmt.where(Frame.session_id == UUID(session_id))
        total_frames = (await db.execute(fstmt)).scalar_one()

    if len(ids) < 2:
        return {"total_frames": int(total_frames), "embedded": len(ids), "embedded_pct": 0.0,
                "mean_nn_sim": None, "duplicate_frames": 0, "novel": [], "duplicates": []}

    sim = mat @ mat.T
    np.fill_diagonal(sim, -1.0)
    nn_sim = sim.max(axis=1)        # nearest-neighbour similarity per frame
    nn_idx = sim.argmax(axis=1)
    novelty = 1.0 - nn_sim          # high = far from everything = coverage gap

    novel_order = np.argsort(-novelty)[:top]
    novel = [{"frame_id": ids[i], "novelty": round(float(novelty[i]), 3),
              "image_url": f"/api/frames/{ids[i]}/image"} for i in novel_order]

    # near-duplicate pairs (each unordered pair once), most-similar first
    dup_pairs = []
    seen = set()
    for i in np.argsort(-nn_sim):
        if nn_sim[i] < dup_threshold:
            break
        j = int(nn_idx[i])
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        dup_pairs.append({"a": ids[i], "b": ids[j], "sim": round(float(nn_sim[i]), 3),
                          "a_url": f"/api/frames/{ids[i]}/image", "b_url": f"/api/frames/{ids[j]}/image"})
        if len(dup_pairs) >= top:
            break

    dup_count = int((nn_sim >= dup_threshold).sum())
    return {
        "total_frames": int(total_frames), "embedded": len(ids),
        "embedded_pct": round(100.0 * len(ids) / total_frames, 1) if total_frames else 0.0,
        "mean_nn_sim": round(float(nn_sim.mean()), 3), "duplicate_frames": dup_count,
        "novel": novel, "duplicates": dup_pairs,
    }


async def diverse_sample(session_id: str | None = None, k: int = 50) -> list[dict]:
    ids, mat = await _load_matrix(session_id)
    if not ids:
        return []
    idx = _farthest_point_sample(mat, k)
    return [{"frame_id": ids[i], "image_url": f"/api/frames/{ids[i]}/image"} for i in idx]
