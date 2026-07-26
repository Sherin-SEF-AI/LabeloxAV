"""Find-similar with a reranking stage: diversity dedup, a score cutoff, and same-class / same-track filters.

Raw pgvector top-k is necessary but not sufficient for a labeling tool. Two problems show up on real data:

  1. Near-duplicate floods. The same physical object across consecutive frames (or an oversampled crop) has
     an almost-identical vector, so the top-k comes back as fifteen copies of one autorickshaw instead of
     fifteen different ones. That is technically correct and practically useless: you search to find *other*
     examples, not the same one again.
  2. No floor. Once the truly-similar results run out, pgvector keeps returning the next-nearest however far
     away they are, so a thin index pads the grid with unrelated crops at similarity 0.3.

This module over-fetches candidates (cheap on the HNSW index), then reranks in Python at rerank scale:
threshold on the score, drop the query's own track, and greedily skip any candidate too close to one already
chosen. The result is k *distinct* neighbours above a real similarity floor.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from core.embeddings import frame_candidates, object_candidates
from core.logging import get_logger

log = get_logger("search_similar")

# Over-fetch this many times k before reranking, so dedup and the threshold have room to work without a
# second round trip. Capped so a huge k cannot ask the index for tens of thousands of rows.
OVER_FETCH = 6
MAX_CANDIDATES = 600

# Two crops closer than this cosine are treated as the same thing for diversity. 0.96 keeps genuinely
# distinct-but-similar objects while collapsing the same object seen a frame apart.
DEFAULT_DUP_THRESH = 0.96


def _dedupe_diverse(cands: list[dict], k: int, dup_thresh: float) -> list[dict]:
    """Greedily keep the highest-scoring candidates, skipping any that duplicate one already kept.

    Candidates arrive already sorted by similarity (pgvector order), so the first time a new visual cluster
    appears its best representative is what we keep; later near-identical members of that cluster are the
    ones dropped, which is exactly the temporal-duplicate case.
    """
    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []
    for c in cands:
        v = np.asarray(c["vec"], dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        if any(float(v @ kv) > dup_thresh for kv in kept_vecs):
            continue
        kept.append(c)
        kept_vecs.append(v)
        if len(kept) >= k:
            break
    return kept


async def find_similar_objects(
    db: AsyncSession, query_vec, *, k: int = 24, min_sim: float = 0.0,
    class_id: int | None = None, exclude_object_id: UUID | None = None,
    exclude_track_id: str | None = None, diversity: bool = True,
    dup_thresh: float = DEFAULT_DUP_THRESH, session_id: UUID | None = None, city: str | None = None,
) -> list[dict]:
    """Reranked object neighbours: threshold, optional same-track drop, optional diversity dedup.

    Returns dicts with object_id / sim / class_id / track_id / frame_id, richest-first, at most k. `class_id`
    scopes the search to one class (the "same class only" toggle); `exclude_track_id` drops other crops of
    the very object you started from so you see different instances.
    """
    fetch = min(MAX_CANDIDATES, max(k, k * OVER_FETCH))
    cands = await object_candidates(db, query_vec, k=fetch, class_id=class_id,
                                    exclude_object_id=exclude_object_id, session_id=session_id, city=city)
    cands = [c for c in cands if c["sim"] >= min_sim]
    if exclude_track_id is not None:
        cands = [c for c in cands if c["track_id"] != exclude_track_id]
    picked = _dedupe_diverse(cands, k, dup_thresh) if diversity else cands[:k]
    log.info("similar.objects", fetched=len(cands), returned=len(picked), diversity=diversity, min_sim=min_sim)
    for c in picked:
        c.pop("vec", None)
    return picked


async def find_similar_frames(
    db: AsyncSession, query_vec, *, space: str = "dino", k: int = 24, min_sim: float = 0.0,
    exclude_frame_id: UUID | None = None, diversity: bool = True,
    dup_thresh: float = DEFAULT_DUP_THRESH, session_id: UUID | None = None, city: str | None = None,
) -> list[dict]:
    """Reranked frame neighbours. Same reranking as objects; consecutive frames of one drive are the dominant
    duplicate here, so diversity matters just as much."""
    fetch = min(MAX_CANDIDATES, max(k, k * OVER_FETCH))
    cands = await frame_candidates(db, query_vec, space=space, k=fetch,
                                   exclude_frame_id=exclude_frame_id, session_id=session_id, city=city)
    cands = [c for c in cands if c["sim"] >= min_sim]
    picked = _dedupe_diverse(cands, k, dup_thresh) if diversity else cands[:k]
    log.info("similar.frames", fetched=len(cands), returned=len(picked), diversity=diversity, min_sim=min_sim)
    for c in picked:
        c.pop("vec", None)
    return picked
