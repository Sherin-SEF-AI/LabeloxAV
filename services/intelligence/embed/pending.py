"""What counts as "still needs embedding", in one place.

Two callers asked this question, the daemon and the backfill, and they answered it differently. Both tested
only whether an embedding *row* existed:

    ~exists().where(ObjectEmbedding.object_id == Object.object_id)

A row is not a vector. `object_embedding.siglip_vec` was added later, in migration 0073, as a nullable
column, so every crop embedded before it has a row with a DINOv3 vector and a NULL SigLIP2 one. Under a
row-existence test those crops are complete forever: the daemon skips them, `--only-missing` skips them, and
nothing counts them. Meanwhile `core/embeddings.py:object_neighbors_by_text` filters on
`siglip_vec IS NOT NULL`, so they are unreachable by text search.

The result on the live corpus was total and silent: all 570,305 object embeddings had a NULL `siglip_vec`,
so `GET /api/search/objects` returned zero results for every query, including exact ontology class names,
while every counter reported full coverage. Only `--reembed` could reach them, and nothing said to run it.

So the rule is stated once, here, and both callers import it. The question is not "is there a row" but "is
every vector this object needs actually present".
"""

from __future__ import annotations

from sqlalchemy import exists

from db.models import Frame, FrameEmbedding, Object, ObjectEmbedding


def object_needs_embedding():
    """SQL predicate: this object has no embedding, or an incomplete one."""
    return ~exists().where(
        ObjectEmbedding.object_id == Object.object_id,
        ObjectEmbedding.dino_vec.isnot(None),
        ObjectEmbedding.siglip_vec.isnot(None),
    )


def frame_needs_embedding():
    """SQL predicate: this frame has no embedding, or an incomplete one.

    Frames carry both vectors for the same reason objects do: DINOv3 backs find-similar and dedup, SigLIP2
    backs text search. A frame with only one of them is half indexed.
    """
    return ~exists().where(
        FrameEmbedding.frame_id == Frame.frame_id,
        FrameEmbedding.dino_vec.isnot(None),
        FrameEmbedding.siglip_vec.isnot(None),
    )


def object_missing_siglip():
    """SQL predicate: this object has a DINOv3 vector but no SigLIP2 one.

    Reported separately from the general backlog because it is the shape of the defect above: a population
    that every existing counter called complete. A non-zero count here with a zero pending count means the
    corpus is silently unreachable by text search, which is worth seeing as its own number rather than
    folded into a total that used to read zero.
    """
    return exists().where(
        ObjectEmbedding.object_id == Object.object_id,
        ObjectEmbedding.dino_vec.isnot(None),
        ObjectEmbedding.siglip_vec.is_(None),
    )


def frame_missing_siglip():
    """SQL predicate: this frame has a DINOv3 vector but no SigLIP2 one."""
    return exists().where(
        FrameEmbedding.frame_id == Frame.frame_id,
        FrameEmbedding.dino_vec.isnot(None),
        FrameEmbedding.siglip_vec.is_(None),
    )


__all__ = ["frame_missing_siglip", "frame_needs_embedding", "object_missing_siglip",
           "object_needs_embedding"]
