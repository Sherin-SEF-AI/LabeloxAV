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

from sqlalchemy import and_, exists, literal, select

from core.origin import REAL
from db.models import Frame, FrameEmbedding, Object, ObjectEmbedding


def frame_is_real():
    """SQL predicate: the frame's pixels came from a camera (core/origin.py).

    A composite from the copy-paste generator is never embedded: no GPU is spent on it, and it stays out
    of dedup, novelty and find-similar, all of which would otherwise pull it into a real neighbourhood.
    """
    return Frame.origin == REAL


def object_on_real_frame():
    """SQL predicate: the object's frame is real. Safe whether or not the caller also joins Frame.

    The explicit `select_from(Frame).correlate(Object)` is the whole point of this shape. Written as a
    bare `exists().where(...)`, SQLAlchemy decides the subquery's FROM by auto-correlation: against a
    query that selects from Object alone it correctly keeps Frame inside the subquery, but against a
    query that already joins Frame it correlates Frame outwards, leaves the subquery with no FROM at all
    and raises `InvalidRequestError` at compile time.

    That is exactly the difference between how the tests called it and how production did.
    `embed_objects` joins Frame to read `img_uri`, so the one caller that mattered raised on every run
    while the suite stayed green. Pinning the subquery's FROM makes the predicate mean the same thing in
    both shapes.
    """
    return exists(
        select(literal(1)).select_from(Frame)
        .where(Frame.frame_id == Object.frame_id, Frame.origin == REAL)
        .correlate(Object)
    )


def object_needs_embedding():
    """SQL predicate: this object has no embedding, or an incomplete one, and sits on a real frame."""
    return and_(object_on_real_frame(), ~exists().where(
        ObjectEmbedding.object_id == Object.object_id,
        ObjectEmbedding.dino_vec.isnot(None),
        ObjectEmbedding.siglip_vec.isnot(None),
    ))


def frame_needs_embedding():
    """SQL predicate: this frame has no embedding, or an incomplete one, and is real.

    Frames carry both vectors for the same reason objects do: DINOv3 backs find-similar and dedup, SigLIP2
    backs text search. A frame with only one of them is half indexed.
    """
    return and_(frame_is_real(), ~exists().where(
        FrameEmbedding.frame_id == Frame.frame_id,
        FrameEmbedding.dino_vec.isnot(None),
        FrameEmbedding.siglip_vec.isnot(None),
    ))


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
