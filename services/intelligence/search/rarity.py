"""Ranking a search by how much a frame would teach, not only by how well it matches.

Semantic search returns the frames closest to the query and stops there, which is the right answer for
"find me this" and the wrong one for the thing this corpus is actually searched for: what to label next.

The spread is what makes it matter. Across 34,132 frames carrying objects there are 152 classes, the most
common appearing on 30,134 frames (88% of them) and the rarest on exactly one. A query like "vehicle at a
junction" matches tens of thousands of frames almost equally well, and the top k comes back full of sedans
because sedans are what the corpus is made of. The frame in that same pool holding an autorickshaw or a
cattle crossing scores a hair lower on cosine distance and is never seen, even though labelling it is worth
far more to the model.

So the semantic score is blended with an inverse-document-frequency term over the classes each frame already
carries. This is the same idea the review queue already ranks on (uncertainty times rarity); search simply
had no access to it.

Two things this deliberately does not do. It does not filter: a rare frame that does not match the query
stays out, because a search that answers a different question than the one asked is not more useful for
being surprising. And it does not hide itself: the weight and each frame's rarity ride in the response, for
the same reason `filtered` and `candidates` already do. A result order the caller cannot account for is one
they cannot trust.
"""

from __future__ import annotations

import math
import time

# How much rarity may move the ranking by default. Small on purpose: at 1.0 the query stops mattering and
# this becomes a rare-class browser wearing a search box. Measured against the corpus, 0.25 is enough to
# lift a one-in-thirty-thousand class over a near-tie on cosine distance without displacing a strong match.
DEFAULT_RARITY_WEIGHT = 0.25

# The pool pulled from the vector index before blending. Rarity cannot promote what was never fetched, so
# ranking the top k alone would leave the weight with nothing to do; this is what gives it reach.
POOL_FACTOR = 6
MAX_POOL = 600

# Class frequencies move only as fast as labelling does, and the count is a full scan of `object`.
_CACHE_TTL_S = 300.0
_cache: dict = {"at": 0.0, "idf": None, "total": 0}


def idf(frame_count: int, total_frames: int) -> float:
    """Inverse document frequency for a class, normalised to roughly [0, 1].

    Smoothed so a class on every single frame scores above zero rather than being erased, and so a class on
    zero frames cannot divide by anything. The log is what keeps the 30,000x frequency spread in this corpus
    from turning into a 30,000x score spread, which would let rarity overwhelm the query outright.
    """
    if total_frames <= 0:
        return 0.0
    c = max(0, int(frame_count))
    raw = math.log((total_frames + 1.0) / (c + 1.0))
    ceiling = math.log(total_frames + 1.0)
    return round(float(raw / ceiling), 6) if ceiling > 0 else 0.0


def build_idf(counts: dict[int, int], total_frames: int) -> dict[int, float]:
    """Per-class idf from {class_id: frames carrying it}."""
    return {int(cid): idf(n, total_frames) for cid, n in counts.items()}


def frame_rarity(class_ids: list[int] | set[int], idf_map: dict[int, float]) -> float:
    """How rare the rarest thing on a frame is.

    The maximum rather than the mean, because a frame's value is set by the most unusual thing on it. A
    cattle crossing on a road full of sedans is a cattle frame; averaging would let the sedans bury it, which
    is the exact failure this is here to fix.
    """
    vals = [idf_map.get(int(c), 0.0) for c in (class_ids or [])]
    return round(max(vals), 6) if vals else 0.0


def blend(semantic: float, rarity: float, weight: float = DEFAULT_RARITY_WEIGHT) -> float:
    """Combine a similarity in [0,1] with a rarity in [0,1].

    A convex combination, so the result stays in the same range as the semantic score it replaces and a
    caller comparing scores across two searches is not comparing different units.
    """
    w = min(1.0, max(0.0, float(weight)))
    return round(float((1.0 - w) * float(semantic) + w * float(rarity)), 6)


def rerank(results: list[dict], rarity_by_frame: dict[str, float], k: int,
           weight: float = DEFAULT_RARITY_WEIGHT) -> list[dict]:
    """Blend, sort, and truncate, annotating every result with what moved it.

    At weight 0 this is a no-op that preserves the incoming order exactly, which is what makes the feature
    safe to turn off and what the test pins.
    """
    out = []
    for r in results:
        rare = float(rarity_by_frame.get(r["frame_id"], 0.0))
        out.append({**r, "rarity": round(rare, 4),
                    "score": blend(r["score"], rare, weight) if weight > 0 else r["score"],
                    "semantic_score": r["score"]})
    if weight > 0:
        out.sort(key=lambda r: -r["score"])
    return out[:k]


async def class_frame_counts(db) -> tuple[dict[int, float], int]:
    """Corpus-wide per-class frame counts as idf, cached.

    Cached because this is a group-by over half a million object rows and a search must not pay for it on
    every keystroke, and because the answer changes on the timescale of a labelling session rather than a
    request.
    """
    now = time.monotonic()
    if _cache["idf"] is not None and (now - _cache["at"]) < _CACHE_TTL_S:
        return _cache["idf"], _cache["total"]

    from sqlalchemy import text

    total = int((await db.execute(text("select count(distinct frame_id) from object"))).scalar() or 0)
    rows = (await db.execute(
        text("select class_id, count(distinct frame_id) from object group by 1"))).all()
    idf_map = build_idf({int(c): int(n) for c, n in rows if c is not None}, total)
    _cache.update({"at": now, "idf": idf_map, "total": total})
    return idf_map, total


def reset_cache() -> None:
    """For tests and for a caller that has just imported a corpus."""
    _cache.update({"at": 0.0, "idf": None, "total": 0})
