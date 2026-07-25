"""Find-similar reranking: the diversity dedup and score threshold that turn raw pgvector top-k into a set of
distinct, above-floor neighbours.

The bug these guard against is real and was observed on the corpus: a query object tracked across frames came
back as fifteen copies of itself at similarity 1.0, because raw nearest-neighbour is blind to duplicates. The
dedup collapses those; the threshold keeps the padding out. Both are pure over the candidate list, so they are
tested without a database.
"""

import numpy as np

from services.intelligence.search.similar import _dedupe_diverse


def _cand(oid, sim, vec):
    return {"object_id": oid, "sim": sim, "vec": vec, "class_id": 1, "track_id": None, "frame_id": "f"}


def test_dedupe_collapses_near_identical_vectors():
    # three copies of one direction, then a genuinely different one
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    cands = [
        _cand("a1", 1.00, a),
        _cand("a2", 0.999, [0.9999, 0.014, 0.0]),  # ~identical to a1
        _cand("a3", 0.998, [0.9998, 0.02, 0.0]),   # ~identical to a1
        _cand("b1", 0.80, b),                        # distinct
    ]
    kept = _dedupe_diverse(cands, k=10, dup_thresh=0.96)
    ids = [c["object_id"] for c in kept]
    # the near-identical a2/a3 are dropped; the distinct b1 survives
    assert ids == ["a1", "b1"]


def test_dedupe_keeps_the_highest_scoring_of_a_duplicate_cluster():
    v = [0.6, 0.8, 0.0]
    cands = [
        _cand("hi", 0.95, v),
        _cand("lo", 0.90, [0.61, 0.79, 0.02]),  # near-identical, lower score
    ]
    kept = _dedupe_diverse(cands, k=10, dup_thresh=0.96)
    # candidates arrive sorted by score, so the representative kept is the higher one
    assert [c["object_id"] for c in kept] == ["hi"]


def test_dedupe_leaves_distinct_candidates_untouched():
    cands = [
        _cand("x", 0.9, [1.0, 0.0, 0.0]),
        _cand("y", 0.8, [0.0, 1.0, 0.0]),
        _cand("z", 0.7, [0.0, 0.0, 1.0]),
    ]
    kept = _dedupe_diverse(cands, k=10, dup_thresh=0.96)
    assert [c["object_id"] for c in kept] == ["x", "y", "z"]


def test_dedupe_respects_k():
    cands = [_cand(str(i), 0.9 - i * 0.01, list(np.eye(1, 20, i)[0])) for i in range(20)]
    kept = _dedupe_diverse(cands, k=5, dup_thresh=0.96)
    assert len(kept) == 5


def test_dedupe_handles_a_zero_vector_without_dividing_by_zero():
    cands = [_cand("zero", 0.5, [0.0, 0.0, 0.0]), _cand("real", 0.4, [1.0, 0.0, 0.0])]
    kept = _dedupe_diverse(cands, k=10, dup_thresh=0.96)
    # a degenerate zero vector is never "identical" to anything, so both survive rather than crashing
    assert len(kept) == 2
