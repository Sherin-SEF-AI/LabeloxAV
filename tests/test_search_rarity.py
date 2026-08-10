"""Search ranked by match alone, on a corpus where match alone is nearly uninformative.

Counted over the live tables: 34,132 frames carry objects, across 152 classes. The most common class is on
30,134 of those frames, 88% of them. The rarest is on one. A query broad enough to be useful for mining
("vehicle at a junction") is close to tens of thousands of frames at once, and the top k comes back full of
the commonest thing in the corpus, because that is what the corpus is made of.

The frame worth labelling is in the pool and just under the cut. The review queue already ranks on
uncertainty times rarity; search had no access to the idea at all.

These tests pin the parts where a rarity weight goes wrong. It must not overwhelm the query, it must not
filter, it must not average away the rare object it exists to find, and it must be visible in the response,
because a result order the caller cannot account for is one they cannot trust.
"""

from __future__ import annotations

import pytest

from services.intelligence.search.rarity import (
    DEFAULT_RARITY_WEIGHT,
    blend,
    build_idf,
    frame_rarity,
    idf,
    rerank,
)

# The real shape: 34,132 frames, the commonest class on 30,134 of them, the rarest on one.
TOTAL = 34_132
COMMON, MID, RARE = 11, 40, 206
COUNTS = {COMMON: 30_134, MID: 1_200, RARE: 1}
IDF = build_idf(COUNTS, TOTAL)


# ------------------------------------------------------------------------------- idf

def test_a_rare_class_scores_above_a_common_one():
    assert IDF[RARE] > IDF[MID] > IDF[COMMON]


def test_a_class_on_almost_every_frame_is_not_erased_entirely():
    """Zero would make the commonest class weightless rather than merely cheap, and a frame holding only it
    would rank identically to a frame holding nothing at all."""
    assert IDF[COMMON] > 0.0


def test_scores_stay_bounded_despite_a_thirty_thousand_fold_spread():
    """Unnormalised, the frequency range in this corpus becomes the score range, and rarity stops being a
    weight and starts being the whole ranking."""
    assert 0.0 <= IDF[COMMON] <= 1.0
    assert 0.0 <= IDF[RARE] <= 1.0


def test_an_unseen_class_does_not_divide_by_zero():
    assert idf(0, TOTAL) > 0
    assert idf(5, 0) == 0.0


# ------------------------------------------------------------------------------- per frame

def test_a_frame_is_worth_the_rarest_thing_on_it():
    """A cattle crossing on a road full of sedans is a cattle frame. A mean would let the sedans bury it,
    which is the failure this exists to fix."""
    crowded = frame_rarity([COMMON] * 12 + [RARE], IDF)
    assert crowded == pytest.approx(IDF[RARE])


def test_a_frame_with_nothing_on_it_has_no_rarity():
    assert frame_rarity([], IDF) == 0.0


def test_an_unknown_class_id_does_not_raise():
    assert frame_rarity([99999], IDF) == 0.0


# ------------------------------------------------------------------------------- the blend

def test_the_blend_stays_in_the_same_units_as_the_score_it_replaces():
    """A caller comparing scores across two searches must not be comparing different things."""
    for w in (0.0, 0.25, 1.0):
        assert 0.0 <= blend(0.8, 0.9, w) <= 1.0


def test_a_near_tie_on_similarity_is_broken_by_rarity():
    """The whole point, at the margin where it should act."""
    common_frame = blend(0.81, IDF[COMMON], DEFAULT_RARITY_WEIGHT)
    rare_frame = blend(0.80, IDF[RARE], DEFAULT_RARITY_WEIGHT)
    assert rare_frame > common_frame


def test_a_strong_match_is_not_displaced_by_a_rare_irrelevance():
    """At the default weight this must remain a search. A rare frame that barely matches the query has to
    stay below a frame that plainly does, or the box stops answering the question asked of it."""
    strong = blend(0.95, IDF[COMMON], DEFAULT_RARITY_WEIGHT)
    weak_but_rare = blend(0.40, IDF[RARE], DEFAULT_RARITY_WEIGHT)
    assert strong > weak_but_rare


def test_the_weight_is_clamped_rather_than_trusted():
    assert blend(0.5, 1.0, 5.0) == blend(0.5, 1.0, 1.0)
    assert blend(0.5, 1.0, -2.0) == blend(0.5, 1.0, 0.0)


# ------------------------------------------------------------------------------- reranking

def _results():
    return [{"frame_id": "f-common", "score": 0.81}, {"frame_id": "f-rare", "score": 0.80},
            {"frame_id": "f-mid", "score": 0.79}]


def test_reranking_promotes_the_rare_frame_and_keeps_every_result():
    """It reorders. It must never filter: a search that answers a different question than the one asked is
    not more useful for being surprising."""
    out = rerank(_results(), {"f-common": IDF[COMMON], "f-rare": IDF[RARE], "f-mid": IDF[MID]}, 3)
    assert out[0]["frame_id"] == "f-rare"
    assert {r["frame_id"] for r in out} == {"f-common", "f-rare", "f-mid"}


def test_the_original_similarity_survives_the_blend():
    """So a caller can see what the query thought before rarity had its say."""
    out = rerank(_results(), {"f-rare": IDF[RARE]}, 3)
    assert next(r for r in out if r["frame_id"] == "f-rare")["semantic_score"] == 0.80


def test_weight_zero_changes_nothing_at_all():
    """The feature has to be switchable off, and off has to mean the previous behaviour exactly."""
    before = _results()
    out = rerank(before, {"f-rare": IDF[RARE]}, 3, weight=0.0)
    assert [r["frame_id"] for r in out] == [r["frame_id"] for r in before]
    assert [r["score"] for r in out] == [r["score"] for r in before]


def test_reranking_truncates_to_what_was_asked_for():
    out = rerank(_results(), {"f-rare": IDF[RARE]}, 2)
    assert len(out) == 2


def test_a_frame_with_no_known_objects_still_ranks():
    """Frames with no labels yet are exactly the ones a mining query wants to be able to surface."""
    out = rerank(_results(), {}, 3)
    assert len(out) == 3 and all(r["rarity"] == 0.0 for r in out)
