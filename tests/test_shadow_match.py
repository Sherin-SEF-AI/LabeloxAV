"""The disagreement matcher, on boxes rather than on a database.

The matcher is where shadow mode can be wrong in a way nothing downstream would catch: a pairing bug
turns one class flip into two misses, and a threshold bug turns the whole low-confidence tail of one
model into evidence against the other. Both are silent, both produce plausible-looking counts, and both
would reach the promotion gate as fact. So they are tested here, in the pure layer, exhaustively.
"""

from __future__ import annotations

import uuid

import pytest

from services.verdyx.shadow_run import (
    CHALLENGER_MISS,
    CHAMPION_MISS,
    CLASS_FLIP,
    CONF_GAP,
    Det,
    compare_frame,
)


def d(cls: int, box: list[float], conf: float) -> Det:
    return Det(prediction_id=uuid.uuid4(), class_id=cls, bbox=box, conf=conf)


BOX = [100.0, 100.0, 200.0, 200.0]
FAR = [500.0, 500.0, 600.0, 600.0]


class TestTheFourKinds:
    def test_agreement_produces_nothing(self):
        assert compare_frame([d(1, BOX, 0.9)], [d(1, BOX, 0.9)], threshold=0.5) == []

    def test_a_box_only_the_challenger_has_is_a_champion_miss(self):
        out = compare_frame([], [d(1, BOX, 0.9)], threshold=0.5)
        assert [x.kind for x in out] == [CHAMPION_MISS]
        assert out[0].score == pytest.approx(0.9)
        assert out[0].champion is None and out[0].challenger is not None

    def test_a_box_only_the_champion_has_is_a_challenger_miss(self):
        out = compare_frame([d(1, BOX, 0.8)], [], threshold=0.5)
        assert [x.kind for x in out] == [CHALLENGER_MISS]
        assert out[0].challenger is None and out[0].champion is not None

    def test_the_same_box_read_as_two_classes_is_one_flip_not_two_misses(self):
        out = compare_frame([d(1, BOX, 0.9)], [d(2, BOX, 0.8)], threshold=0.5)
        assert [x.kind for x in out] == [CLASS_FLIP]
        # Scored by the lower confidence: the interesting flips are the ones both models are sure about.
        assert out[0].score == pytest.approx(0.8)
        assert out[0].iou == pytest.approx(1.0)

    def test_the_same_box_and_class_at_very_different_confidence_is_a_gap(self):
        out = compare_frame([d(1, BOX, 0.95)], [d(1, BOX, 0.55)], threshold=0.5)
        assert [x.kind for x in out] == [CONF_GAP]
        assert out[0].conf_gap == pytest.approx(0.40)

    def test_a_small_confidence_difference_is_agreement(self):
        assert compare_frame([d(1, BOX, 0.90)], [d(1, BOX, 0.80)], threshold=0.5) == []


class TestTheOperatingPoint:
    def test_the_low_confidence_tail_is_not_a_disagreement(self):
        """Inference writes down to 0.001 so a PR curve can be drawn. Comparing raw floors would make
        every tail detection of one model a miss by the other, which is an artefact of the floor rather
        than a disagreement between the models."""
        assert compare_frame([], [d(1, BOX, 0.2)], threshold=0.5) == []

    def test_a_box_that_crosses_the_cut_for_one_model_only_is_a_miss(self):
        out = compare_frame([d(1, BOX, 0.4)], [d(1, BOX, 0.9)], threshold=0.5)
        assert [x.kind for x in out] == [CHAMPION_MISS]

    def test_raising_the_threshold_can_only_remove_disagreements(self):
        champ = [d(1, BOX, 0.9), d(2, FAR, 0.6)]
        chall = [d(1, BOX, 0.55)]
        low = compare_frame(champ, chall, threshold=0.5)
        high = compare_frame(champ, chall, threshold=0.7)
        assert len(high) <= len(low)


class TestPairing:
    def test_pairing_is_symmetric_in_the_two_models(self):
        """Swapping the two models must swap every verdict and change nothing else. A pairing that
        depended on argument order would make the champion's misses and the challenger's incomparable."""
        champ = [d(1, BOX, 0.9), d(3, FAR, 0.7)]
        chall = [d(2, BOX, 0.85)]
        a = compare_frame(champ, chall, threshold=0.5)
        b = compare_frame(chall, champ, threshold=0.5)
        flip = {CHAMPION_MISS: CHALLENGER_MISS, CHALLENGER_MISS: CHAMPION_MISS,
                CLASS_FLIP: CLASS_FLIP, CONF_GAP: CONF_GAP}
        assert sorted(flip[x.kind] for x in a) == sorted(x.kind for x in b)

    def test_each_box_is_used_at_most_once(self):
        """Two overlapping challenger boxes on one champion box must not both pair with it, or one
        detection would be counted as agreeing and disagreeing at the same time."""
        champ = [d(1, BOX, 0.9)]
        chall = [d(1, [102.0, 102.0, 198.0, 198.0], 0.9), d(1, [104.0, 104.0, 196.0, 196.0], 0.9)]
        out = compare_frame(champ, chall, threshold=0.5)
        assert [x.kind for x in out] == [CHAMPION_MISS]

    def test_boxes_that_do_not_overlap_enough_are_two_misses(self):
        out = compare_frame([d(1, BOX, 0.9)], [d(1, FAR, 0.9)], threshold=0.5)
        assert sorted(x.kind for x in out) == sorted([CHAMPION_MISS, CHALLENGER_MISS])

    def test_the_worst_disagreement_comes_first(self):
        out = compare_frame([d(1, FAR, 0.6)], [d(2, BOX, 0.99)], threshold=0.5)
        assert out == sorted(out, key=lambda x: -x.score)


class TestSharedVocabulary:
    """A class one model was never trained on is not a disagreement about the picture.

    The first real sweep on this corpus compared a 12-class champion with a 9-class challenger and
    produced 3,091 "the challenger missed this" rows out of 3,240. The challenger had not missed
    anything: it had never been taught the word. Counting those as evidence would make a win share
    a measure of vocabulary size.
    """

    def test_a_class_the_other_model_cannot_emit_is_not_a_miss(self):
        champ = [d(1, BOX, 0.9), d(7, FAR, 0.9)]
        out = compare_frame(champ, [d(1, BOX, 0.9)], threshold=0.5, shared_classes={1})
        assert out == []

    def test_a_shared_class_still_disagrees(self):
        champ = [d(1, BOX, 0.9), d(7, FAR, 0.9)]
        out = compare_frame(champ, [], threshold=0.5, shared_classes={1, 7})
        assert sorted(x.kind for x in out) == [CHALLENGER_MISS, CHALLENGER_MISS]

    def test_none_compares_everything_the_way_it_used_to(self):
        champ = [d(7, FAR, 0.9)]
        assert len(compare_frame(champ, [], threshold=0.5, shared_classes=None)) == 1

    def test_an_empty_shared_vocabulary_yields_no_comparison(self):
        """Two models with nothing in common cannot be compared, and saying so is better than
        reporting every detection either made as evidence against the other."""
        assert compare_frame([d(1, BOX, 0.9)], [d(2, FAR, 0.9)], threshold=0.5,
                             shared_classes=set()) == []
