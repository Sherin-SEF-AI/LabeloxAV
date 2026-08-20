"""Hierarchical AP: charging for how wrong a confusion is, not merely that one happened.

Flat AP scores a scooter called a motorcycle and a scooter called a truck identically, which makes it
unable to separate two models that differ mainly in which mistakes they make. The fixtures below are small
enough that each level's answer is countable: four boxes, perfect localisation, and only the labels
change.
"""

from __future__ import annotations

import numpy as np

from core.accel.hier_ap import hierarchical_ap

# leaf 1 and 2 are both two_wheeler; leaf 3 is a four_wheeler. Everything is an object at the root.
_LEVELS = {
    "leaf": {1: "scooter", 2: "motorcycle", 3: "truck"},
    "l1": {1: "two_wheeler", 2: "two_wheeler", 3: "four_wheeler"},
    "root": {1: "object", 2: "object", 3: "object"},
}


def _boxes(n: int) -> np.ndarray:
    # Disjoint, so every prediction either matches its own gold box exactly or matches nothing.
    return np.array([[i * 100.0, 0.0, i * 100.0 + 50.0, 50.0] for i in range(n)])


class TestTheLevels:
    def test_a_perfect_detector_scores_one_everywhere(self):
        b = _boxes(3)
        out = hierarchical_ap(b, [0.9, 0.9, 0.9], [1, 2, 3], b, [1, 2, 3], levels=_LEVELS,
                              iou_thresholds=(0.5,))
        for name in _LEVELS:
            assert abs(out["levels"][name].ap50 - 1.0) < 1e-9, name
        assert all(v == 0.0 for v in out["gap"].values())

    def test_a_confusion_within_a_superclass_is_free_one_level_up(self):
        """The whole claim. Every scooter is called a motorcycle: zero at the leaf, perfect at l1.

        Flat AP reports the same zero here as it would if they had been called trucks, and those are not
        the same event.
        """
        b = _boxes(3)
        out = hierarchical_ap(b, [0.9, 0.9, 0.9], [2, 2, 2], b, [1, 1, 1], levels=_LEVELS,
                              iou_thresholds=(0.5,))
        assert out["levels"]["leaf"].ap50 == 0.0
        assert abs(out["levels"]["l1"].ap50 - 1.0) < 1e-9
        assert abs(out["gap"]["l1"] - 1.0) < 1e-9

    def test_a_confusion_across_a_superclass_is_not_free(self):
        """Scooters called trucks: zero at the leaf AND at l1, and only the root forgives it."""
        b = _boxes(3)
        out = hierarchical_ap(b, [0.9, 0.9, 0.9], [3, 3, 3], b, [1, 1, 1], levels=_LEVELS,
                              iou_thresholds=(0.5,))
        assert out["levels"]["leaf"].ap50 == 0.0
        assert out["levels"]["l1"].ap50 == 0.0
        assert abs(out["levels"]["root"].ap50 - 1.0) < 1e-9

    def test_a_missed_object_is_missed_at_every_level(self):
        """Naming cannot rescue a detection that never happened, which is why AP is recomputed per level
        rather than reweighting a confusion matrix: a matrix only sees detections that matched."""
        b = _boxes(3)
        out = hierarchical_ap(b[:1], [0.9], [1], b, [1, 1, 1], levels=_LEVELS, iou_thresholds=(0.5,))
        for name in _LEVELS:
            assert out["levels"][name].ap50 < 0.5, name

    def test_the_vocabulary_shrinks_as_the_level_coarsens(self):
        b = _boxes(3)
        out = hierarchical_ap(b, [0.9, 0.9, 0.9], [1, 2, 3], b, [1, 2, 3], levels=_LEVELS,
                              iou_thresholds=(0.5,))
        assert out["levels"]["leaf"].n_classes == 3
        assert out["levels"]["l1"].n_classes == 2
        assert out["levels"]["root"].n_classes == 1

    def test_a_class_the_tree_forgot_keeps_its_own_identity(self):
        """Pooling it into a catch-all would make a level's AP depend on which classes the tree forgot.

        An incomplete tree would then look like a better model, which is the wrong direction for an
        omission to push a metric.
        """
        b = _boxes(2)
        levels = {"l1": {1: "two_wheeler"}}          # class 9 is missing from the mapping
        out = hierarchical_ap(b, [0.9, 0.9], [1, 9], b, [1, 9], levels=levels, iou_thresholds=(0.5,))
        assert out["levels"]["l1"].n_classes == 2, "the forgotten class was pooled away"
        assert abs(out["levels"]["l1"].ap50 - 1.0) < 1e-9

    def test_no_ground_truth_is_unmeasured_rather_than_zero(self):
        out = hierarchical_ap(_boxes(2), [0.9, 0.9], [1, 2], np.zeros((0, 4)), [], levels=_LEVELS)
        for name in _LEVELS:
            assert out["levels"][name].measured is False
            assert out["levels"][name].reason

    def test_mismatched_inputs_raise(self):
        for bad in (lambda: hierarchical_ap(_boxes(2), [0.9], [1, 2], _boxes(2), [1, 2], levels=_LEVELS),
                    lambda: hierarchical_ap(_boxes(2), [0.9, 0.9], [1, 2], _boxes(2), [1], levels=_LEVELS)):
            try:
                bad()
            except ValueError:
                continue
            raise AssertionError("expected a ValueError")


def test_the_gap_is_measured_against_the_leaf_and_not_between_coarse_levels():
    """Each level macro-averages over a different vocabulary, so two coarse levels are not comparable.

    On the champion's real run the levels come out leaf 0.072, l1 0.143, l0 0.088, root 0.190: l0 sits
    below l1 because averaging over three labels weights one weak label far more than averaging over
    twelve does. Comparing them to each other would read as a defect and is not one, so the gap is always
    against the leaf.
    """
    b = _boxes(3)
    out = hierarchical_ap(b, [0.9, 0.9, 0.9], [2, 2, 2], b, [1, 1, 1], levels=_LEVELS,
                          iou_thresholds=(0.5,))
    finest = out["levels"]["leaf"]
    for name, lv in out["levels"].items():
        if lv.ap50 is None:
            continue
        assert abs(out["gap"][name] - (lv.ap50 - finest.ap50)) < 1e-9, name
