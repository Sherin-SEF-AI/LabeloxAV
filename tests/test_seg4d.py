"""Milestone G: 4D semantic consistency. The temporal majority filter corrects an isolated flicker but keeps
a sustained transition; the consistency metric reports the dominant label, its hold fraction, and the
flicker count."""

from __future__ import annotations

from services.temporal.seg4d import label_consistency, temporal_majority_filter

A, B = "building", "wall"


def test_isolated_flicker_is_corrected():
    assert temporal_majority_filter([A, A, B, A, A], window=2) == [A, A, A, A, A]


def test_sustained_transition_survives():
    assert temporal_majority_filter([A, A, A, B, B, B], window=2) == [A, A, A, B, B, B]


def test_tie_keeps_original_label():
    # window of exactly two distinct labels ties -> no arbitrary flip
    assert temporal_majority_filter([A, B], window=1) == [A, B]


def test_consistency_metric():
    c = label_consistency([A, A, B, A])
    assert c["dominant"] == A and c["consistency"] == 0.75 and c["transitions"] == 2


def test_empty_sequence():
    c = label_consistency([])
    assert c["dominant"] is None and c["consistency"] == 0.0 and c["transitions"] == 0
