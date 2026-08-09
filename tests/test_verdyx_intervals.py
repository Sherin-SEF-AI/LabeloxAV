"""Rates that carry the uncertainty they actually have.

`evaluate_gold_patches` reported `precision: 0.334` and `recall: 0.556` computed over nine matched objects.
As point estimates those are indistinguishable from the same figures measured over nine thousand, and the
promotion gate compared them as though they were the same claim: it refused a challenger for "does not beat
champion mAP (0.142 vs 0.169)" on a gold set where one object moves mAP by roughly ten points.

Wilson rather than the textbook normal approximation, because the approximation is wrong in exactly this
corpus's situation. At small n it puts bounds outside [0, 1], and at p = 0 or p = 1 it collapses to zero
width, so a class with 0 of 6 recalled would report "0.000, and we are certain" when almost nothing is known.
That case has a test of its own.
"""

from __future__ import annotations

import pytest

from services.verdyx.intervals import (
    Interval,
    annotate_per_class,
    compare,
    from_counts,
    required_n,
    separated,
    wilson,
)


def test_an_interval_contains_its_point_estimate():
    iv = wilson(7, 10)
    assert iv.low <= iv.point <= iv.high
    assert iv.point == 0.7 and iv.n == 10


def test_a_small_sample_is_wide_and_a_large_one_is_narrow():
    """The property that makes the number mean something."""
    assert wilson(5, 10).width > wilson(500, 1000).width


def test_zero_successes_still_carries_uncertainty():
    """The case the normal approximation gets dangerously wrong. A safety class with 0 of 6 recalled is not
    'recall 0.0, certain'; it is 'recall could plausibly be up to about 0.4 and we have barely looked'. The
    promotion gate applies its floors per class, so this is where a false certainty blocks or passes a model
    on almost no evidence."""
    iv = wilson(0, 6)
    assert iv.point == 0.0
    assert iv.low == 0.0
    assert iv.high > 0.25, "zero of six cannot honestly claim the rate is near zero"


def test_perfect_score_still_carries_uncertainty():
    iv = wilson(6, 6)
    assert iv.point == 1.0 and iv.high == 1.0
    assert iv.low < 0.75, "six of six is not proof of a perfect detector"


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (0, 3), (3, 3), (1, 2), (50, 100), (0, 10000)])
def test_bounds_never_leave_the_unit_interval(k, n):
    """The normal approximation routinely produces negative lower bounds at small n, which then render as a
    negative recall."""
    iv = wilson(k, n)
    assert 0.0 <= iv.low <= iv.high <= 1.0


def test_nothing_measured_reports_the_whole_range_not_zero():
    """A 0.0 with no interval reads as a measured failure. It was not measured at all."""
    iv = wilson(0, 0)
    assert (iv.low, iv.high, iv.n) == (0.0, 1.0, 0)


def test_successes_cannot_exceed_the_sample():
    assert wilson(20, 10).point == 1.0


# ------------------------------------------------------------------------------- sample size

def test_required_n_grows_as_the_target_margin_shrinks():
    """The answer to 'what would plus or minus 0.02 cost', which is the thing a buyer is actually purchasing."""
    assert required_n(0.02) > required_n(0.05) > required_n(0.10)


def test_required_n_uses_the_worst_case_by_default():
    """p = 0.5 maximises variance, so the default answer suffices whatever the true rate turns out to be."""
    assert required_n(0.05) >= required_n(0.05, p=0.9)


def test_a_meaningless_margin_is_refused():
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError):
            required_n(bad)


def test_the_current_gold_set_is_visibly_too_small():
    """302 objects is not enough for a plus or minus 0.02 claim, and saying so is the point."""
    assert required_n(0.02) > 302


# ------------------------------------------------------------------------------- comparison

def test_two_rates_on_tiny_samples_are_not_separable():
    """The exact situation the gate was in: 0.142 against 0.169 on single-digit denominators."""
    out = compare(wilson(1, 7), wilson(2, 7))
    assert out["decisive"] is False
    assert "cannot separate" in out["detail"]


def test_the_same_gap_becomes_decisive_with_enough_evidence():
    """It is the sample that changed, not the models: the same rates, measured properly."""
    assert compare(wilson(100, 700), wilson(200, 700))["decisive"] is True


def test_separation_is_symmetric():
    a, b = wilson(90, 100), wilson(10, 100)
    assert separated(a, b) and separated(b, a)


def test_overlap_is_not_a_claim_of_equality():
    """Deliberately weaker than 'they are the same': it says this sample cannot tell them apart, which is
    what a promotion decision needs to know."""
    out = compare(wilson(5, 10), wilson(6, 10))
    assert out["decisive"] is False
    assert "cannot separate" in out["detail"] and "equal" not in out["detail"]


# ------------------------------------------------------------------------------- from an evaluation

def test_precision_and_recall_get_their_own_denominators():
    """Precision is over predictions and recall over ground truth. A single n printed beside both would be
    wrong for one of them."""
    out = from_counts(tp=8, fp=2, fn=12)
    assert out["precision"]["n"] == 10
    assert out["recall"]["n"] == 20


def test_per_class_intervals_skip_classes_with_no_gold():
    """A class with no instances has no rate. Reporting 0.0 for it would put a fabricated failure into a
    per-class safety check."""
    out = annotate_per_class({"truck": 3}, {"truck": 6, "cattle": 0})
    assert "cattle" not in out
    assert out["truck"]["n"] == 6


def test_per_class_intervals_are_wide_where_the_classes_are_rare():
    """The real DashLab result: autorickshaw recalled 0 of 3, reported as a flat 0.000."""
    out = annotate_per_class({"autorickshaw": 0}, {"autorickshaw": 3})
    assert out["autorickshaw"]["value"] == 0.0
    assert out["autorickshaw"]["high"] > 0.4, "three instances cannot establish that recall is near zero"


def test_the_dict_shape_carries_everything_a_reader_needs():
    d = wilson(7, 10).as_dict()
    assert set(d) == {"value", "low", "high", "n", "margin"}
    assert d["margin"] == pytest.approx((d["high"] - d["low"]) / 2, abs=1e-4)


def test_str_reads_the_way_a_result_should_be_quoted():
    assert str(Interval(0.87, 0.83, 0.91, 180)) == "0.870 [0.830, 0.910] n=180"
