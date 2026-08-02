"""Every quality number in this system was a bare point estimate.

`measured_precision` returns a fraction, `honeypot_accuracy` a ratio, the overnight auditor samples a
hardcoded 200. None says how sure it is, so "precision is 0.87" reads the same whether it came from 12
objects or 12,000, and the interval is most of what a quality claim is actually worth.

These tests cover the two places that would mislead rather than merely approximate: an interval that flatters
at the extremes, and an accept/reject decision made on a point estimate the sample cannot support.
"""

import pytest

from services.labelops.sampling import acceptance_decision, sample_size_for, wilson_interval


def test_nothing_sampled_is_unknown_not_zero():
    """Reporting an unchecked batch as 0.0 is a lie in the confident direction."""
    ci = wilson_interval(0, 0)
    assert ci["p"] is None
    assert (ci["lo"], ci["hi"]) == (0.0, 1.0)


def test_a_clean_small_sample_does_not_claim_certainty():
    """20 of 20 correct is not proof of perfection, and the normal approximation would say it is."""
    ci = wilson_interval(20, 20)
    assert ci["p"] == 1.0
    assert ci["lo"] < 0.9, f"a lower bound of {ci['lo']} on n=20 is overconfident"


def test_the_interval_never_leaves_zero_to_one():
    """The textbook normal interval produces negative bounds at these rates, which is why Wilson is used."""
    for s, n in ((0, 20), (1, 20), (19, 20), (20, 20), (0, 5), (5, 5)):
        ci = wilson_interval(s, n)
        assert 0.0 <= ci["lo"] <= ci["hi"] <= 1.0, (s, n, ci)


def test_more_samples_narrow_the_interval():
    wide = wilson_interval(45, 50)["half_width"]
    narrow = wilson_interval(450, 500)["half_width"]
    assert narrow < wide / 2


def test_the_planned_sample_size_matches_what_is_claimed():
    """The batch builder is sized on this: 300 judged objects pins a rate near 0.9 to about +/-3.4%."""
    assert wilson_interval(270, 300)["half_width"] < 0.04
    # And the planner agrees with the interval, rather than the two drifting apart.
    n = sample_size_for(0.034, expected_p=0.9)
    assert 250 <= n <= 400, n


def test_the_worst_case_is_the_default_planning_assumption():
    """p=0.5 needs the most samples, so planning against it cannot under-provision."""
    assert sample_size_for(0.05) >= sample_size_for(0.05, expected_p=0.9)


def test_a_batch_too_small_to_judge_says_so():
    """The answer a bare threshold cannot give, and the one that matters.

    One defect in 20 observes 5% against a 10% limit, which looks like a pass. The upper bound is near 24%,
    so the evidence does not support that call.
    """
    r = acceptance_decision(1, 20, max_defect_rate=0.10)
    assert r["verdict"] == "inconclusive"
    assert "would settle it" in r["reason"]


def test_a_clearly_clean_batch_is_accepted_on_the_upper_bound():
    r = acceptance_decision(6, 300, max_defect_rate=0.10)
    assert r["verdict"] == "accept"
    assert r["hi"] <= 0.10


def test_a_clearly_bad_batch_is_rejected_on_the_lower_bound():
    r = acceptance_decision(45, 300, max_defect_rate=0.10)
    assert r["verdict"] == "reject"
    assert r["lo"] > 0.10


def test_an_impossible_precision_target_is_refused():
    with pytest.raises(ValueError):
        sample_size_for(0.0)


def test_the_precision_batch_samples_randomly_not_by_value():
    """The distinction the whole thing rests on.

    The active-learning queue surfaces the hardest objects, which is right for improving a model and ruinous
    for measuring one: judging it gives the accuracy of the worst objects rather than of the corpus.
    """
    import inspect

    from services.labelops import precision_batch

    src = inspect.getsource(precision_batch.build_precision_batch)
    assert "func.random()" in src, "a precision sample must be random with respect to correctness"
    assert "score_candidates" not in src, "the value ranking must not choose the measurement sample"


def test_the_precision_batch_excludes_what_a_human_already_ruled_on():
    """Including reviewed objects measures agreement with past decisions, not accuracy of unreviewed ones."""
    import inspect

    from services.labelops import precision_batch

    assert "notin_(reviewed)" in inspect.getsource(precision_batch.build_precision_batch)


def test_the_review_link_carries_the_states_the_batch_actually_contains():
    """Triage defaults to review and annotate; this batch includes auto_accept on purpose.

    Without the states in the link the grid hands back a subset, and the precision figure silently excludes
    exactly the labels the gate was most confident about, which are the ones most worth checking. Caught on
    the live batch: 284 of 300 came back.
    """
    import inspect

    from services.labelops import precision_batch

    src = inspect.getsource(precision_batch.build_precision_batch)
    assert "states=" in src, "the review link must pin the states"
    assert "auto_accept" in precision_batch.MACHINE_STATES
