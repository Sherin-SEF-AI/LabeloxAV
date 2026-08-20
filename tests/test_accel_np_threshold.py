"""Neyman-Pearson thresholds, against outcomes small enough to count on paper.

The fixtures below are twenty-odd detections whose correct answer is worked out in the comments by walking
the sorted scores and dividing, so a wrong tail rule or an off-by-one in the accumulation shows up as a
different number rather than as a plausible one.

The interesting cases are not the happy path. They are the cut that a first-crossing rule would stop at
five detections too early, the class that cannot meet its bound at any threshold, and the sample too small
to have earned an operating point at all: each of those is a way to return a confident number that means
nothing, which is exactly what the hand-picked constants this replaces were already doing.
"""

from __future__ import annotations

import numpy as np

from core.accel.np_threshold import fit_per_class, np_threshold


def _pairs(spec: list[tuple[float, bool]]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([s for s, _ in spec], dtype=float),
            np.array([m for _, m in spec], dtype=bool))


class TestTheTailRule:
    def test_the_threshold_is_the_deepest_cut_that_holds_the_bound(self):
        """Twenty detections, alpha 0.20, and the answer walked by hand.

        Sorted descending, the false-accept rate of the accepted set at each cut is:

          rank  score  right?  accepted  wrong  FAR     within 0.20?
             1   0.99    Y         1       0    0.000   yes
             2   0.95    Y         2       0    0.000   yes
             3   0.90    Y         3       0    0.000   yes
             4   0.85    N         4       1    0.250   no
             5   0.80    Y         5       1    0.200   yes
             6   0.75    Y         6       1    0.167   yes
             7   0.70    Y         7       1    0.143   yes
             8   0.65    N         8       2    0.250   no
             9   0.60    Y         9       2    0.222   no
            10   0.55    Y        10       2    0.200   yes   <- deepest
            11   0.50    N        11       3    0.273   no
            12   0.45    Y        12       3    0.250   no
            13   0.40    N        13       4    0.308   no
            14   0.35    N        14       5    0.357   no
            15   0.30    Y        15       5    0.333   no
            16-20 all wrong, rising to 0.500

        The deepest cut still inside the bound is rank 10, so t = 0.55 and the gate accepts half the
        sample at a measured 0.200. Note rank 5 also sits at exactly 0.200: a rule that stopped at the
        first crossing would return 0.80 and accept 5, discarding five more correct detections without
        excluding a single error, since both errors above rank 10 are inside the accepted set either way.
        """
        s, m = _pairs([
            (0.99, True), (0.95, True), (0.90, True), (0.85, False), (0.80, True),
            (0.75, True), (0.70, True), (0.65, False), (0.60, True), (0.55, True),
            (0.50, False), (0.45, True), (0.40, False), (0.35, False), (0.30, True),
            (0.25, False), (0.20, False), (0.15, False), (0.10, False), (0.05, False),
        ])
        est = np_threshold(s, m, alpha=0.20, min_support=20, n_boot=200)
        assert est.measured is True, est.reason
        assert abs(est.threshold - 0.55) < 1e-9
        assert est.n_accept == 10
        assert abs(est.far_at - 0.20) < 1e-9
        assert abs(est.accept_rate - 0.50) < 1e-9

    def test_an_error_at_the_top_forbids_shallow_cuts_and_dilutes_with_depth(self):
        """One wrong detection at the very top, fifty-nine right ones below, alpha 0.10.

        At rank k the accepted set holds that single error, so FAR(k) = 1/k and the bound needs k >= 10.
        Every cut from rank 10 down qualifies and the deepest wins: the gate accepts all 60 at a measured
        1/60 = 0.0167. The shallow cuts are refused, which is the right answer and not a special case:
        accepting only the top three detections really would be wrong a third of the time.
        """
        s, m = _pairs([(0.99, False)] + [(0.9 - 0.01 * i, True) for i in range(59)])
        est = np_threshold(s, m, alpha=0.10, n_boot=200)
        assert est.measured is True, est.reason
        assert est.n_accept == 60
        assert abs(est.far_at - 1.0 / 60.0) < 1e-6      # far_at is stored to six places
        # The bound really is binding at the top: at 5% a single error needs twenty accepted to dilute,
        # which this sample has, but the first nineteen cuts are all still refused.
        assert np_threshold(s, m, alpha=0.005, n_boot=200).measured is False

    def test_tied_scores_are_accepted_or_rejected_together(self):
        """A threshold sits between distinct scores, never inside a run of equal ones.

        Six detections share 0.70, two of them wrong. A cut placed partway through that run would describe
        an accept rule the gate cannot implement: it thresholds on the score, so it takes all six or none.
        With alpha 0.10, taking all six pushes FAR to 2/10 = 0.200, and everything below is wrong, so the
        deepest admissible cut is the last score above the run.
        """
        spec = ([(0.9 - 0.01 * i, True) for i in range(4)]
                + [(0.70, True), (0.70, False), (0.70, True), (0.70, False), (0.70, True), (0.70, True)]
                + [(0.5 - 0.01 * i, False) for i in range(50)])
        est = np_threshold(*_pairs(spec), alpha=0.10, n_boot=200)
        assert est.measured is True
        assert est.threshold > 0.70
        assert est.n_accept == 4

    def test_a_perfect_class_takes_everything_it_can(self):
        # No errors anywhere, so the bound never binds and the deepest cut is the whole sample.
        est = np_threshold(*_pairs([(1.0 - 0.01 * i, True) for i in range(60)]), alpha=0.05, n_boot=200)
        assert est.measured is True
        assert est.n_accept == 60 and est.far_at == 0.0
        assert abs(est.accept_rate - 1.0) < 1e-9

    def test_a_tighter_alpha_never_accepts_more(self):
        rng = np.random.default_rng(7)
        s = np.sort(rng.uniform(0, 1, 400))[::-1]
        m = rng.uniform(0, 1, 400) < s          # correctness rises with score, as calibration intends
        counts = []
        for a in (0.30, 0.20, 0.10, 0.05):
            e = np_threshold(s, m, alpha=a, n_boot=200)
            counts.append(e.n_accept if e.measured else 0)
        assert counts == sorted(counts, reverse=True), counts


class TestWhatItRefusesToFit:
    def test_too_few_outcomes_is_a_refusal_not_a_strict_threshold(self):
        """Returning the top score would look like a cautious threshold and rest on nothing.

        A class with a handful of labelled examples has not earned an operating point, and the failure
        mode of pretending otherwise is silent: the gate would auto-accept at a number nobody measured,
        which is the defect this module exists to remove.
        """
        est = np_threshold(*_pairs([(0.9, True), (0.8, True), (0.7, False)]), alpha=0.10)
        assert est.measured is False and est.threshold is None
        assert "below the 50 needed" in est.reason
        assert est.n_pairs == 3

    def test_a_class_that_was_never_right_gets_no_threshold(self):
        est = np_threshold(*_pairs([(0.5 + 0.005 * i, False) for i in range(60)]), alpha=0.10)
        assert est.measured is False
        assert "no detection in this sample was correct" in est.reason

    def test_a_fragile_fit_reports_its_fragility_rather_than_hiding_it(self):
        """The point estimate can be real and still rest on which rows happened to be drawn.

        Exactly one of sixty detections is correct, and it is not the top-scoring one. On the original
        data the cut at rank 2 accepts one right and one wrong, FAR 0.500, which meets alpha 0.50 exactly.
        A resample only reproduces that if it draws the single correct row about as often as the one above
        it, so roughly half of them yield no threshold at all.

        The primitive does not refuse on that. Whether a threshold is located well enough to switch on is
        an activation decision and lives in the service layer, next to the other activation decisions. What
        this owes the caller is the evidence: an interval, and how many resamples it rests on.
        """
        spec = [(0.99, False), (0.98, True)] + [(0.9 - 0.01 * i, False) for i in range(58)]
        est = np_threshold(*_pairs(spec), alpha=0.50, n_boot=200)
        assert est.measured is True and est.threshold == 0.98
        assert 0 < est.n_boot_fit < 200, "most resamples should fail on data this thin"
        # A well-supported fit on the same-sized sample rests on all of them, which is the contrast the
        # activation decision is made on.
        solid = np_threshold(*_pairs([(1.0 - 0.004 * i, i % 5 != 4) for i in range(200)]),
                             alpha=0.20, n_boot=200)
        assert solid.n_boot_fit == 200

    def test_the_arguments_have_to_line_up(self):
        for bad in (lambda: np_threshold([0.5, 0.6], [True], alpha=0.1),
                    lambda: np_threshold([0.5], [True], alpha=0.0),
                    lambda: np_threshold([0.5], [True], alpha=1.0)):
            try:
                bad()
            except ValueError:
                continue
            raise AssertionError("expected a ValueError")


class TestTheInterval:
    def test_it_is_seeded_so_a_refit_that_moves_is_data_moving(self):
        s, m = _pairs([(1.0 - 0.005 * i, i % 7 != 0) for i in range(200)])
        a = np_threshold(s, m, alpha=0.15, n_boot=300)
        b = np_threshold(s, m, alpha=0.15, n_boot=300)
        assert (a.lo, a.hi, a.threshold) == (b.lo, b.hi, b.threshold)

    def test_it_brackets_the_point_estimate_and_narrows_with_evidence(self):
        """More of the same data locates the threshold better, which is the whole claim of the interval."""
        rng = np.random.default_rng(11)

        def sample(n: int):
            s = rng.uniform(0.3, 1.0, n)
            return s, rng.uniform(0, 1, n) < s

        small = np_threshold(*sample(120), alpha=0.20, n_boot=300)
        large = np_threshold(*sample(4000), alpha=0.20, n_boot=300)
        for e in (small, large):
            assert e.measured is True
            assert e.lo <= e.threshold <= e.hi, (e.lo, e.threshold, e.hi)
        assert (large.hi - large.lo) < (small.hi - small.lo)


class TestPerClass:
    def test_each_class_carries_its_own_bound(self):
        """A mislabelled pedestrian and a mislabelled bollard are not equally expensive.

        One alpha for the whole ontology means the bound is wrong for at least one class, so the per-class
        alpha has to reach the fit rather than being applied afterwards. Same data, two bounds, two
        thresholds is the check that it does.
        """
        # Every fifth detection is wrong, and the top-scoring one is right, so both bounds are reachable:
        # at 0.20 the whole sample sits exactly on the bound (40 wrong of 200), while at 0.02 only the
        # four above the first error qualify.
        s, m = _pairs([(1.0 - 0.004 * i, i % 5 != 4) for i in range(200)])
        out = fit_per_class({1: (s, m), 2: (s, m)},
                            alpha_for={1: 0.02}, default_alpha=0.20, n_boot=200)
        strict, loose = out["per_class"][1], out["per_class"][2]
        assert strict.alpha == 0.02 and loose.alpha == 0.20
        assert (strict.n_accept, loose.n_accept) == (4, 200)
        assert strict.threshold > loose.threshold
        assert strict.far_at == 0.0 and abs(loose.far_at - 0.20) < 1e-9

    def test_a_class_that_earned_nothing_is_present_with_its_reason(self):
        """Omitting it would make "no threshold" indistinguishable from "nobody looked"."""
        good = _pairs([(1.0 - 0.004 * i, i % 5 != 4) for i in range(200)])
        thin = _pairs([(0.9, True), (0.8, False)])
        out = fit_per_class({1: good, 2: thin}, alpha_for={}, default_alpha=0.20, n_boot=200)
        assert out["n_fitted"] == 1 and out["n_refused"] == 1
        assert set(out["per_class"]) == {1, 2}
        assert out["per_class"][2].measured is False and out["per_class"][2].reason


def test_numpy_torch_agree_when_cuda_present():
    import pytest

    try:
        import torch
    except ImportError:
        pytest.skip("no torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.default_rng(4242)
    s = rng.uniform(0.2, 1.0, 500)
    m = rng.uniform(0, 1, 500) < s
    a = np_threshold(s, m, alpha=0.15, n_boot=400, device="cpu")
    b = np_threshold(s, m, alpha=0.15, n_boot=400, device="cuda:0")
    assert abs(a.threshold - b.threshold) < 1e-6
    assert abs(a.lo - b.lo) < 1e-6 and abs(a.hi - b.hi) < 1e-6
