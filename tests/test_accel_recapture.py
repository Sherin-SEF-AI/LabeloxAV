"""Capture-recapture, against arithmetic done by hand rather than against itself.

The expected values below are worked out from Chapman (1951) and the Seber variance in the comments, using
the textbook two-sample example (100 marked, 100 recaptured in the second sample, 20 carrying marks). None
of them is read back from the implementation, which is the only way this test can fail when the estimator
is wrong.
"""

from __future__ import annotations

import numpy as np

from core.accel.recapture import lincoln_petersen, stratified_recapture

# The worked example, computed here and not by the module under test.
#
#   n1 = 20 + 80 = 100      everything the first observer found
#   n2 = 20 + 80 = 100      everything the second observer found
#   m2 = 20                 found by both
#
#   Chapman   N = (101)(101)/21 - 1 = 10201/21 - 1        = 484.761904...
#   Seber   var = (101)(101)(80)(80) / [(21^2)(22)]
#               = 65,286,400 / 9,702                      = 6729.169243...
#   sd                                                    =   82.031514...
#   half-width = 1.959963984540054 * 82.031514            =  160.778813...
#   CI                                                    = [323.9831, 645.5407]
#   model recall = 100 / 484.7619                         =    0.206287
_N1, _N2, _M2 = 100, 100, 20
_CHAPMAN_N = 10201.0 / 21.0 - 1.0
_CHAPMAN_VAR = (101.0 * 101.0 * 80.0 * 80.0) / ((21.0**2) * 22.0)


class TestTheWorkedExample:
    def test_chapman_population(self):
        est = lincoln_petersen(n_both=_M2, n_model_only=_N1 - _M2, n_human_only=_N2 - _M2)
        assert est.measured is True
        assert abs(est.population - _CHAPMAN_N) < 1e-3
        assert abs(est.population - 484.7619) < 1e-3

    def test_chapman_variance(self):
        est = lincoln_petersen(n_both=_M2, n_model_only=_N1 - _M2, n_human_only=_N2 - _M2)
        assert abs(est.variance - _CHAPMAN_VAR) < 1e-3
        assert abs(est.variance - 6729.169243) < 1e-3

    def test_the_confidence_interval(self):
        half = 1.959963984540054 * float(np.sqrt(_CHAPMAN_VAR))
        est = lincoln_petersen(n_both=_M2, n_model_only=_N1 - _M2, n_human_only=_N2 - _M2)
        assert abs(est.lo - (_CHAPMAN_N - half)) < 1e-3
        assert abs(est.hi - (_CHAPMAN_N + half)) < 1e-3
        assert abs(est.lo - 323.9831) < 1e-3
        assert abs(est.hi - 645.5407) < 1e-3

    def test_recall_is_the_interval_inverted(self):
        """Recall falls as the estimated population rises, so its bounds cross over.

        Getting this backwards would report the optimistic end of the interval as the pessimistic one,
        which on a safety metric is the wrong direction to be wrong in.
        """
        est = lincoln_petersen(n_both=_M2, n_model_only=_N1 - _M2, n_human_only=_N2 - _M2)
        assert abs(est.model_recall - _N1 / _CHAPMAN_N) < 1e-5
        assert abs(est.recall_lo - _N1 / est.hi) < 1e-5
        assert abs(est.recall_hi - _N1 / est.lo) < 1e-5
        assert est.recall_lo < est.model_recall < est.recall_hi

    def test_the_uncorrected_form_is_the_plain_ratio(self):
        est = lincoln_petersen(n_both=_M2, n_model_only=_N1 - _M2, n_human_only=_N2 - _M2,
                               chapman=False)
        assert abs(est.population - (_N1 * _N2 / _M2)) < 1e-9
        assert abs(est.population - 500.0) < 1e-9

    def test_chapman_sits_below_the_uncorrected_ratio(self):
        # The correction exists because the plain ratio is biased upward at small overlap. If Chapman ever
        # came out higher, the correction has been applied in the wrong direction.
        corrected = lincoln_petersen(n_both=3, n_model_only=20, n_human_only=20)
        plain = lincoln_petersen(n_both=3, n_model_only=20, n_human_only=20, chapman=False)
        assert corrected.population < plain.population


class TestWhatItRefusesToSay:
    def test_no_overlap_is_unmeasured_not_infinite(self):
        """With nothing found by both, the population is unbounded above.

        Chapman still yields a finite number here ((n1+1)(n2+1) - 1) and it means nothing at all. Returning
        it would be the same defect as reporting an unmeasured precision as 1.0.
        """
        est = lincoln_petersen(n_both=0, n_model_only=40, n_human_only=40)
        assert est.measured is False
        assert est.population is None and est.model_recall is None
        assert "unbounded" in est.reason

    def test_an_observer_that_found_nothing_is_unmeasured(self):
        assert lincoln_petersen(n_both=0, n_model_only=0, n_human_only=10).measured is False

    def test_negative_counts_are_refused(self):
        try:
            lincoln_petersen(n_both=1, n_model_only=-1, n_human_only=1)
        except ValueError:
            return
        raise AssertionError("a negative capture count should raise")

    def test_the_population_is_never_below_what_was_actually_seen(self):
        """Perfect agreement drives the estimate toward the observed union and must not fall under it.

        The arithmetic can land slightly below when every object was found by both; a population smaller
        than the objects in hand is not a bound, it is a contradiction.
        """
        est = lincoln_petersen(n_both=50, n_model_only=0, n_human_only=0)
        assert est.population >= 50.0
        assert est.model_recall <= 1.0


class TestStratified:
    def test_pooling_sums_the_strata_rather_than_collapsing_them(self):
        """Two strata with different capture rates must not be averaged into one rate.

        Stratum A: 20/80/80 (the worked example). Stratum B: 10/10/10, so n1 = n2 = 20 and m2 = 10.
          B: N = (21)(21)/11 - 1 = 441/11 - 1 = 39.0909...
          pooled population = 484.7619 + 39.0909 = 523.8528
        Collapsing to (30, 90, 90) instead gives (121)(121)/31 - 1 = 471.2903, which is smaller than the
        objects the two strata jointly imply. That difference is the whole reason to stratify.
        """
        out = stratified_recapture([[20, 80, 80], [10, 10, 10]], labels=["dense", "sparse"])
        assert out["measured"] is True
        b_expected = 441.0 / 11.0 - 1.0
        assert abs(out["per_stratum"][1]["population"] - b_expected) < 1e-3
        assert abs(out["pooled"]["population"] - (_CHAPMAN_N + b_expected)) < 1e-3

        collapsed = lincoln_petersen(n_both=30, n_model_only=90, n_human_only=90)
        assert out["pooled"]["population"] > collapsed.population

    def test_variance_adds_across_independent_strata(self):
        # Stratum B: n1 = n2 = 20, m2 = 10, so var = (21)(21)(10)(10) / [(11^2)(12)] = 44100/1452.
        out = stratified_recapture([[20, 80, 80], [10, 10, 10]])
        b_var = 44100.0 / 1452.0
        assert abs(b_var - 30.371901) < 1e-5
        assert abs(out["per_stratum"][1]["variance"] - b_var) < 1e-3
        assert abs(out["pooled"]["variance"] - (_CHAPMAN_VAR + b_var)) < 1e-3

    def test_an_unmeasurable_stratum_is_named_and_excluded(self):
        """It must not be pooled as though it had contributed zero missed objects.

        Counting a stratum with no overlap as a zero would make the pooled population smaller, i.e. recall
        would look better precisely where the estimator knows least.
        """
        out = stratified_recapture([[20, 80, 80], [0, 5, 5]], labels=["dense", "empty"])
        assert out["measured"] is True
        assert out["unmeasured"] == ["empty"]
        assert out["pooled"]["n_strata_pooled"] == 1
        assert abs(out["pooled"]["population"] - _CHAPMAN_N) < 1e-3

    def test_every_stratum_unmeasurable_is_not_a_zero_result(self):
        out = stratified_recapture([[0, 5, 5], [0, 2, 2]])
        assert out["measured"] is False
        assert out["pooled"] is None
        assert "found by both" in out["reason"]

    def test_no_strata_at_all(self):
        out = stratified_recapture(np.zeros((0, 3)))
        assert out["measured"] is False and out["n_strata"] == 0

    def test_label_count_must_match(self):
        try:
            stratified_recapture([[1, 1, 1], [2, 2, 2]], labels=["only_one"])
        except ValueError:
            return
        raise AssertionError("a label/stratum count mismatch should raise")


def test_numpy_torch_agree_when_cuda_present():
    import pytest

    try:
        import torch
    except ImportError:
        pytest.skip("no torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.default_rng(20260819)
    counts = np.stack([rng.integers(1, 40, 256), rng.integers(0, 60, 256), rng.integers(0, 60, 256)],
                      axis=1)
    a = stratified_recapture(counts, device="cpu")
    b = stratified_recapture(counts, device="cuda:0")
    assert abs(a["pooled"]["population"] - b["pooled"]["population"]) < 1e-6
    assert abs(a["pooled"]["variance"] - b["pooled"]["variance"]) < 1e-6
    for x, y in zip(a["per_stratum"], b["per_stratum"], strict=True):
        assert abs(x["population"] - y["population"]) < 1e-6
