"""The sequential acceptance rule, pinned as pure math before any lot uses it.

Four properties the settlement engine depends on: the test's error rates are what Wald says they
are; a conjunctive accept never settles on weaker evidence than the fixed Wilson draw; the expected
remaining verdicts fall as the evidence accumulates toward either bound; and reject fires on a
handful of defects, which is where the whole saving is.
"""

from __future__ import annotations

import math

import pytest

from services.labelops.sampling import (
    acceptance_decision,
    sprt_bounds,
    sprt_decision,
    sprt_expected_remaining,
    sprt_oc,
    sprt_steps,
)
from services.labelops.settlement import sample_target

FARS = (0.05, 0.02, 0.01)


class TestBoundsAndSteps:
    def test_wald_bounds(self):
        b = sprt_bounds(alpha=0.05, beta=0.10)
        assert b["bound_accept"] == pytest.approx(math.log(0.10 / 0.95))
        assert b["bound_reject"] == pytest.approx(math.log(0.90 / 0.05))
        assert b["bound_accept"] < 0 < b["bound_reject"]

    def test_steps_move_in_the_right_directions(self):
        st = sprt_steps(p0=0.05, p1=0.025)
        assert st["defect"] == pytest.approx(math.log(2))
        assert st["clean"] < 0

    def test_bad_inputs_refuse(self):
        with pytest.raises(ValueError):
            sprt_steps(p0=0.025, p1=0.05)
        with pytest.raises(ValueError):
            sprt_bounds(alpha=0, beta=0.1)
        with pytest.raises(ValueError):
            sprt_decision(3, 2, p0=0.05, p1=0.025)


class TestOperatingCharacteristic:
    @pytest.mark.parametrize("far", FARS)
    def test_error_rates_are_walds(self, far):
        """L(p1) = 1 - alpha and L(p0) = beta: the test accepts a good class 95% of the time and a
        bad one 10% of the time. Wald's approximation is what the bounds are built from, so the OC
        reproduces it to the rounding."""
        assert sprt_oc(far / 2, p0=far, p1=far / 2) == pytest.approx(0.95, abs=1e-3)
        assert sprt_oc(far, p0=far, p1=far / 2) == pytest.approx(0.10, abs=1e-3)

    @pytest.mark.parametrize("far", FARS)
    def test_oc_is_monotone_and_bounded(self, far):
        grid = [i / 200 for i in range(0, 201)]
        vals = [sprt_oc(p, p0=far, p1=far / 2) for p in grid]
        assert vals[0] == 1.0 and vals[-1] == 0.0
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:], strict=False))


class TestConjunctiveAccept:
    @pytest.mark.parametrize("far", FARS)
    def test_the_engine_accept_is_never_weaker_than_wilson(self, far):
        """The engine accepts only when BOTH rules accept, so the settled evidence is at least the
        fixed draw's by construction. The honest shape, pinned: with the good rate at half the far
        bound the SPRT is the stricter rule at 0 to 2 defects and Wilson at 3 or more, so the
        conjunction is what keeps a 3-defect lot from settling before the fixed rule would. The
        saving on a clean class is against the fixed CAP, not against Wilson: 0 defects accepts at
        roughly 80% of the cap instead of at the cap."""
        cap = sample_target(far)
        first_sprt: dict[int, int | None] = {k: None for k in range(5)}
        first_wilson: dict[int, int | None] = {k: None for k in range(5)}
        for n in range(1, cap * 3):
            for k in range(5):
                if k > n:
                    continue
                if first_sprt[k] is None and sprt_decision(k, n, p0=far, p1=far / 2)["verdict"] == "accept":
                    first_sprt[k] = n
                if first_wilson[k] is None and acceptance_decision(
                        k, n, max_defect_rate=far)["verdict"] == "accept":
                    first_wilson[k] = n
        for k in range(3):
            assert first_sprt[k] >= first_wilson[k], f"{k} defects: the SPRT is the binding rule"
        assert first_sprt[3] < first_wilson[3] or first_sprt[4] < first_wilson[4], \
            "at 3 to 4 defects Wilson binds; without the conjunction the SPRT would settle first"
        for k in range(5):
            engine = max(first_sprt[k], first_wilson[k])
            assert engine >= first_wilson[k]
        assert first_sprt[0] < cap, "a clean class stops before the cap"
        assert first_sprt[0] <= 0.8 * cap + 1, "and saves at least a fifth of the fixed draw"

    @pytest.mark.parametrize("far", FARS)
    def test_reject_fires_on_a_handful_of_defects(self, far):
        """Where the saving is: a bad class fails after five straight defects, or a few more spread
        over clean crops, long before the fixed draw of a hundred or more would be complete."""
        cap = sample_target(far)
        straight = next(k for k in range(1, 50)
                        if sprt_decision(k, k, p0=far, p1=far / 2)["verdict"] == "reject")
        assert straight == 5
        # spread over the first increment of 25: how many defects reject at n=25
        k25 = next(k for k in range(1, 26)
                   if sprt_decision(k, 25, p0=far, p1=far / 2)["verdict"] == "reject")
        assert k25 < cap * far * 3, "reject at 25 needs only a few defects"
        assert acceptance_decision(k25, 25, max_defect_rate=far)["verdict"] in ("reject", "inconclusive")


class TestExpectedRemaining:
    @pytest.mark.parametrize("far", FARS)
    def test_remaining_falls_toward_the_accept_bound_on_clean_verdicts(self, far):
        prev = None
        for n in range(5, 400, 5):
            d = sprt_decision(0, n, p0=far, p1=far / 2)
            if d["verdict"] != "continue":
                assert d["expected_remaining"] == 0
                break
            if prev is not None:
                assert d["expected_remaining"] <= prev + 1e-6
            prev = d["expected_remaining"]

    def test_zero_once_crossed_and_finite_between(self):
        assert sprt_expected_remaining(0, 200, p0=0.05, p1=0.025) == 0.0
        assert sprt_expected_remaining(8, 10, p0=0.05, p1=0.025) == 0.0
        mid = sprt_expected_remaining(1, 40, p0=0.05, p1=0.025)
        assert 0 < mid < 10_000

    def test_an_unjudged_lot_is_undecided_not_half_bad(self):
        d = sprt_decision(0, 0, p0=0.05, p1=0.025)
        assert d["verdict"] == "continue"
        assert 0.4 < d["oc"] < 0.6, "the indifference prior reads a fresh lot as a coin, not a loss"
        assert d["expected_remaining"] > 0


class TestDecisionShape:
    def test_reason_names_the_bound_and_the_count(self):
        d = sprt_decision(0, 10, p0=0.35, p1=0.175)
        assert d["verdict"] == "accept" and "10 verdicts" in d["reason"]
        d = sprt_decision(8, 10, p0=0.35, p1=0.175)
        assert d["verdict"] == "reject"
        d = sprt_decision(1, 10, p0=0.35, p1=0.175)
        assert d["verdict"] == "continue" and d["bound_accept"] < d["llr"] < d["bound_reject"]
