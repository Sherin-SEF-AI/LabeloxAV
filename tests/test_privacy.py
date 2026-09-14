"""The geographic release: what leaves, what is suppressed, and what the budget refuses.

This is the one module here whose failure mode is not a wrong number but a disclosure. A driving trace
reconstructs a home address, a workplace and a daily route from a handful of points, so the tests that
matter most are the negative ones: that no raw coordinate survives, that a thin cell is dropped rather
than noised, and that the accountant refuses rather than degrading quietly.
"""

from __future__ import annotations

import math
import statistics
import uuid

import pytest
from sqlalchemy import delete

from core.privacy import (
    DEFAULT_BUDGET,
    GEO_CELL_M,
    K_ANONYMITY,
    PrivacyBudgetExhausted,
    aggregate_points,
    cell_of,
    coarsen_time,
    gaussian,
    laplace,
    noisy_count,
)
from db.session import get_sessionmaker

BLR = (12.9716, 77.5946)


class TestTheMechanisms:
    def test_laplace_is_centred_on_zero(self):
        draws = [laplace(1.0) for _ in range(20000)]
        assert abs(statistics.mean(draws)) < 0.1

    def test_laplace_spreads_with_its_scale(self):
        """The scale is sensitivity over epsilon, so a smaller epsilon has to mean more noise. A
        mechanism whose spread did not follow its scale would provide the privacy of whatever spread it
        happened to have."""
        tight = statistics.pstdev([laplace(0.5) for _ in range(20000)])
        loose = statistics.pstdev([laplace(4.0) for _ in range(20000)])
        assert loose > tight * 3

    def test_a_zero_scale_adds_nothing(self):
        assert laplace(0.0) == 0.0
        assert gaussian(0.0) == 0.0

    def test_gaussian_is_centred_and_scales(self):
        draws = [gaussian(2.0) for _ in range(20000)]
        assert abs(statistics.mean(draws)) < 0.2
        assert 1.7 < statistics.pstdev(draws) < 2.3

    def test_the_noise_is_not_reproducible(self):
        """Every other random draw here is seeded so a result can be replayed. This one must not be: an
        attacker who can replay the noise can subtract it, and a seeded privacy mechanism provides none."""
        import inspect

        from core import privacy

        src = inspect.getsource(privacy)
        assert "secrets" in src
        assert "np.random.default_rng" not in src and "random.seed" not in src

    def test_a_noisy_count_is_never_negative(self):
        """A negative count is not a possible answer, and a reader seeing minus three would rightly stop
        trusting every other number on the page."""
        for _ in range(500):
            assert noisy_count(0, epsilon=0.1) >= 0

    def test_a_noisy_count_stays_near_the_truth_on_average(self):
        draws = [noisy_count(1000, epsilon=1.0) for _ in range(2000)]
        assert abs(statistics.mean(draws) - 1000) < 15

    def test_a_zero_epsilon_release_is_refused_as_an_exact_release(self):
        with pytest.raises(ValueError, match="exact release"):
            noisy_count(5, epsilon=0.0)


class TestCells:
    def test_a_cell_is_the_same_metric_size_at_any_latitude(self):
        """A fixed decimal-degree grid gives cells of different real sizes at different latitudes, so the
        privacy guarantee would vary with where the vehicle drove."""
        def width_m(lat: float) -> float:
            """The real ground width of one cell at this latitude, stepped east until the cell changes."""
            _cid, _clat, lon1 = cell_of(lat, 0.0)
            lon2, j = lon1, 1
            while lon2 == lon1 and j < 100000:
                _cid2, _clat2, lon2 = cell_of(lat, j * 1e-5)
                j += 1
            return abs(lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat))

        assert width_m(0.0) == pytest.approx(width_m(45.0), rel=0.05)

    def test_nearby_fixes_share_a_cell(self):
        a = cell_of(BLR[0], BLR[1])[0]
        b = cell_of(BLR[0] + 0.0005, BLR[1] + 0.0005)[0]
        assert a == b

    def test_distant_fixes_do_not(self):
        a = cell_of(BLR[0], BLR[1])[0]
        b = cell_of(BLR[0] + 0.05, BLR[1])[0]
        assert a != b

    def test_the_returned_centre_is_the_cell_not_the_fix(self):
        """Returning the fix's own coordinates rounded would still be the fix. The centre is the cell."""
        _cid, lat, lon = cell_of(BLR[0], BLR[1])
        assert (lat, lon) != BLR
        assert abs(lat - BLR[0]) < 0.01 and abs(lon - BLR[1]) < 0.01

    def test_the_cell_is_about_the_configured_size(self):
        assert 100.0 <= GEO_CELL_M <= 1000.0


class TestSuppression:
    def _cluster(self, n: int, lat=BLR[0], lon=BLR[1]):
        # All inside one cell.
        return [(lat + i * 1e-6, lon + i * 1e-6) for i in range(n)]

    def test_a_thin_cell_is_dropped_entirely(self):
        """A cell with one fix is one vehicle at one place at one time, and noise on a count of one still
        says somebody was there."""
        out = aggregate_points(self._cluster(K_ANONYMITY - 1))
        assert out["cells"] == []
        assert out["suppressed_cells"] == 1
        assert out["suppressed_points"] == K_ANONYMITY - 1

    def test_a_full_cell_survives(self):
        out = aggregate_points(self._cluster(K_ANONYMITY * 5))
        assert len(out["cells"]) == 1
        assert out["cells"][0]["count"] > 0

    def test_suppression_uses_the_true_count_not_the_noisy_one(self):
        """Suppressing on the noisy count would leak which cells were near the threshold, and noising
        first would let a cell holding one fix acquire a count of eleven and be released."""
        import inspect

        from core import privacy

        src = inspect.getsource(privacy.aggregate_points)
        suppress_at = src.index("if n < k:")
        noise_at = src.index("noisy_count(n, epsilon)")
        assert suppress_at < noise_at

    def test_the_response_says_how_much_it_hid(self):
        """A map missing most of its data because every cell was thin reads as a fleet that did not drive
        there, and that is a different fact."""
        out = aggregate_points(self._cluster(3) + self._cluster(3, lat=BLR[0] + 0.05))
        assert out["suppressed_cells"] == 2 and out["suppressed_points"] == 6

    def test_no_raw_coordinate_survives_the_aggregation(self):
        """The property this module exists for, checked directly: not one released number is a fix."""
        pts = self._cluster(60)
        out = aggregate_points(pts)
        released = {(c["lat"], c["lon"]) for c in out["cells"]}
        assert released, "the test is vacuous if nothing was released"
        for lat, lon in pts:
            assert (lat, lon) not in released

    def test_points_with_no_fix_are_skipped_rather_than_counted_at_zero(self):
        out = aggregate_points([(None, None)] * 50 + self._cluster(K_ANONYMITY * 2))
        assert len(out["cells"]) == 1
        assert "0_0" not in {c["cell_id"] for c in out["cells"]}


class TestTimeCoarsening:
    def test_a_timestamp_is_rounded_down_to_the_window(self):
        """A cell id beside a one-second timestamp re-identifies the trip the cell was meant to hide: two
        coarse locations at exact times are a trajectory."""
        ts = 1_700_000_123_456_789_000
        got = coarsen_time(ts, 60)
        assert got <= ts
        assert got % (60 * 1_000_000_000) == 0

    def test_two_nearby_times_collapse_together(self):
        base = 1_700_000_000_000_000_000
        assert coarsen_time(base + 1_000_000_000, 60) == coarsen_time(base + 30_000_000_000, 60)


@pytest.mark.db
class TestTheAccountant:
    async def _clean(self, scope: str):
        from db.models import PrivacyBudget, PrivacyReleaseLog

        async with get_sessionmaker()() as db:
            await db.execute(delete(PrivacyReleaseLog).where(PrivacyReleaseLog.scope == scope))
            await db.execute(delete(PrivacyBudget).where(PrivacyBudget.scope == scope))
            await db.commit()

    async def test_a_missing_budget_is_created_at_the_default_rather_than_unlimited(self):
        """Unlimited is the one interpretation that cannot be right: a scope nobody configured is a scope
        nobody thought about, and the first release would be unbounded."""
        from services.analytics.privacy_release import budget_state, spend

        scope = f"t-{uuid.uuid4().hex[:8]}"
        try:
            async with get_sessionmaker()() as db:
                await spend(db, scope=scope, epsilon=0.5, endpoint="/t", mechanism="laplace")
                st = await budget_state(db, scope)
            assert st["epsilon_total"] == DEFAULT_BUDGET
            assert st["epsilon_spent"] == pytest.approx(0.5)
        finally:
            await self._clean(scope)

    async def test_releases_compose_and_the_budget_runs_out(self):
        """Ten releases at 0.1 leak as much as one at 1.0. A system applying a per-query epsilon without
        tracking the total provides a guarantee it has already spent."""
        from services.analytics.privacy_release import spend

        scope = f"t-{uuid.uuid4().hex[:8]}"
        try:
            async with get_sessionmaker()() as db:
                for _ in range(10):
                    await spend(db, scope=scope, epsilon=0.5, endpoint="/t", mechanism="laplace")
                with pytest.raises(PrivacyBudgetExhausted):
                    await spend(db, scope=scope, epsilon=0.5, endpoint="/t", mechanism="laplace")
        finally:
            await self._clean(scope)

    async def test_the_refusal_names_what_is_left_and_what_was_asked(self):
        from services.analytics.privacy_release import spend

        scope = f"t-{uuid.uuid4().hex[:8]}"
        try:
            async with get_sessionmaker()() as db:
                with pytest.raises(PrivacyBudgetExhausted) as exc:
                    await spend(db, scope=scope, epsilon=DEFAULT_BUDGET * 2, endpoint="/t",
                                mechanism="laplace")
            assert "epsilon left" in str(exc.value)
        finally:
            await self._clean(scope)

    async def test_every_release_is_logged_and_the_log_sums_to_the_counter(self):
        from services.analytics.privacy_release import budget_state, spend

        scope = f"t-{uuid.uuid4().hex[:8]}"
        try:
            async with get_sessionmaker()() as db:
                for eps in (0.1, 0.25, 0.4):
                    await spend(db, scope=scope, epsilon=eps, endpoint="/t", mechanism="laplace")
                st = await budget_state(db, scope)
            assert st["epsilon_logged"] == pytest.approx(0.75)
            assert st["epsilon_spent"] == pytest.approx(st["epsilon_logged"])
        finally:
            await self._clean(scope)

    async def test_a_query_that_returns_nothing_still_costs(self):
        """Asking the question is what costs privacy. A query returning no cells has still told the asker
        that every cell in that region is thin."""
        from services.analytics.privacy_release import budget_state, release_geo_cells

        scope = f"t-{uuid.uuid4().hex[:8]}"
        try:
            async with get_sessionmaker()() as db:
                out = await release_geo_cells(db, [(BLR[0], BLR[1])], endpoint="/t", scope=scope)
                st = await budget_state(db, scope)
            assert out["cells"] == []
            assert st["epsilon_spent"] > 0
        finally:
            await self._clean(scope)


class TestTheEndpointNoLongerLeaks:
    def test_the_geo_handler_releases_cells_rather_than_points(self):
        import inspect

        from services.api.routers import analytics

        src = inspect.getsource(analytics.geo)
        assert "release_geo_cells" in src
        assert "return await dashboards.geo_points" not in src

    def test_it_refuses_with_a_status_when_the_budget_is_gone(self):
        import inspect

        from services.api.routers import analytics

        src = inspect.getsource(analytics.geo)
        assert "PrivacyBudgetExhausted" in src and "429" in src


class TestDistillation:
    """Shrinking a model until it fits a board's latency budget, and reporting every round.

    The measurable part on a host that is not the board is the refusal: a student nobody can compile for
    a target cannot be measured against its budget, and training one first would spend GPU hours to
    arrive at the same answer.
    """

    def test_an_unknown_student_architecture_is_refused_by_name(self):
        import asyncio

        from services.forgyx.distill import STUDENT_ARCHS, distill_to_budget

        async def go():
            async with get_sessionmaker()() as db:
                return await distill_to_budget(db, teacher_version="x", target="orin_nano_trt",
                                               student_arch="resnet50")

        res = asyncio.run(go())
        assert res["ok"] is False
        assert "resnet50" in res["reason"] and str(list(STUDENT_ARCHS)) in res["reason"]

    def test_an_unregistered_teacher_is_refused(self):
        import asyncio

        from services.forgyx.distill import distill_to_budget

        async def go():
            async with get_sessionmaker()() as db:
                return await distill_to_budget(db, teacher_version="no-such-model",
                                               target="orin_nano_trt")

        res = asyncio.run(go())
        assert res["ok"] is False and "not registered" in res["reason"]

    def test_the_round_refuses_before_training_when_the_toolchain_is_missing(self):
        """Training first would spend GPU hours to arrive at the same refusal, so the capability check
        comes before the training call and the test pins the order."""
        import inspect

        from services.forgyx import distill

        src = inspect.getsource(distill._measure_round)
        require_at = src.index("require(target)")
        train_at = src.index("enqueue_job")
        assert require_at < train_at

    def test_every_target_the_loop_accepts_has_a_budget(self):
        """A target with no budget is one nothing can fit, and the loop refuses rather than looping four
        times against a null."""
        from services.forgyx.cooptimize import TARGET_BUDGET_MS

        assert TARGET_BUDGET_MS and all(v > 0 for v in TARGET_BUDGET_MS.values())

    def test_agreement_is_measured_with_the_shadow_matcher_not_a_second_notion(self):
        """A distillation quality number computed a different way from the shadow number would be two
        numbers nobody could compare."""
        import inspect

        from services.forgyx import distill

        assert "agreements_for_runs" in inspect.getsource(distill._teacher_agreement)
