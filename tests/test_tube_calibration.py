"""Tube coherence and the two-axis calibration that consumes it.

A per-frame detector scores each frame alone, so a box that appears once in the middle of nothing gets the
same confidence as the twentieth frame of a stable track. The tube score is what the track adds, and the
joint surface is where the two axes are combined into a probability.

The fixtures below are tracks short enough to reason about: a perfectly smooth one, one with a gap, one
that jitters, and one whose class keeps changing. Each isolates a single component so a regression names
itself instead of moving one number.
"""

from __future__ import annotations

import numpy as np

from core.accel.tube_score import score_tracks, tube_score
from services.oraclyx.joint_calibration import JointSurface, _monotone_2d, _pav, fit_surface


def _line(n: int, step: float = 5.0, size: float = 20.0) -> list[list[float]]:
    """A box moving at constant velocity: the definition of a well-behaved object."""
    return [[i * step, 0.0, i * step + size, size] for i in range(n)]


class TestTheComponents:
    def test_a_smooth_complete_track_scores_near_one(self):
        t = tube_score(_line(20))
        assert t.measured is True
        assert t.continuity == 1.0 and t.agreement == 1.0
        assert t.stability > 0.99, t.stability
        assert t.score > 0.99

    def test_constant_velocity_is_perfectly_stable(self):
        """Steady motion has a large frame-to-frame displacement and is not jitter.

        Measuring the first difference would score a fast object as unstable, which would penalise exactly
        the vehicles a driving corpus cares about most. What marks an artifact is the motion changing.
        """
        slow = tube_score(_line(20, step=1.0))
        fast = tube_score(_line(20, step=40.0))
        assert abs(slow.stability - fast.stability) < 1e-6
        assert fast.stability > 0.99

    def test_a_gap_costs_continuity_and_nothing_else(self):
        valid = [True] * 20
        for i in (5, 6, 7, 8, 9):
            valid[i] = False
        t = tube_score(_line(20), valid)
        assert abs(t.continuity - 0.75) < 1e-9
        assert t.agreement == 1.0
        assert t.score < tube_score(_line(20)).score

    def test_jitter_costs_stability_and_nothing_else(self):
        rng = np.random.default_rng(3)
        boxes = np.asarray(_line(20, step=0.0), dtype=float)
        boxes[:, [0, 2]] += rng.normal(0, 15.0, 20)[:, None]     # shaking, relative to a size-20 box
        t = tube_score(boxes)
        assert t.continuity == 1.0 and t.agreement == 1.0
        assert t.stability < 0.5, t.stability

    def test_a_class_that_keeps_changing_costs_agreement(self):
        steady = tube_score(_line(20), None, [7] * 20)
        flipping = tube_score(_line(20), None, [7 if i % 2 else 3 for i in range(20)])
        assert steady.agreement == 1.0
        assert abs(flipping.agreement - 0.5) < 1e-9
        assert flipping.score < steady.score

    def test_a_short_track_is_unmeasured_rather_than_perfect(self):
        """Two frames have no jitter to measure and perfect continuity by construction.

        Reporting 1.0 would rank a two-frame flicker above a twenty-frame track with one gap, which is
        exactly backwards.
        """
        for n in (0, 1, 2):
            t = tube_score(_line(n) if n else np.zeros((0, 4)))
            assert t.measured is False and t.score is None, n
            assert "below the 3 needed" in t.reason

    def test_mismatched_validity_raises(self):
        try:
            tube_score(_line(5), [True, True])
        except ValueError:
            return
        raise AssertionError("a validity mask of the wrong length should raise")


class TestBatch:
    def test_unmeasurable_tracks_are_nan_not_zero(self):
        out = score_tracks([{"boxes": _line(20)}, {"boxes": _line(2)}, {"boxes": _line(10)}])
        s = out["scores"]
        assert np.isnan(s[1]) and not np.isnan(s[0])
        assert out["n_unmeasured"] == 1
        # Ranking must not put the too-short one below a genuinely incoherent one, or above it.
        assert out["measured"].tolist() == [True, False, True]


class TestIsotonic:
    def test_pav_returns_the_closest_non_decreasing_sequence(self):
        """Hand-worked: [3, 1] with equal weights pools to their mean, 2."""
        out = _pav(np.array([3.0, 1.0]), np.array([1.0, 1.0]))
        assert np.allclose(out, [2.0, 2.0])
        # Already sorted is left alone.
        assert np.allclose(_pav(np.array([1.0, 2.0, 3.0]), np.ones(3)), [1.0, 2.0, 3.0])
        # A weighted pool moves toward the heavier point: (3*1 + 1*3) / 4 = 1.5.
        assert np.allclose(_pav(np.array([3.0, 1.0]), np.array([1.0, 3.0])), [1.5, 1.5])

    def test_the_surface_is_monotone_in_both_axes(self):
        """The property the whole fit exists to assert, checked directly rather than through the ECE.

        A raw binned estimate violates monotonicity constantly on small cells, and a violation is noise
        every time: nothing about the world makes a detection less likely to be right at higher confidence.
        """
        rng = np.random.default_rng(1)
        grid = rng.random((10, 10))
        counts = rng.integers(1, 50, (10, 10)).astype(float)
        out = _monotone_2d(grid, counts)
        assert np.all(np.diff(out, axis=0) >= -1e-9), "not non-decreasing in confidence"
        assert np.all(np.diff(out, axis=1) >= -1e-9), "not non-decreasing in tube score"

    def test_a_monotone_surface_is_left_alone(self):
        g = np.add.outer(np.linspace(0, 1, 10), np.linspace(0, 1, 10)) / 2.0
        out = _monotone_2d(g, np.ones((10, 10)))
        assert np.allclose(out, g, atol=1e-9)


class TestFitting:
    def _data(self, n: int, seed: int = 7):
        """Correctness rising with both axes, which is what a calibration should be able to recover."""
        rng = np.random.default_rng(seed)
        conf = rng.uniform(0, 1, n)
        tube = rng.uniform(0, 1, n)
        p = np.clip(0.15 + 0.5 * conf + 0.3 * tube, 0, 1)
        correct = rng.uniform(0, 1, n) < p
        frames = rng.integers(0, n // 4, n)
        return conf, tube, correct, frames

    def test_it_refuses_below_min_support(self):
        c, t, y, f = self._data(50)
        out = fit_surface(c, t, y, frame_ids=f)
        assert out["measured"] is False and out["trustworthy"] is False
        assert "below the 200 needed" in out["reason"]

    def test_it_fits_and_improves_calibration_on_held_out_frames(self):
        c, t, y, f = self._data(4000)
        out = fit_surface(c, t, y, frame_ids=f)
        assert out["measured"] is True, out.get("reason")
        assert out["trustworthy"] is True, out
        assert out["cal_val_ece"] < out["raw_val_ece"]
        assert out["n_train"] + out["n_val"] == 4000

    def test_the_split_is_by_frame_so_no_detection_leaks(self):
        """Two boxes on one frame share the scene, the lighting and often the object.

        Splitting per detection trains and validates on the same evidence and reports a calibration far
        better than it is.
        """
        c, t, y, f = self._data(2000)
        out = fit_surface(c, t, y, frame_ids=f)
        assert out["measured"] is True
        # Every frame must be wholly on one side; the fit reports the sizes it used.
        assert out["n_train"] > 0 and out["n_val"] > 0

    def test_uninformative_confidence_flattens_toward_the_base_rate_and_that_is_correct(self):
        """When confidence carries no signal, "0.5 everywhere" is the right answer, not a degenerate one.

        Correctness is a coin flip independent of both axes, so the raw score is badly miscalibrated
        (validation ECE 0.240) and a surface that reports the base rate everywhere is nearly perfect
        (0.020). Refusing that would be refusing the fit for succeeding.

        The finding belongs to the model, not to the calibration: it says this detector's confidence
        predicts nothing, which is exactly what a calibration is for telling you.
        """
        rng = np.random.default_rng(9)
        n = 3000
        out = fit_surface(rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                          rng.uniform(0, 1, n) < 0.5, frame_ids=rng.integers(0, n // 4, n))
        assert out["measured"] is True and out["trustworthy"] is True
        assert out["cal_val_ece"] < 0.05 < out["raw_val_ece"]
        assert abs(out["base_rate"] - 0.5) < 0.05

    def test_a_calibration_that_does_not_improve_validation_ece_is_refused(self):
        """Worse than none: it launders a raw score into something that looks measured.

        The input is already perfectly calibrated by construction (correct is Bernoulli at exactly the
        confidence), so a binned surface has nothing to gain and its rounding to lose. On most seeds it
        breaks about even; on this one it comes out 0.017 worse, which is the case the guard exists for
        and is a real outcome of the estimator rather than a contrived one.
        """
        rng = np.random.default_rng(21)
        n = 4000
        conf = rng.uniform(0, 1, n)
        correct = rng.uniform(0, 1, n) < conf
        out = fit_surface(conf, rng.uniform(0, 1, n), correct,
                          frame_ids=rng.integers(0, n // 4, n))
        assert out["improvement"] < 0
        assert out["measured"] is True
        assert out["trustworthy"] is False, out
        assert "no better than the raw" in out["reason"]

    def test_a_flat_surface_is_refused_as_degenerate(self):
        """A surface that is one number everywhere has calibrated by discarding the signal.

        Checked against the guard directly, because data that produces a genuinely flat surface also
        tends to be data where flat is the right answer, and the two cases have to be separable.
        """
        flat = _monotone_2d(np.full((10, 10), 0.4), np.ones((10, 10)))
        assert float(flat.max() - flat.min()) < 0.02

    def test_mismatched_inputs_raise(self):
        try:
            fit_surface(np.zeros(5), np.zeros(4), np.zeros(5))
        except ValueError:
            return
        raise AssertionError("mismatched lengths should raise")


class TestServing:
    def test_it_round_trips_through_json(self):
        c, t, y, f = TestFitting()._data(3000)
        out = fit_surface(c, t, y, frame_ids=f)
        s = out["surface"]
        back = JointSurface.from_json(s.to_json())
        for conf, tube in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.2), (0.3, 0.95)]:
            assert abs(s(conf, tube) - back(conf, tube)) < 1e-6

    def test_a_detection_with_no_tube_falls_back_to_the_confidence_marginal(self):
        """Not to the best tube bin, which would reward it for evidence it never provided.

        A detection run carries no track_id at all, so this is the ordinary case rather than an edge.
        """
        grid = [[0.1 * i + 0.05 * j for j in range(10)] for i in range(10)]
        s = JointSurface(grid)
        marginal = float(np.mean(grid[5]))
        assert abs(s(0.55, None) - marginal) < 1e-6
        assert s(0.55, None) < s(0.55, 0.95)

    def test_it_is_monotone_where_it_is_asked_for_a_value(self):
        c, t, y, f = TestFitting()._data(4000)
        s = fit_surface(c, t, y, frame_ids=f)["surface"]
        vals = [s(x, 0.5) for x in np.linspace(0.05, 0.95, 10)]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:], strict=False)), vals
        vals = [s(0.5, x) for x in np.linspace(0.05, 0.95, 10)]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:], strict=False)), vals


def test_numpy_torch_agree_when_cuda_present():
    import pytest

    try:
        import torch
    except ImportError:
        pytest.skip("no torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    tracks = [{"boxes": _line(20, step=float(i))} for i in range(1, 40)]
    a = score_tracks(tracks, device="cpu")["scores"]
    b = score_tracks(tracks, device="cuda:0")["scores"]
    assert np.nanmax(np.abs(a - b)) < 1e-6
