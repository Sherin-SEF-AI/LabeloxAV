"""Reflections in the bonnet, and the streaming rewrite of the hood-mask estimator.

A dashcam looks out over a glossy bonnet that mirrors the scene above it, so the detector finds a
pedestrian there: plausible size, plausible confidence, upside down. The hood mask removes detections
inside the bonnet region where a mask was estimated at all, and nothing at all on a windscreen reflection
or wet tarmac.

The fixture below builds a real reflection: a textured patch and its vertical mirror, darkened and
lower-contrast the way a reflection is. The darkening is the point of the test, because it is what a
non-normalised correlation gets wrong.
"""

from __future__ import annotations

import numpy as np

from core.accel.reflection_twin import MIN_STD, reflection_twin, screen_detections
from services.autolabel.ego_mask import (
    WelfordStd,
    estimate_from_gray_frames,
    estimate_from_gray_stack,
)


def _scene(h: int = 300, w: int = 200, *, reflect: bool = True, dim: float = 0.45,
           seed: int = 5) -> tuple[np.ndarray, list[float], list[float]]:
    """A textured object at the top and, optionally, its darkened mirror below.

    Returns (gray, source_box, candidate_box).
    """
    rng = np.random.default_rng(seed)
    g = np.full((h, w), 120.0)
    g += rng.normal(0, 2.0, (h, w))                       # a little sensor noise everywhere
    obj = rng.normal(140, 40, (40, 30))                    # the real object: strongly textured
    g[60:100, 80:110] = obj
    if reflect:
        # Mirrored vertically, dimmer and lower-contrast, which is what a bonnet does to a scene.
        g[160:200, 80:110] = 120.0 + (obj[::-1, :] - obj.mean()) * dim
    else:
        g[160:200, 80:110] = rng.normal(140, 40, (40, 30))  # an unrelated second object
    return g, [80.0, 60.0, 110.0, 100.0], [80.0, 160.0, 110.0, 200.0]


class TestTheCorrelation:
    def test_a_darkened_mirror_is_found_despite_the_brightness_difference(self):
        """The normalisation is the whole test: a reflection is dimmer than its source.

        A raw dot product scores a reflection worst exactly when it is most obviously one. Subtracting
        the mean and dividing by the standard deviation compares shape, which survives the reflection,
        and discards brightness, which does not.
        """
        g, _src, cand = _scene(reflect=True, dim=0.45)
        v = reflection_twin(g, cand)
        assert v.measured is True, v.reason
        assert v.is_twin is True, v.ncc
        assert v.ncc > 0.7
        # And it found the source at the right distance: 160 - 60 = 100 px above.
        assert abs(v.offset_px - 100) <= 4, v.offset_px

    def test_an_unrelated_object_below_is_not_a_twin(self):
        g, _src, cand = _scene(reflect=False)
        v = reflection_twin(g, cand)
        assert v.measured is True
        assert v.is_twin is False, v.ncc

    def test_the_dimmer_the_reflection_the_more_a_raw_correlation_would_have_missed_it(self):
        """Across a range of dimming the verdict must not move, which is what normalisation buys."""
        for dim in (0.9, 0.6, 0.35, 0.2):
            g, _src, cand = _scene(reflect=True, dim=dim)
            v = reflection_twin(g, cand)
            assert v.measured and v.is_twin, (dim, v.ncc)


class TestWhatItRefusesToJudge:
    def test_a_flat_patch_is_unmeasured_rather_than_a_confident_twin(self):
        """A uniformly grey patch is the most common thing in a road scene.

        Its standard deviation is near zero, so the normalisation divides by noise and the correlation
        becomes noise amplified to 1.0. Calling all of those reflections would delete a lot of real road.
        """
        g = np.full((300, 200), 120.0)
        v = reflection_twin(g, [80.0, 160.0, 110.0, 200.0])
        assert v.measured is False and v.is_twin is False
        assert "nearly uniform" in v.reason

    def test_a_textured_box_at_the_top_of_the_frame_has_nothing_to_be_a_reflection_of(self):
        """Textured on purpose: an untextured one is refused for flatness before reaching this branch.

        Building the fixture the lazy way tested the wrong refusal and would have passed with the
        upward-search check deleted entirely.
        """
        rng = np.random.default_rng(11)
        g = np.full((300, 200), 120.0)
        g[0:30, 80:110] = rng.normal(140, 40, (30, 30))
        v = reflection_twin(g, [80.0, 0.0, 110.0, 30.0])
        assert v.measured is False
        assert "nothing above" in v.reason, v.reason

    def test_a_tiny_box_is_unmeasured(self):
        g, _src, _cand = _scene()
        v = reflection_twin(g, [80.0, 160.0, 82.0, 162.0])
        assert v.measured is False and "too small" in v.reason

    def test_a_non_grayscale_image_raises(self):
        try:
            reflection_twin(np.zeros((10, 10, 3)), [0, 0, 5, 5])
        except ValueError:
            return
        raise AssertionError("a colour image should raise rather than be silently reduced")


class TestScreening:
    def test_an_unmeasurable_detection_is_kept(self):
        """The null result is "we could not tell", and dropping on that deletes real objects."""
        g, src, cand = _scene(reflect=True)
        flat_box = [10.0, 250.0, 40.0, 280.0]        # nearly uniform background
        out = screen_detections(g, [src, cand, flat_box])
        assert out["n_twins"] == 1
        assert 1 not in out["keep"], "the reflection should have been dropped"
        assert 0 in out["keep"] and 2 in out["keep"], "the source and the unmeasurable must survive"
        assert out["n_unmeasured"] >= 1

    def test_the_source_object_is_never_mistaken_for_its_own_reflection(self):
        """The search is upward only, because a reflection always appears below its source.

        Searching both ways would flag the real object as the reflection of its own mirror image, which
        deletes the object and keeps the artifact.
        """
        g, src, _cand = _scene(reflect=True)
        v = reflection_twin(g, src)
        assert not (v.measured and v.is_twin), v.ncc


class TestTheStreamingHoodMask:
    def _stack(self, t: int = 12, h: int = 64, w: int = 64, seed: int = 2) -> np.ndarray:
        """A static bright hood along the bottom third, moving noise above it."""
        rng = np.random.default_rng(seed)
        stack = rng.normal(100, 30, (t, h, w)).astype(np.float32)
        stack[:, int(h * 0.7):, :] = 200.0            # perfectly static bottom band
        return stack

    def test_welford_matches_numpy_std_to_floating_point(self):
        """The array form materialises (T, H, W): 1.7 GB at 1080p over 200 frames.

        Welford gets the same variance in one pass with two accumulators, and is the better answer
        numerically too: the sum-of-squares shortcut subtracts two large nearly-equal numbers and loses
        precision exactly where the variance is small, which is the hood.
        """
        stack = self._stack()
        acc = WelfordStd(stack.shape[1:])
        for f in stack:
            acc.update(f)
        assert np.allclose(acc.std, stack.astype(np.float64).std(axis=0), atol=1e-9)

    def test_the_streaming_estimator_gives_the_same_mask_as_the_array_one(self):
        """Both paths share _mask_from_std, so the rewrite cannot drift from the four tests that pin it."""
        stack = self._stack()
        a = estimate_from_gray_stack(stack)
        b = estimate_from_gray_frames(iter(stack))
        assert (a is None) == (b is None)
        if a is not None:
            assert a.grid == b.grid

    def test_it_never_holds_more_than_two_accumulators(self):
        # A thousand frames at 480p would be 900 MB as a stack; here it is two 480p float64 arrays.
        acc = WelfordStd((64, 64))
        for _ in range(1000):
            acc.update(np.full((64, 64), 50.0))
        assert acc.n == 1000
        assert np.allclose(acc.std, 0.0)

    def test_too_few_frames_is_none_rather_than_a_mask_from_noise(self):
        assert estimate_from_gray_frames(iter(np.zeros((2, 64, 64)))) is None
        assert estimate_from_gray_frames(iter([])) is None

    def test_a_wrongly_shaped_frame_raises_rather_than_being_broadcast(self):
        acc = WelfordStd((8, 8))
        acc.update(np.zeros((8, 8)))
        try:
            acc.update(np.zeros((9, 9)))
        except ValueError:
            return
        raise AssertionError("a frame of the wrong shape should raise")


def test_min_std_is_high_enough_to_reject_sensor_noise():
    """A patch of pure sensor noise must not clear the flatness bar and get correlated.

    At sigma 2 the standard deviation is below MIN_STD, so it is refused rather than being matched
    against every other noisy patch in the frame.
    """
    rng = np.random.default_rng(1)
    assert float(rng.normal(0, 2.0, (40, 30)).std()) < MIN_STD
