"""Sensor-QA extras verification: batched histogram equalization matches cv2.equalizeHist per frame, and the
motion-blur estimate scores a directionally-blurred frame as lower-gradient and more anisotropic than the sharp
original, with the anisotropy orientation aligned to the blur direction."""

import numpy as np
import pytest

from core.accel.sensorqa import equalize_hist_batch, motion_blur_score


def test_equalize_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    frames = rng.integers(20, 200, size=(8, 120, 160), dtype=np.uint8)   # a low-contrast batch
    out = equalize_hist_batch(frames)
    for i in range(len(frames)):
        assert np.array_equal(out[i], cv2.equalizeHist(frames[i]))       # exact match


def test_motion_blur_detects_directional_smear():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, size=(200, 260)).astype(np.uint8)
    # horizontal motion blur: average over a 1x15 kernel -> gradients suppressed in x
    kernel = np.ones((1, 15)) / 15.0
    blurred = cv2.filter2D(sharp, -1, kernel)

    s = motion_blur_score(sharp[None], device="cpu")
    b = motion_blur_score(blurred[None], device="cpu")
    assert b["blur_extent"][0] < s["blur_extent"][0]                     # blurred frame has less gradient energy
    assert b["anisotropy"][0] > s["anisotropy"][0]                       # and is more directional
    assert b["anisotropy"][0] > 0.3
    # horizontal smear kills x-gradients, so the dominant remaining gradient is vertical (angle ~ +/- pi/2)
    assert abs(abs(b["blur_angle"][0]) - np.pi / 2) < 0.3


def test_batched_and_shapes():
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 255, size=(5, 64, 64), dtype=np.uint8)
    r = motion_blur_score(frames, device="cpu")
    assert r["blur_extent"].shape == (5,) and r["anisotropy"].shape == (5,)
    assert (r["anisotropy"] >= 0).all() and (r["anisotropy"] <= 1.0 + 1e-9).all()
