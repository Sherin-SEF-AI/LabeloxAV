"""SEC-M2: the static-camera background prior (core/scene/background.py).

The static-camera capability a moving camera can never have. Fixtures are generated procedurally (numpy), never
downloaded: a fixed background with a bright square walking across it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.scene.background import Mog2Background, foreground_mask, temporal_median


def _clip(n: int = 12, size: int = 24, bg_value: int = 90):
    """A fixed grey background with a 4x4 white square at a different position in each frame."""
    bg = np.full((size, size, 3), bg_value, dtype=np.uint8)
    frames = []
    for i in range(n):
        f = bg.copy()
        x = 1 + (i % (size - 5))
        f[x:x + 4, x:x + 4] = 255
        frames.append(f)
    return bg, frames


def test_temporal_median_recovers_the_background():
    bg, frames = _clip()
    prior = temporal_median(frames)
    # The transient square is a minority at every pixel, so the median is the clean background everywhere.
    assert prior.shape == bg.shape
    assert prior.dtype == bg.dtype
    assert np.array_equal(prior, bg)


def test_temporal_median_subsamples_deterministically():
    _, frames = _clip(n=200)
    a = temporal_median(frames, max_samples=32)
    b = temporal_median(frames, max_samples=32)
    assert np.array_equal(a, b)  # deterministic, no Random/Date use


def test_foreground_mask_flags_only_the_moving_square():
    bg, frames = _clip()
    prior = temporal_median(frames)
    mask = foreground_mask(frames[0], prior, threshold=25.0)
    assert mask.shape == bg.shape[:2]
    assert mask.dtype == bool
    # exactly the 4x4 square differs from the background in frame 0 (x = 1)
    assert mask.sum() == 16
    assert mask[1:5, 1:5].all()


def test_foreground_mask_rejects_mismatched_shapes():
    bg, frames = _clip()
    with pytest.raises(ValueError):
        foreground_mask(frames[0][:, :10], bg)


def test_temporal_median_needs_frames():
    with pytest.raises(ValueError):
        temporal_median([])


def test_mog2_learns_a_background_or_is_unavailable():
    _, frames = _clip(n=30)
    try:
        mog = Mog2Background(history=30)
    except RuntimeError:
        pytest.skip("this OpenCV build has no MOG2")
    bg = mog.fit(frames)
    # MOG2 may return None on some builds until enough history; when it returns an image it is the right shape.
    if bg is not None:
        assert bg.shape == frames[0].shape
