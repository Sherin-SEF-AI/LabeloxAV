"""Ego-hood mask estimation: a static bottom band is found, a static sky is not, and box containment works.

Pure tests on synthetic stacks: a constant bottom region across frames is the hood; everything else varies.
"""

from __future__ import annotations

import numpy as np

from services.autolabel.ego_mask import estimate_from_gray_stack


def _stack_with_bottom_hood(t=20, h=100, w=100, hood_rows=20, seed=0):
    rng = np.random.default_rng(seed)
    stack = rng.integers(0, 255, size=(t, h, w)).astype(np.float32)   # everything varies frame to frame
    hood = rng.integers(0, 255, size=(hood_rows, w)).astype(np.float32)
    for i in range(t):
        stack[i, h - hood_rows:h, :] = hood                          # identical hood pixels every frame
    return stack


def test_estimates_bottom_hood():
    m = estimate_from_gray_stack(_stack_with_bottom_hood(), var_thresh=6.0)
    assert m is not None
    assert 0.1 < m.area_frac < 0.35                                  # ~ the bottom 20% band
    assert m.contains_bbox((10, 85, 90, 99), 100, 100, frac=0.5)     # a box on the hood
    assert not m.contains_bbox((10, 5, 90, 30), 100, 100, frac=0.5)  # a box up in the scene


def test_all_dynamic_has_no_hood():
    rng = np.random.default_rng(3)
    stack = rng.integers(0, 255, size=(20, 100, 100)).astype(np.float32)
    assert estimate_from_gray_stack(stack) is None                   # nothing static at the bottom


def test_static_sky_is_not_a_hood():
    # Static band at the TOP (sky), dynamic bottom (road). The hood must be bottom-anchored, so this is None.
    rng = np.random.default_rng(4)
    stack = rng.integers(0, 255, size=(20, 100, 100)).astype(np.float32)
    sky = rng.integers(0, 255, size=(20, 100)).astype(np.float32)
    for i in range(20):
        stack[i, 0:20, :] = sky
    assert estimate_from_gray_stack(stack) is None


def test_too_few_frames_returns_none():
    assert estimate_from_gray_stack(np.zeros((2, 100, 100), dtype=np.float32)) is None
