"""M-4D.2: trajectory smoothing. The smoothed path must reduce jitter, keep the true endpoints, and leave
paths too short to smooth unchanged."""

from __future__ import annotations

import numpy as np

from services.temporal.trajectory import smooth_path


def test_smoothing_reduces_jitter_and_keeps_endpoints():
    rng = np.random.default_rng(0)
    pts = [[float(i), float(rng.normal(0.0, 1.0))] for i in range(24)]   # x linear, y jittery
    sm = smooth_path(pts, window=7)
    assert np.var([p[1] for p in sm]) < np.var([p[1] for p in pts])      # jitter reduced
    assert abs(sm[0][1] - pts[0][1]) < 1e-3 and abs(sm[-1][1] - pts[-1][1]) < 1e-3  # endpoints fixed
    assert len(sm) == len(pts)


def test_short_path_unchanged():
    assert smooth_path([[0.0, 0.0], [1.0, 2.0]]) == [[0.0, 0.0], [1.0, 2.0]]


def test_three_d_path_smooths_all_axes():
    pts = [[float(i), float(i % 2), float(-i)] for i in range(10)]       # jitter on y
    sm = smooth_path(pts, window=5)
    assert len(sm) == 10 and len(sm[0]) == 3
    assert np.var([p[1] for p in sm]) < np.var([p[1] for p in pts])
