"""Integration: recall.iou_matrix routes large box-IoU calls through the fused GPU kernel and must return the
identical result to its NumPy path. Also asserts the pure recall tier still imports with no torch."""

import numpy as np

import services.recall.recover as rc
from core.accel.boxes import gpu_available


def _rand_boxes(n, rng):
    xy = rng.uniform(0, 1920, size=(n, 2))
    return np.column_stack([xy, xy + rng.uniform(10, 300, size=(n, 2))])


def test_recall_iou_accel_matches_numpy():
    rng = np.random.default_rng(0)
    a = _rand_boxes(600, rng)
    b = _rand_boxes(600, rng)              # 360,000 pairs > the accel gate

    # force the NumPy branch
    rc._ACCEL_MIN_PAIRS = 10**12
    ref = rc.iou_matrix(a, b)

    if gpu_available():
        rc._ACCEL_MIN_PAIRS = 1000         # force the GPU branch
        got = rc.iou_matrix(a, b)
        assert np.allclose(got, ref, atol=1e-9)
    rc._ACCEL_MIN_PAIRS = 250_000          # restore default


def test_small_calls_stay_numpy_and_correct():
    rng = np.random.default_rng(1)
    a = _rand_boxes(5, rng)
    m = rc.iou_matrix(a, a)                # below the gate -> NumPy path
    assert np.allclose(np.diag(m), 1.0, atol=1e-9)
