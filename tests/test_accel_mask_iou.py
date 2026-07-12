"""All-pairs mask-IoU kernel (Tier 1) verification: the bit-packed Triton kernel must match the exact NumPy
raster reference it accelerates, and a session-scale timing is printed so the win is measurable."""

import time

import numpy as np

from core.accel.mask_iou import _iou_matrix_np, gpu_available, mask_iou_matrix


def test_matches_reference_exact():
    rng = np.random.default_rng(0)
    masks = rng.random((60, 40, 56)) > 0.5
    ref = _iou_matrix_np(masks)
    got = mask_iou_matrix(masks)
    assert got.shape == (60, 60)
    assert np.allclose(got, ref, atol=1e-5)          # bit-packed popcount == raster intersection, exactly
    assert np.allclose(np.diag(got), 1.0, atol=1e-5)  # a mask is identical to itself
    # symmetry
    assert np.allclose(got, got.T, atol=1e-6)


def test_disjoint_and_identical():
    a = np.zeros((3, 8, 8), dtype=bool)
    a[0, :4, :] = True          # top half
    a[1, 4:, :] = True          # bottom half (disjoint from 0)
    a[2, :4, :] = True          # identical to 0
    iou = mask_iou_matrix(a)
    assert abs(iou[0, 1]) < 1e-6      # disjoint -> 0
    assert abs(iou[0, 2] - 1.0) < 1e-6  # identical -> 1


def test_empty_and_single():
    assert mask_iou_matrix(np.zeros((0, 4, 4), dtype=bool)).shape == (0, 0)
    one = mask_iou_matrix(np.ones((1, 4, 4), dtype=bool))
    assert one.shape == (1, 1) and abs(one[0, 0] - 1.0) < 1e-6


def test_measurable():
    if not gpu_available():
        return
    rng = np.random.default_rng(1)
    N = 512                       # 512 masks, all-pairs = 262,144 IoUs
    masks = rng.random((N, 128, 128)) > 0.6
    got = mask_iou_matrix(masks)
    assert np.allclose(got, _iou_matrix_np(masks), atol=1e-5)

    for _ in range(3):
        mask_iou_matrix(masks)                        # warm up
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        mask_iou_matrix(masks)
    gpu_ms = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(n):
        _iou_matrix_np(masks)
    cpu_ms = (time.perf_counter() - t0) / n * 1000
    print(f"\nmask IoU {N}x{N} ({N * N:,} pairs, 128x128): Triton bit-packed {gpu_ms:.2f} ms | "
          f"NumPy raster {cpu_ms:.2f} ms | speedup {cpu_ms / gpu_ms:.1f}x")
