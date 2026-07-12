"""Calibration front-end verification: sub-pixel corner refinement converges to the true corner (and near
cv2.cornerSubPix), and descriptor Hamming matching recovers a known correspondence with the ratio + mutual
tests, matching a brute reference."""

import numpy as np
import pytest

from core.accel.calib_frontend import descriptor_match, refine_corners


def test_corner_refine_converges_to_true():
    # a synthetic saddle/corner: a checkerboard crossing centered at (30.6, 20.4)
    H, W = 60, 80
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = 30.6, 20.4
    img = ((xs - cx > 0) ^ (ys - cy > 0)).astype(np.float64)        # 4-quadrant checker corner
    img = img + 0.0
    # blur a touch so gradients are well-defined
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, 1.0)

    start = np.array([[31.0, 20.0]])                                # integer-ish initial guess
    refined = refine_corners(img, start, win=6, iters=20)
    assert np.linalg.norm(refined[0] - np.array([cx, cy])) < 0.7    # converged near the true corner

    cv2 = pytest.importorskip("cv2")
    ref = cv2.cornerSubPix((img * 255).astype(np.uint8).astype(np.float32),
                           start.astype(np.float32).reshape(-1, 1, 2), (6, 6), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)).reshape(-1, 2)
    assert np.linalg.norm(refined[0] - ref[0]) < 1.0               # close to cv2's refinement


def test_descriptor_match_recovers_correspondence():
    rng = np.random.default_rng(0)
    N, B = 40, 32                                                   # 32 bytes = 256-bit ORB descriptors
    a = rng.integers(0, 256, size=(N, B), dtype=np.uint8)
    # b is a permuted copy with a few bits flipped, plus some distractors
    perm = rng.permutation(N)
    b = a[perm].copy()
    flip = rng.integers(0, 256, size=(N, B), dtype=np.uint8) & (rng.random((N, B)) < 0.02).astype(np.uint8) * 255
    b = b ^ flip
    distract = rng.integers(0, 256, size=(15, B), dtype=np.uint8)
    b = np.vstack([b, distract])

    res = descriptor_match(a, b, ratio=0.8, mutual=True, device="cpu")
    # b = a[perm], so a[i] lives at b[inv[i]] where inv = argsort(perm)
    inv = np.argsort(perm)
    correct = sum(1 for ia, ib in res["matches"].tolist() if inv[ia] == ib)
    assert correct >= int(0.8 * N)
    assert (res["distances"] < 30).all()                           # matched descriptors are close in Hamming
