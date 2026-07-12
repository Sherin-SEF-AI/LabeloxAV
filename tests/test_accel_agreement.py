"""Pseudo-GT agreement kernel (Tier 2) verification: the composed agreement matrix must equal a direct NumPy
composition of box IoU + class match + mask IoU, the GPU path must equal the CPU path, and the consensus
reduction must reproduce the ORACLYX vote() decision for a matched detection."""

import numpy as np

from core.accel.agreement import agreement_matrix, consensus_scores
from core.accel.boxes import _iou_matrix_np
from core.accel.mask_iou import _iou_matrix_np as _mask_iou_np


def test_agreement_matches_direct_composition():
    rng = np.random.default_rng(0)
    n = 40
    xy = rng.uniform(0, 500, size=(n, 2))
    boxes = np.column_stack([xy, xy + rng.uniform(20, 120, size=(n, 2))])
    classes = rng.integers(0, 5, size=n)
    masks = rng.random((n, 24, 32)) > 0.5

    A = agreement_matrix(boxes, classes, masks, w_box=0.5, w_class=0.3, w_mask=0.2, device="cpu")
    box = _iou_matrix_np(boxes, boxes)
    cls = (classes[:, None] == classes[None, :]).astype(float)
    msk = _mask_iou_np(masks)
    ref = 0.5 * box + 0.3 * cls + 0.2 * msk
    assert np.allclose(A, ref, atol=1e-6)
    assert np.allclose(np.diag(A), 1.0, atol=1e-6)     # a detection fully agrees with itself


def test_no_masks_renormalizes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    classes = np.array([1, 1, 2])
    A = agreement_matrix(boxes, classes, masks=None, w_box=0.5, w_class=0.3, device="cpu")
    # identical box + same class -> full agreement; disjoint + different class -> 0
    assert abs(A[0, 1] - 1.0) < 1e-6
    assert abs(A[0, 2] - 0.0) < 1e-6


def test_consensus_matches_vote_rule():
    # three detections that agree (same class, overlapping) + one dissenter -> consensus by majority
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 9, 9], [500, 500, 520, 520]], dtype=float)
    classes = np.array([6, 6, 6, 2])
    confs = np.array([0.9, 0.8, 0.85, 0.7])
    A = agreement_matrix(boxes, classes, masks=None, device="cpu")
    res = consensus_scores(A, confs, agree_thr=0.5, min_agree=3)
    assert res["consensus"][0] and res["agree_count"][0] == 3      # the three agree
    assert not res["consensus"][3]                                  # the dissenter is alone
    # the agreeing detections score higher than the dissenter
    assert res["score"][0] > res["score"][3]


def test_gpu_matches_cpu():
    from core.accel.agreement import agreement_matrix as am
    from core.accel.boxes import gpu_available
    if not gpu_available():
        return
    rng = np.random.default_rng(2)
    n = 128
    xy = rng.uniform(0, 800, size=(n, 2))
    boxes = np.column_stack([xy, xy + rng.uniform(20, 150, size=(n, 2))])
    classes = rng.integers(0, 8, size=n)
    masks = rng.random((n, 32, 32)) > 0.5
    cpu = am(boxes, classes, masks, device="cpu")
    gpu = am(boxes, classes, masks, device="cuda")
    assert np.allclose(cpu, gpu, atol=1e-5)
