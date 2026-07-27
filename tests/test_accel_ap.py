"""AP@50 primitive (core/accel/ap.py). The reference values are computed by hand from the 101-point
interpolation definition, so this validates the implementation against an independent derivation, not against
a copy of itself. Pure math, no database.
"""

from __future__ import annotations

import numpy as np

from core.accel.ap import average_precision, iou_thresholds_50_95, mean_ap


def test_ap_matches_hand_derived_reference():
    # Fixture: one class, 2 ground-truth objects, predictions [tp, fp, tp] in descending score.
    # PR points: (r=0.5, p=1.0), (r=0.5, p=0.5), (r=1.0, p=2/3).
    # 101-point interp: 51 recall points in [0, 0.5] see max precision 1.0; 50 points in (0.5, 1.0] see 2/3.
    # AP = (51*1.0 + 50*(2/3)) / 101 = 253/303.
    ap = average_precision([0.9, 0.8, 0.7], [True, False, True], n_gt=2)
    assert abs(ap - (253.0 / 303.0)) < 1e-6


def test_perfect_detector_is_one():
    assert abs(average_precision([0.9, 0.8], [True, True], n_gt=2) - 1.0) < 1e-6


def test_all_false_positives_is_zero():
    assert average_precision([0.9, 0.8, 0.7], [False, False, False], n_gt=2) == 0.0


def test_undefined_without_ground_truth_is_none():
    assert average_precision([0.9], [True], n_gt=0) is None


def test_no_predictions_but_gt_is_zero():
    assert average_precision([], [], n_gt=3) == 0.0


def test_score_order_does_not_matter_for_input():
    # The function sorts internally, so passing predictions out of score order gives the same AP.
    a = average_precision([0.9, 0.8, 0.7], [True, False, True], n_gt=2)
    b = average_precision([0.7, 0.9, 0.8], [True, True, False], n_gt=2)
    assert abs(a - b) < 1e-9


def test_mean_ap_skips_classes_without_gt():
    m, aps = mean_ap(
        {1: ([0.9, 0.8], [True, True]), 2: ([0.5], [False])},
        {1: 2, 2: 1, 3: 0},   # class 3 has no ground truth
    )
    assert 3 not in aps
    assert abs(aps[1] - 1.0) < 1e-6
    assert aps[2] == 0.0
    assert abs(m - (aps[1] + aps[2]) / 2) < 1e-9


def test_iou_thresholds_are_the_ten_coco_points():
    thr = iou_thresholds_50_95()
    assert thr[0] == 0.5 and thr[-1] == 0.95 and len(thr) == 10


def test_numpy_torch_agree_when_cuda_present():
    import pytest
    try:
        import torch
    except ImportError:
        pytest.skip("no torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    tp = rng.random(500) > 0.4
    a = average_precision(scores, tp, n_gt=200, device="cpu")
    b = average_precision(scores, tp, n_gt=200, device="cuda:0")
    assert abs(a - b) < 1e-6
