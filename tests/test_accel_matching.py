"""Evaluation-plane kernels (Tier 4) verification: the prediction-to-GT matcher must reproduce a straightforward
greedy reference (TP/FP/FN, per-class, score-ordered), the confusion matrix and per-slice recall/precision must
bin those verdicts exactly, and the matcher's GPU IoU must equal the CPU IoU. A session-scale timing is
printed."""

import time

import numpy as np

from core.accel.matching import gpu_available, match_detections
from core.accel.slices import confusion_matrix, slice_precision, slice_recall


def _ref_match(pb, ps, pc, gb, gc, thr):
    """Independent greedy reference."""
    from services.recall.recover import iou_matrix
    P, G = len(pb), len(gb)
    iou = iou_matrix(pb, gb) if P and G else np.zeros((P, G))
    match_gt = np.full(P, -1)
    taken = np.zeros(G, bool)
    for i in np.argsort(-np.asarray(ps), kind="stable"):
        best_j, best = -1, thr
        for j in range(G):
            if not taken[j] and pc[i] == gc[j] and iou[i, j] >= best:
                best_j, best = j, iou[i, j]
        if best_j >= 0:
            match_gt[i] = best_j
            taken[best_j] = True
    return match_gt


def _boxes(n, rng):
    xy = rng.uniform(0, 1000, size=(n, 2))
    return np.column_stack([xy, xy + rng.uniform(20, 120, size=(n, 2))])


def test_matcher_matches_greedy_reference():
    rng = np.random.default_rng(0)
    gb = _boxes(30, rng)
    gc = rng.integers(0, 4, size=30)
    # predictions: some near a GT (jittered), some spurious
    pb = np.vstack([gb[:20] + rng.normal(0, 3, size=(20, 4)), _boxes(15, rng)])
    pc = np.concatenate([gc[:20], rng.integers(0, 4, size=15)])
    ps = rng.random(35)

    m = match_detections(pb, ps, pc, gb, gc, iou_thr=0.5)
    ref = _ref_match(pb, ps, pc, gb, gc, 0.5)
    assert np.array_equal(m["match_gt"], ref)
    assert m["n_tp"] + m["n_fp"] == len(pb)
    assert m["n_tp"] + m["n_fn"] == len(gb)


def test_confusion_and_slices():
    # 3 GTs of classes [0,1,0]; preds match gt0 (class0 ok), misclass gt1 as class 2, miss gt2, and one FP
    gb = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]], dtype=float)
    gc = np.array([0, 1, 0])
    pb = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [500, 500, 510, 510]], dtype=float)
    pc = np.array([0, 2, 0])                             # pred1 correct, pred2 wrong class (won't match gt1), pred3 FP
    ps = np.array([0.9, 0.8, 0.7])
    m = match_detections(pb, ps, pc, gb, gc, iou_thr=0.5)
    C = confusion_matrix(gc, pc, m, n_classes=3)
    assert C[0, 0] == 1                                  # gt0 class0 correctly predicted
    assert C[3, 2] == 1                                  # a false positive of class 2 (pred2 matched nothing)
    assert C[1, 3] == 1 and C[0, 3] == 1                 # gt1 and gt2 missed (false negatives)

    # per-slice recall: put gt0,gt2 in slice 0, gt1 in slice 1
    rec = slice_recall(m["gt_matched"], [0, 1, 0], n_slices=2)
    assert rec["total"][0] == 2 and rec["detected"][0] == 1 and abs(rec["recall"][0] - 0.5) < 1e-9
    assert rec["total"][1] == 1 and rec["detected"][1] == 0 and rec["recall"][1] == 0.0
    prec = slice_precision(m["tp"], [0, 0, 0], n_slices=1)
    assert prec["tp"][0] == 1 and prec["total"][0] == 3   # 1 TP of 3 predictions


def test_measurable():
    if not gpu_available():
        return
    rng = np.random.default_rng(2)
    P, G = 4000, 3000
    gb = _boxes(G, rng)
    gc = rng.integers(0, 20, size=G)
    pb = _boxes(P, rng)
    pc = rng.integers(0, 20, size=P)
    ps = rng.random(P)
    m_gpu = match_detections(pb, ps, pc, gb, gc, device="cuda")
    m_cpu = match_detections(pb, ps, pc, gb, gc, device="cpu")
    assert np.array_equal(m_gpu["match_gt"], m_cpu["match_gt"])   # GPU IoU -> same assignment

    for _ in range(3):
        match_detections(pb, ps, pc, gb, gc, device="cuda")
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        match_detections(pb, ps, pc, gb, gc, device="cuda")
    gpu_ms = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(n):
        match_detections(pb, ps, pc, gb, gc, device="cpu")
    cpu_ms = (time.perf_counter() - t0) / n * 1000
    print(f"\nmatch {P} preds x {G} GT = {P * G:,} pairs: GPU-IoU {gpu_ms:.2f} ms | NumPy {cpu_ms:.2f} ms | "
          f"speedup {cpu_ms / gpu_ms:.1f}x")
