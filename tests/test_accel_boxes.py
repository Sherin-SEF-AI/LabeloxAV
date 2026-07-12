"""Box IoU + NMS kernel (Tier 2) verification: box_iou_matrix must match the recall-recovery reference exactly,
NMS must match a straightforward greedy reference (and torchvision when available), and a session-scale timing
is printed."""

import time

import numpy as np
import pytest

from core.accel.boxes import box_iou_matrix, gpu_available, nms
from services.recall.recover import iou_matrix as recall_iou_matrix


def _rand_boxes(n, rng, scale=1920):
    xy = rng.uniform(0, scale, size=(n, 2))
    wh = rng.uniform(10, 300, size=(n, 2))
    return np.column_stack([xy, xy + wh])


def test_box_iou_matches_recall_reference():
    rng = np.random.default_rng(0)
    a = _rand_boxes(80, rng)
    b = _rand_boxes(50, rng)
    got = box_iou_matrix(a, b)
    ref = recall_iou_matrix(a, b)
    assert got.shape == (80, 50)
    assert np.allclose(got, ref, atol=1e-9)
    # self-IoU diagonal is 1
    assert np.allclose(np.diag(box_iou_matrix(a, a)), 1.0, atol=1e-9)


def test_nms_matches_greedy_reference():
    rng = np.random.default_rng(1)
    boxes = _rand_boxes(200, rng)
    scores = rng.random(200)
    thr = 0.5

    def _ref_nms(boxes, scores, thr):
        order = np.argsort(-scores, kind="stable")
        keep = []
        taken = np.zeros(len(boxes), bool)
        for i in order:
            if taken[i]:
                continue
            keep.append(int(i))
            for j in order:
                if not taken[j] and j != i and recall_iou_matrix([boxes[i]], [boxes[j]])[0, 0] > thr:
                    taken[j] = True
        return keep

    assert list(nms(boxes, scores, thr)) == _ref_nms(boxes, scores, thr)


def test_nms_matches_torchvision():
    tv = pytest.importorskip("torchvision")
    import torch
    rng = np.random.default_rng(2)
    boxes = _rand_boxes(300, rng)
    scores = rng.random(300)
    ours = set(nms(boxes, scores, 0.5).tolist())
    ref = set(tv.ops.nms(torch.as_tensor(boxes, dtype=torch.float32),
                         torch.as_tensor(scores, dtype=torch.float32), 0.5).tolist())
    assert ours == ref


def test_measurable():
    if not gpu_available():
        return
    rng = np.random.default_rng(3)
    n = 3000
    a = _rand_boxes(n, rng)
    assert np.allclose(box_iou_matrix(a, a, device="cuda"), box_iou_matrix(a, a, device="cpu"), atol=1e-9)
    for _ in range(3):
        box_iou_matrix(a, a, device="cuda")
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        box_iou_matrix(a, a, device="cuda")
    gpu_ms = (time.perf_counter() - t0) / reps * 1000
    t0 = time.perf_counter()
    for _ in range(reps):
        box_iou_matrix(a, a, device="cpu")
    cpu_ms = (time.perf_counter() - t0) / reps * 1000
    print(f"\nbox IoU {n}x{n} = {n * n:,} pairs: GPU {gpu_ms:.2f} ms | NumPy {cpu_ms:.2f} ms | "
          f"speedup {cpu_ms / gpu_ms:.1f}x")
