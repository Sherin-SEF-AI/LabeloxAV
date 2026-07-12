"""Box IoU matrix + NMS (Tier 2). All-pairs xyxy IoU is the primitive under recall-recovery mining
(mine_unmatched), cross-channel fusion (fuse_channels), and the pseudo-GT agreement score, and NMS is the
postprocess that collapses overlapping detections. Both are all-pairs O(n^2), so they move to the GPU at scale.

box_iou_matrix matches services.recall.recover.iou_matrix exactly; nms is greedy (score-ordered suppression)
with the all-pairs IoU computed on the GPU. Runs through torch when a CUDA device is present, else the
identical NumPy math, which is also the correctness oracle."""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def _iou_matrix_np(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def _iou_matrix_torch(a, b, device):
    f64 = torch.float64
    a = torch.as_tensor(np.asarray(a, dtype=np.float64).reshape(-1, 4), device=device, dtype=f64)
    b = torch.as_tensor(np.asarray(b, dtype=np.float64).reshape(-1, 4), device=device, dtype=f64)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = torch.where(union > 0, inter / torch.where(union > 0, union, torch.ones_like(union)),
                      torch.zeros_like(union))
    return iou.cpu().numpy()


def box_iou_matrix(a, b, device=None) -> np.ndarray:
    """All-pairs IoU of two xyxy box sets, shape (len(a), len(b)). Matches recall.iou_matrix exactly."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    if _HAS_TORCH and device != "cpu" and torch.cuda.is_available():
        return _iou_matrix_torch(a, b, torch.device("cuda"))
    return _iou_matrix_np(a, b)


def nms(boxes, scores, iou_thr: float = 0.5, device=None) -> np.ndarray:
    """Greedy non-max suppression. Returns the kept indices (into the input order), highest score first. The
    all-pairs IoU is computed on the GPU; the score-ordered suppression is the standard greedy pass, so the
    result matches a reference NMS exactly."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = boxes.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    iou = box_iou_matrix(boxes, boxes, device=device)
    suppressed = np.zeros(n, dtype=bool)
    kept = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(int(i))
        # suppress every later, lower-score box that overlaps this one past the threshold
        overlap = iou[i] > iou_thr
        overlap[i] = False
        suppressed |= overlap
    return np.array(kept, dtype=np.int64)
