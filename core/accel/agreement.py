"""Pseudo-GT cross-path agreement (Tier 2). When ORACLYX fuses the three auto-label paths (YOLO26, SAM 3.1,
Qwen3-VL) into pseudo ground truth, it scores how much the paths agree on each detection: box IoU + mask IoU +
class consistency, weighted into a confidence. That is an all-pairs match-and-score over detections, so it
composes the box-IoU and mask-IoU kernels into one pairwise agreement matrix, then reduces it to a per-detection
consensus that feeds the pseudo-label confidence and the active-learning uncertainty signal.

agreement_matrix returns the (N, N) weighted agreement; consensus_scores reduces it the way
services.oraclyx.consensus.vote does (a voter agrees past a threshold; the score is the confidence-weighted
agreeing fraction). Both run on the GPU through the underlying kernels when a CUDA device is present, else the
identical NumPy math, which is also the correctness oracle."""

from __future__ import annotations

import numpy as np

from core.accel.boxes import box_iou_matrix
from core.accel.mask_iou import mask_iou_matrix


def agreement_matrix(boxes, classes, masks=None, w_box=0.5, w_class=0.3, w_mask=0.2, device=None) -> np.ndarray:
    """Pairwise agreement (N, N) between detections: a weighted sum of box IoU, class match (1/0), and, when
    masks are supplied, mask IoU. The weights are renormalized when a term is absent (no masks), so the score
    stays in [0, 1]. Entry (i, j) is how strongly detection i and detection j agree."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    classes = np.asarray(classes).reshape(-1)
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    box_iou = box_iou_matrix(boxes, boxes, device=device)
    class_match = (classes[:, None] == classes[None, :]).astype(np.float64)
    wb, wc, wm = float(w_box), float(w_class), float(w_mask)
    if masks is not None:
        mask_iou = mask_iou_matrix(np.asarray(masks), device=device).astype(np.float64)
    else:
        mask_iou = None
        wm = 0.0
    total = wb + wc + wm
    if total <= 0:
        return np.zeros((n, n), dtype=np.float64)
    A = (wb * box_iou + wc * class_match) / total
    if mask_iou is not None:
        A = A + (wm * mask_iou) / total
    return A


def consensus_scores(A, confs=None, agree_thr: float = 0.5, min_agree: int = 3) -> dict:
    """Reduce an agreement matrix to a per-detection consensus, matching the ORACLYX vote() rule: detection i
    counts a voter j as agreeing when A[i, j] >= agree_thr (i agrees with itself); consensus holds when the
    agreeing count reaches ``min_agree``; the score is the confidence-weighted agreeing fraction. Returns
    per-detection agree_count, consensus flag, and score."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return {"agree_count": np.zeros(0, np.int64), "consensus": np.zeros(0, bool), "score": np.zeros(0)}
    confs = np.ones(n) if confs is None else np.asarray(confs, dtype=np.float64).reshape(-1)
    agree = A >= agree_thr
    np.fill_diagonal(agree, True)                       # a detection always agrees with itself
    agree_count = agree.sum(axis=1).astype(np.int64)
    conf_sum = float(confs.sum())
    conf_agree = agree @ confs                          # per-row conf-weighted agreeing mass
    score = np.where(conf_sum > 0, conf_agree / conf_sum, 0.0)
    return {"agree_count": agree_count, "consensus": agree_count >= min_agree, "score": score}
