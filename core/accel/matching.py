"""Prediction-to-GT matching for evaluation (Tier 4). The mAP-style match between predictions and ground truth
(IoU gate + class gate + greedy score-ordered assignment) is the bottleneck at eval scale, not the metric
arithmetic. This computes the all-pairs IoU on the GPU (the box-IoU kernel), gates it by class, then does the
standard greedy assignment, producing the per-prediction TP/FP verdict and the matched GT index that a
confusion matrix and per-slice precision/recall are then aggregated from.

match_detections runs the IoU on the GPU when a CUDA device is present; the greedy assignment itself is a cheap
sequential pass. The result matches a straightforward NumPy reference exactly."""

from __future__ import annotations

import numpy as np

from core.accel.boxes import box_iou_matrix, gpu_available  # noqa: F401 (re-export gpu_available)


def match_detections(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_thr=0.5,
                     device=None) -> dict:
    """COCO-style greedy match of predictions to ground truth.

    Predictions are taken in descending score order; each claims the highest-IoU unclaimed GT of the same class
    with IoU >= iou_thr (a true positive), else it is a false positive. Unclaimed GTs are false negatives.
    Returns {match_gt (P,) int (GT index per prediction, -1 if FP), tp (P,) bool, gt_matched (G,) bool,
    n_tp, n_fp, n_fn}."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    pred_classes = np.asarray(pred_classes).reshape(-1)
    gt_classes = np.asarray(gt_classes).reshape(-1)
    P, G = pred_boxes.shape[0], gt_boxes.shape[0]

    match_gt = np.full(P, -1, dtype=np.int64)
    gt_matched = np.zeros(G, dtype=bool)
    if P == 0 or G == 0:
        return {"match_gt": match_gt, "tp": np.zeros(P, bool), "gt_matched": gt_matched,
                "n_tp": 0, "n_fp": int(P), "n_fn": int(G)}

    iou = box_iou_matrix(pred_boxes, gt_boxes, device=device)          # (P, G) on the GPU
    class_ok = pred_classes[:, None] == gt_classes[None, :]
    iou = np.where(class_ok, iou, 0.0)                                 # a match must share the class

    order = np.argsort(-pred_scores, kind="stable")
    for i in order:
        row = iou[i].copy()
        row[gt_matched] = 0.0                                          # cannot claim an already-matched GT
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            match_gt[i] = j
            gt_matched[j] = True

    tp = match_gt >= 0
    return {"match_gt": match_gt, "tp": tp, "gt_matched": gt_matched,
            "n_tp": int(tp.sum()), "n_fp": int((~tp).sum()), "n_fn": int((~gt_matched).sum())}
