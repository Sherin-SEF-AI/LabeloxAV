"""Confusion + error-slice aggregation for evaluation (Tier 4). Given a prediction-to-GT match
(core.accel.matching), bin the results into the signals the champion-challenger gate reads: a class confusion
matrix (which classes are mistaken for which, with a background row/column for false negatives and false
positives) and per-slice recall/precision (by class, distance band, camera, occlusion, or any slice id). The
binning is exact NumPy bincount over the match verdict; the GPU-accelerated part of evaluation is the matching
IoU, this is the cheap aggregation on top."""

from __future__ import annotations

import numpy as np


def confusion_matrix(gt_classes, pred_classes, match, n_classes: int) -> np.ndarray:
    """Confusion matrix of shape (n_classes+1, n_classes+1): rows are the true class, columns the predicted
    class, and index n_classes is background. A matched prediction adds to (gt_class, pred_class); a false
    positive adds to (background, pred_class); a false negative adds to (gt_class, background). Built from a
    match dict (match_gt, gt_matched) as returned by match_detections."""
    gt_classes = np.asarray(gt_classes).reshape(-1).astype(np.int64)
    pred_classes = np.asarray(pred_classes).reshape(-1).astype(np.int64)
    match_gt = np.asarray(match["match_gt"]).reshape(-1)
    gt_matched = np.asarray(match["gt_matched"]).reshape(-1)
    bg = n_classes
    C = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)

    tp = match_gt >= 0
    if tp.any():                                          # matched predictions: (true gt class, predicted class)
        rows = gt_classes[match_gt[tp]]
        cols = pred_classes[tp]
        np.add.at(C, (rows, cols), 1)
    if (~tp).any():                                       # false positives: (background, predicted class)
        np.add.at(C, (bg, pred_classes[~tp]), 1)
    if (~gt_matched).any():                               # false negatives: (true class, background)
        np.add.at(C, (gt_classes[~gt_matched], bg), 1)
    return C


def slice_recall(gt_matched, gt_slice_ids, n_slices: int) -> dict:
    """Per-slice recall from the GT match verdict. gt_slice_ids assigns each GT to a slice (0..n_slices-1).
    Returns per-slice detected/total/recall, so a champion-challenger gate can see which slice regressed."""
    gt_matched = np.asarray(gt_matched).reshape(-1).astype(bool)
    ids = np.asarray(gt_slice_ids).reshape(-1).astype(np.int64)
    total = np.bincount(ids, minlength=n_slices).astype(np.int64)
    detected = np.bincount(ids, weights=gt_matched.astype(np.float64), minlength=n_slices)
    recall = np.divide(detected, total, out=np.full(n_slices, np.nan), where=total > 0)
    return {"total": total, "detected": detected.astype(np.int64), "recall": recall}


def slice_precision(tp, pred_slice_ids, n_slices: int) -> dict:
    """Per-slice precision from the prediction TP verdict. pred_slice_ids assigns each prediction to a slice."""
    tp = np.asarray(tp).reshape(-1).astype(bool)
    ids = np.asarray(pred_slice_ids).reshape(-1).astype(np.int64)
    total = np.bincount(ids, minlength=n_slices).astype(np.int64)
    hits = np.bincount(ids, weights=tp.astype(np.float64), minlength=n_slices)
    precision = np.divide(hits, total, out=np.full(n_slices, np.nan), where=total > 0)
    return {"total": total, "tp": hits.astype(np.int64), "precision": precision}
