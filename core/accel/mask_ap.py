"""Mask (segmentation) average precision.

The box AP kernel in ap.py answers "did the detector find the object". It cannot answer "did it trace the
object", which is the whole point of a mask, and the system labels masks (SAM-assisted polygons, dense
semantic segmentation) that it could never score. `core/accel/mask_iou.py` already implements a fast
pairwise mask-IoU matrix, but no evaluation path called it.

The matching rule mirrors the box path exactly (greedy, descending score, class-aware, one ground truth per
prediction) so a mask AP number is directly comparable to a box AP number rather than being a differently
defined statistic that happens to share a name.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from core.accel.ap import average_precision

# Mask IoU thresholds. COCO reports mask AP over the same 0.50:0.05:0.95 sweep as boxes.
MASK_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def polygons_to_mask(polygons: list[list[float]], width: int, height: int) -> npt.NDArray[np.bool_]:
    """Rasterize flattened [x,y,x,y,...] polygons into a boolean mask.

    Annotations are stored as polygons (compact, editable); IoU needs pixels. Holes are handled by XOR so a
    donut-shaped annotation (an occluded interior) does not fill in, which a naive union would do.
    """
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons or []:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_iou(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float:
    """IoU of two boolean masks of the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def match_masks(pred_masks: list[npt.NDArray[np.bool_]], pred_scores: list[float],
                pred_classes: list[int], gt_masks: list[npt.NDArray[np.bool_]],
                gt_classes: list[int], iou_thr: float = 0.5) -> dict:
    """Greedy class-aware match of predicted masks to ground truth, mirroring core/accel/matching.

    Predictions are taken in descending score order; each claims the highest-IoU unclaimed ground truth of
    the same class at or above iou_thr. Returns per-prediction true-positive flags plus the counts.
    """
    order = sorted(range(len(pred_masks)), key=lambda i: -float(pred_scores[i]))
    gt_taken = [False] * len(gt_masks)
    tp = [False] * len(pred_masks)

    for i in order:
        best_j, best_iou = -1, iou_thr
        for j, gm in enumerate(gt_masks):
            if gt_taken[j] or gt_classes[j] != pred_classes[i]:
                continue
            iou = mask_iou(pred_masks[i], gm)
            if iou >= best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            gt_taken[best_j] = True
            tp[i] = True

    n_tp = sum(tp)
    return {"tp": [tp[i] for i in order], "scores": [float(pred_scores[i]) for i in order],
            "n_tp": n_tp, "n_fp": len(pred_masks) - n_tp, "n_fn": len(gt_masks) - n_tp}


def mask_average_precision(pred_masks: list[npt.NDArray[np.bool_]], pred_scores: list[float],
                           pred_classes: list[int], gt_masks: list[npt.NDArray[np.bool_]],
                           gt_classes: list[int], iou_thr: float = 0.5) -> float | None:
    """Mask AP at one IoU threshold. None when there is no ground truth (AP is undefined, not zero)."""
    if not gt_masks:
        return None
    m = match_masks(pred_masks, pred_scores, pred_classes, gt_masks, gt_classes, iou_thr)
    return average_precision(m["scores"], m["tp"], len(gt_masks))


def mask_ap_50_95(pred_masks: list[npt.NDArray[np.bool_]], pred_scores: list[float],
                  pred_classes: list[int], gt_masks: list[npt.NDArray[np.bool_]],
                  gt_classes: list[int]) -> dict:
    """Mask AP averaged over the ten COCO IoU thresholds, plus AP@50 on its own.

    Reporting both matters: AP@50 says whether the object was found with roughly the right extent, while the
    sweep is sensitive to boundary quality, which is what separates a usable mask from a blob.
    """
    per_thr: dict[float, float] = {}
    for thr in MASK_IOU_THRESHOLDS:
        ap = mask_average_precision(pred_masks, pred_scores, pred_classes, gt_masks, gt_classes, thr)
        if ap is not None:
            per_thr[thr] = round(ap, 6)
    if not per_thr:
        return {"ap50": None, "ap50_95": None, "per_threshold": {}}
    return {
        "ap50": per_thr.get(0.5),
        "ap50_95": round(float(np.mean(list(per_thr.values()))), 6),
        "per_threshold": per_thr,
    }


def boundary_f1(pred: npt.NDArray[np.bool_], gt: npt.NDArray[np.bool_], tolerance_px: int = 2) -> float:
    """Boundary F1: how well the mask edges agree, within a pixel tolerance.

    IoU is dominated by an object's interior, so a mask can score well while tracing the silhouette badly.
    For annotation quality the boundary is what a human is actually judged on, so this is measured
    separately rather than folded into the IoU number.
    """
    import cv2

    if pred.shape != gt.shape:
        raise ValueError("mask shapes differ")
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    def _edges(m: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
        u = m.astype(np.uint8)
        eroded = cv2.erode(u, np.ones((3, 3), np.uint8), iterations=1)
        return (u - eroded).astype(np.uint8)

    pe, ge = _edges(pred), _edges(gt)
    k = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
    pe_d = cv2.dilate(pe, k, iterations=1).astype(bool)
    ge_d = cv2.dilate(ge, k, iterations=1).astype(bool)

    n_pe, n_ge = int(pe.sum()), int(ge.sum())
    if n_pe == 0 or n_ge == 0:
        return 0.0
    precision = float(np.count_nonzero(pe.astype(bool) & ge_d)) / n_pe
    recall = float(np.count_nonzero(ge.astype(bool) & pe_d)) / n_ge
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
