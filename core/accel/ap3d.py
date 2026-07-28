"""3D and bird's-eye-view average precision for cuboid detection.

The system annotates 3D cuboids (`Object3D`, the LiDAR workspace, the lift-from-2D path) and its quality
checker flags physically impossible boxes, but nothing ever scored a 3D prediction against 3D ground truth,
so "is the 3D any good" had no answer. The geometry primitives already existed in services/lidar/boxes.py;
what was missing was the matching and the AP integral on top of them.

Two IoU variants are reported because they answer different questions. BEV IoU ignores height and is what
planning mostly cares about (footprint on the road). Full 3D IoU includes the vertical extent and is the
stricter measure. KITTI and nuScenes both report the pair for this reason.
"""

from __future__ import annotations

import numpy as np

from core.accel.ap import average_precision
from services.lidar.boxes import bev_intersection_area, iou_3d

# KITTI-style operating points. 3D IoU thresholds are far lower than 2D by convention because the task is
# harder: 0.7 for vehicles, 0.5 for pedestrians and cyclists.
IOU_3D_VEHICLE = 0.7
IOU_3D_VRU = 0.5


def bev_iou(a: dict, b: dict) -> float:
    """Bird's-eye-view IoU: footprint overlap, ignoring height.

    A cuboid is {center: [x,y,z], dims: [w,l,h], yaw}. Two boxes that overlap on the ground but differ in
    height score 1.0 here and less under iou_3d, which is exactly the distinction the pair is meant to make.
    """
    inter = bev_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = float(a["dims"][0]) * float(a["dims"][1])
    area_b = float(b["dims"][0]) * float(b["dims"][1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_cuboids(preds: list[dict], pred_scores: list[float], pred_classes: list[int],
                  gts: list[dict], gt_classes: list[int], iou_thr: float = 0.5,
                  mode: str = "3d") -> dict:
    """Greedy class-aware match of predicted cuboids to ground truth, mirroring the 2D matcher.

    mode is "3d" (volumetric) or "bev" (footprint). Same rule as the box and mask paths: descending score,
    each prediction claims the best unclaimed same-class ground truth at or above the threshold.
    """
    iou_fn = iou_3d if mode == "3d" else bev_iou
    order = sorted(range(len(preds)), key=lambda i: -float(pred_scores[i]))
    gt_taken = [False] * len(gts)
    tp = [False] * len(preds)

    for i in order:
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if gt_taken[j] or gt_classes[j] != pred_classes[i]:
                continue
            iou = iou_fn(preds[i], g)
            if iou >= best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            gt_taken[best_j] = True
            tp[i] = True

    n_tp = sum(tp)
    return {"tp": [tp[i] for i in order], "scores": [float(pred_scores[i]) for i in order],
            "n_tp": n_tp, "n_fp": len(preds) - n_tp, "n_fn": len(gts) - n_tp,
            "matched_gt": gt_taken}


def cuboid_average_precision(preds: list[dict], pred_scores: list[float], pred_classes: list[int],
                             gts: list[dict], gt_classes: list[int], iou_thr: float = 0.5,
                             mode: str = "3d") -> float | None:
    """AP for cuboids at one IoU threshold, 3D or BEV. None when there is no ground truth."""
    if not gts:
        return None
    m = match_cuboids(preds, pred_scores, pred_classes, gts, gt_classes, iou_thr, mode)
    return average_precision(m["scores"], m["tp"], len(gts))


def translation_error(preds: list[dict], gts: list[dict], matched_pairs: list[tuple[int, int]]) -> float | None:
    """Mean centre distance over matched pairs, in metres (the nuScenes mATE idea).

    AP says whether a box was found; this says how far off it was. A detector can hold its AP while drifting
    systematically in range, which matters for anything downstream that uses the position.
    """
    if not matched_pairs:
        return None
    errs = [float(np.linalg.norm(np.asarray(preds[i]["center"], dtype=float)
                                 - np.asarray(gts[j]["center"], dtype=float)))
            for i, j in matched_pairs]
    return float(np.mean(errs))


def orientation_error(preds: list[dict], gts: list[dict],
                      matched_pairs: list[tuple[int, int]]) -> float | None:
    """Mean absolute yaw error over matched pairs, in radians, wrapped to [0, pi].

    Wrapping matters: a box rotated by pi is the same box for a symmetric vehicle, so the raw difference
    would report a large error for a correct detection.
    """
    if not matched_pairs:
        return None
    errs = []
    for i, j in matched_pairs:
        d = abs(float(preds[i].get("yaw", 0.0)) - float(gts[j].get("yaw", 0.0))) % (2 * np.pi)
        errs.append(min(d, 2 * np.pi - d))
    return float(np.mean(errs))


def evaluate_cuboids(preds: list[dict], pred_scores: list[float], pred_classes: list[int],
                     gts: list[dict], gt_classes: list[int], iou_thr: float = 0.5) -> dict:
    """The 3D panel: AP in both modes plus the localisation errors on the matched set."""
    if not gts:
        return {"measured": False, "reason": "no 3D ground truth", "support": 0}

    m3 = match_cuboids(preds, pred_scores, pred_classes, gts, gt_classes, iou_thr, "3d")
    order = sorted(range(len(preds)), key=lambda i: -float(pred_scores[i]))
    pairs: list[tuple[int, int]] = []
    taken = [False] * len(gts)
    for i in order:
        for j, g in enumerate(gts):
            if taken[j] or gt_classes[j] != pred_classes[i]:
                continue
            if iou_3d(preds[i], g) >= iou_thr:
                taken[j] = True
                pairs.append((i, j))
                break

    return {
        "measured": True,
        "support": len(gts),
        "ap_3d": round(cuboid_average_precision(preds, pred_scores, pred_classes, gts, gt_classes,
                                                iou_thr, "3d") or 0.0, 4),
        "ap_bev": round(cuboid_average_precision(preds, pred_scores, pred_classes, gts, gt_classes,
                                                 iou_thr, "bev") or 0.0, 4),
        "tp": m3["n_tp"], "fp": m3["n_fp"], "fn": m3["n_fn"],
        "translation_error_m": (round(v, 4) if (v := translation_error(preds, gts, pairs)) is not None else None),
        "orientation_error_rad": (round(v, 4) if (v := orientation_error(preds, gts, pairs)) is not None else None),
        "iou_thr": iou_thr,
    }
