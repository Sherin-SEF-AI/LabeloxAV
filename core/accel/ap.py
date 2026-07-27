"""Average precision (Tier 4 metric primitive). Given, per prediction, a score and whether it is a true
positive, plus the number of ground-truth objects for the class, this computes the 101-point interpolated AP
that COCO uses. It lives beside the other metric kernels (boxes.py, matching.py, slices.py) because it is the
arithmetic the prediction plane's PR curve reduces to once matching has produced the per-prediction tp verdict.

Only computable now that predictions are persisted with their full raw score distribution: the old harness
scored a thresholded residue and could not draw a curve. AP is undefined for a class with no ground truth
(returns None) so a mean is taken over the classes that actually appear, never diluted by empty classes.

NumPy is the oracle; the torch path mirrors it for large prediction sets on the GPU and must agree to 1e-6.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

_N_POINTS = 101
_IOU_50_95 = tuple(round(x, 2) for x in np.arange(0.5, 1.0, 0.05))  # 0.5, 0.55, ..., 0.95


def _ap_np(scores: npt.NDArray[np.float64], tp: npt.NDArray[np.bool_], n_gt: int) -> float:
    order = np.argsort(-scores, kind="stable")
    tp_sorted = tp[order].astype(np.float64)
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1.0 - tp_sorted)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    # 101-point interpolation: at each recall threshold, the max precision achieved at recall >= threshold.
    rec_pts = np.linspace(0.0, 1.0, _N_POINTS)
    ap = 0.0
    for t in rec_pts:
        mask = recall >= t
        ap += float(precision[mask].max()) if bool(mask.any()) else 0.0
    return ap / _N_POINTS


def _ap_torch(scores: npt.NDArray[np.float64], tp: npt.NDArray[np.bool_],
              n_gt: int) -> float:  # pragma: no cover - GPU path
    s = torch.as_tensor(scores, dtype=torch.float64, device="cuda")
    t = torch.as_tensor(tp, dtype=torch.float64, device="cuda")
    order = torch.argsort(s, descending=True, stable=True)
    t = t[order]
    cum_tp = torch.cumsum(t, 0)
    cum_fp = torch.cumsum(1.0 - t, 0)
    recall = cum_tp / n_gt
    precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-12)
    rec_pts = torch.linspace(0.0, 1.0, _N_POINTS, dtype=torch.float64, device="cuda")
    ap = 0.0
    for k in range(_N_POINTS):
        mask = recall >= rec_pts[k]
        ap += float(precision[mask].max()) if bool(mask.any()) else 0.0
    return ap / _N_POINTS


def average_precision(scores: npt.ArrayLike, tp: npt.ArrayLike, n_gt: int,
                      device: str | None = None) -> float | None:
    """101-point interpolated AP for one class. `scores` and `tp` are per prediction (any order); `n_gt` is the
    number of ground-truth objects of the class. Returns None when n_gt == 0 (AP undefined)."""
    if n_gt <= 0:
        return None
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    t = np.asarray(tp).reshape(-1).astype(bool)
    if s.size == 0:
        return 0.0
    if device is not None and str(device) != "cpu" and _HAS_TORCH and torch.cuda.is_available():
        return _ap_torch(s, t, n_gt)
    return _ap_np(s, t, n_gt)


def mean_ap(per_class: dict[int, tuple[npt.ArrayLike, npt.ArrayLike]], gt_counts: dict[int, int],
            device: str | None = None) -> tuple[float, dict[int, float]]:
    """mAP and per-class AP. `per_class` maps class_id -> (scores, tp); `gt_counts` maps class_id -> n_gt.
    Classes with no ground truth are skipped, so the mean is over classes that actually appear."""
    aps: dict[int, float] = {}
    for cid, n_gt in gt_counts.items():
        scores, tp = per_class.get(cid, ([], []))
        ap = average_precision(scores, tp, n_gt, device=device)
        if ap is not None:
            aps[cid] = round(ap, 6)
    m = float(np.mean(list(aps.values()))) if aps else 0.0
    return m, aps


def iou_thresholds_50_95() -> tuple[float, ...]:
    """The ten IoU thresholds AP@[.50:.95] averages over."""
    return _IOU_50_95
