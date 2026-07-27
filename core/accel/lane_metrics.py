"""Lane-detection metrics: the CULane-style F1 at a lateral distance tolerance.

The system proposes lanes (CLRerNet on the pod, a classical Canny/Hough fallback locally), stores them as
control-point splines, and lets a human edit them. It could not score any of that: there was no lane metric
anywhere, so "did the lane model get better" had no answer and a lane model could never be gated.

A lane is a curve, not a box, so IoU does not apply. The accepted measure samples both curves at fixed
heights and calls them matched when the mean lateral offset is within a tolerance, which is what CULane and
TuSimple do and what makes the number comparable to published figures.
"""

from __future__ import annotations

import numpy as np

# CULane matches at 30px on 1640x590 imagery. Expressed as a fraction of image width so it transfers across
# resolutions instead of silently tightening on a larger image.
DEFAULT_TOLERANCE_FRAC = 30.0 / 1640.0


def sample_lane_at_rows(points: list[list[float]], rows: np.ndarray) -> np.ndarray:
    """Interpolate a lane's x at each sample row. NaN where the lane does not span that row.

    Lanes are stored as sparse control points and rarely share y positions between prediction and ground
    truth, so both are resampled onto a common row grid before comparison. Without that the two curves are
    not comparable at all.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return np.full(rows.shape, np.nan)
    order = np.argsort(pts[:, 1])
    ys, xs = pts[order, 1], pts[order, 0]
    out = np.interp(rows, ys, xs, left=np.nan, right=np.nan)
    # np.interp clamps rather than extrapolating, so mask anything outside the lane's own y range: a lane
    # must not be credited for a region it never covered.
    return np.where((rows >= ys[0]) & (rows <= ys[-1]), out, np.nan)


def lane_distance(pred: list[list[float]], gt: list[list[float]], height: int,
                  n_samples: int = 20) -> float:
    """Mean lateral distance in pixels between two lanes over their shared vertical extent.

    Returns inf when they never overlap vertically, which keeps an unrelated lane from matching by accident.
    """
    rows = np.linspace(0, max(height - 1, 1), n_samples)
    px, gx = sample_lane_at_rows(pred, rows), sample_lane_at_rows(gt, rows)
    both = ~np.isnan(px) & ~np.isnan(gx)
    if not both.any():
        return float("inf")
    return float(np.mean(np.abs(px[both] - gx[both])))


def match_lanes(preds: list[list[list[float]]], gts: list[list[list[float]]], width: int, height: int,
                tolerance_frac: float = DEFAULT_TOLERANCE_FRAC) -> dict:
    """Greedy one-to-one lane match at a lateral tolerance, closest pair first."""
    tol_px = tolerance_frac * width
    candidates = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            d = lane_distance(p, g, height)
            if d <= tol_px:
                candidates.append((d, i, j))
    candidates.sort()

    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for d, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j, d))

    return {"pairs": pairs, "tp": len(pairs), "fp": len(preds) - len(pairs), "fn": len(gts) - len(pairs),
            "tolerance_px": round(tol_px, 2)}


def lane_f1(preds: list[list[list[float]]], gts: list[list[list[float]]], width: int, height: int,
            tolerance_frac: float = DEFAULT_TOLERANCE_FRAC) -> dict:
    """Precision, recall, and F1 over lanes, plus the mean lateral error of the matched ones.

    F1 says how many lanes were found; the mean offset says how well they were placed. A model can hold its
    F1 while drifting laterally, and for a lane that offset is the part that matters downstream.
    """
    if not gts:
        return {"measured": False, "reason": "no lane ground truth", "support": 0}

    m = match_lanes(preds, gts, width, height, tolerance_frac)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    offsets = [d for _, _, d in m["pairs"]]

    return {
        "measured": True,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "support": len(gts),
        "mean_lateral_error_px": round(float(np.mean(offsets)), 3) if offsets else None,
        "tolerance_px": m["tolerance_px"],
    }
