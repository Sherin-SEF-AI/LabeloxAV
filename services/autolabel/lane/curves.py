"""Lane geometry (M2.1): polyline to editable spline control points, ego-lane marking, and cross-frame
propagation of control points by optical flow (so the annotator corrects keyframes, not every frame)."""

from __future__ import annotations

import cv2
import numpy as np

from core.config import get_settings


def fit_control_points(polyline: list, k: int | None = None) -> list[list[float]]:
    """Resample a polyline to k evenly-spaced control points (the editable spline handles)."""
    k = k or get_settings().models.lane.control_points
    pts = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return [[float(p[0]), float(p[1])] for p in pts] or [[0.0, 0.0]]
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if d[-1] == 0:
        return [[float(pts[0][0]), float(pts[0][1])] for _ in range(k)]
    u = np.linspace(0.0, d[-1], k)
    x = np.interp(u, d, pts[:, 0])
    y = np.interp(u, d, pts[:, 1])
    return [[float(a), float(b)] for a, b in zip(x, y)]


def mark_ego(lanes_cp: list[list[list[float]]], width: int, height: int) -> int | None:
    """The ego lane line is the one whose lowest point sits nearest the image bottom-center."""
    cx = width / 2
    best, bd = None, 1e18
    for i, cp in enumerate(lanes_cp):
        bottom = max(cp, key=lambda p: p[1])
        dist = abs(bottom[0] - cx) + (height - bottom[1]) * 0.2
        if dist < bd:
            bd, best = dist, i
    return best


def propagate_control_points(prev_gray: np.ndarray, cur_gray: np.ndarray, cp: list[list[float]]) -> list[list[float]] | None:
    """Carry control points from one frame to the next via Lucas-Kanade optical flow."""
    p0 = np.asarray(cp, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                         winSize=(21, 21), maxLevel=3,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if p1 is None or st is None or int(st.sum()) < max(2, len(cp) // 2):
        return None
    out = p1.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in out]
