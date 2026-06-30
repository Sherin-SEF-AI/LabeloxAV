"""Lane polylines from a lane-marking segmentation mask. CLRerNet is the configured pod lane model but is
blocked on the pod's modern torch (mmcv/mmdet have no wheel for torch 2.11/cu128). This derives lanes from
the lane-marking classes the Mapillary Mask2Former segmenter already produces: cluster the marking pixels
into individual lines (connected components), then sample each line into a control-point polyline in the
Lane.control_points shape. Pure over a binary mask, so it is tested without a model or a pod.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.logging import get_logger

log = get_logger("lane_marking")


def lanes_from_marking_mask(mask, min_pixels: int = 80, min_height: int = 20, n_points: int = 8,
                            max_lanes: int = 8) -> list[list[list[float]]]:
    """mask: HxW bool/uint8 of lane-marking pixels. Returns up to max_lanes polylines [[x, y], ...], one per
    marking line, ordered longest first. A line must have >= min_pixels and span >= min_height rows to count
    (filters specks and short dashes' noise; a real lane line is tall and thin)."""
    m = np.asarray(mask).astype(np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    n_labels, labels = cv2.connectedComponents(m)
    lanes: list[list[list[float]]] = []
    for lab in range(1, n_labels):
        ys, xs = np.where(labels == lab)
        if len(xs) < min_pixels:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        if y1 - y0 < min_height:
            continue
        band = max(2.0, (y1 - y0) / (2.0 * n_points))
        pts: list[list[float]] = []
        for yl in np.linspace(y0, y1, n_points):
            sel = np.abs(ys - yl) <= band
            if sel.any():
                pts.append([round(float(xs[sel].mean()), 1), round(float(yl), 1)])
        if len(pts) >= 2:
            lanes.append(pts)
    lanes.sort(key=lambda p: -(p[-1][1] - p[0][1]))           # longest vertical span first
    return lanes[:max_lanes]
