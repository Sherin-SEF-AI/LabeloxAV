"""Lane-line proposals (M2.1). These are starting points for human spline editing, not final labels.

  local fallback: classical CV (Canny + probabilistic Hough) in the road region produces rough lane-line
                  proposals so the workflow is testable without the pod.
  pod path:       CLRerNet (or UFLDv2 by flag), the real proposer, dispatched to the RunPod A100 via the
                  cloud seam (cloud/perception_pod.py) when models.lane.backend=pod.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("lane_detect")


def _classical(image_bgr: np.ndarray, max_lanes: int) -> list[list[list[float]]]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    edges[: int(h * 0.45)] = 0  # only the lower road region
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=int(h * 0.12), maxLineGap=int(h * 0.06))
    if lines is None:
        return []
    cand = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if 20 < ang < 85:  # lane-like slope: skip horizontal markings and vertical poles
            cand.append((math.hypot(x2 - x1, y2 - y1), (x1 + x2) / 2,
                         [[float(x1), float(y1)], [float(x2), float(y2)]]))
    cand.sort(reverse=True)
    kept: list[tuple[float, list]] = []
    for _length, midx, poly in cand:
        if all(abs(midx - mx) > w * 0.06 for mx, _ in kept):  # one proposal per lane corridor
            kept.append((midx, poly))
        if len(kept) >= max_lanes:
            break
    return [poly for _, poly in kept]


def _propose_pod(image_bgr: np.ndarray) -> list[list[list[float]]]:
    raise NotImplementedError(
        "CLRerNet/UFLDv2 lane proposals run on the RunPod pod via cloud/perception_pod.py. Start the pod "
        "and set models.lane.backend=pod, or use the local classical fallback (backend=local).")


def propose_lanes(image_bgr: np.ndarray) -> list[list[list[float]]]:
    """Return lane-line proposals as polylines [[x,y],...] in image coordinates."""
    cfg = get_settings().models.lane
    if cfg.backend == "pod":
        return _propose_pod(image_bgr)
    return _classical(image_bgr, cfg.max_lanes)


def model_tag() -> str:
    cfg = get_settings().models.lane
    return f"{cfg.model}:{cfg.backend}"
