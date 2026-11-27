"""Lane-line proposals (M2.1). These are starting points for human spline editing, not final labels.

  local:     lanes from the lane-marking pixels of the local Mapillary Mask2Former (the same model, and the
             same grouping, as the pod path in cloud/perception_pod.py). Tagged `mapillary-marking:local`.
  fallback:  classical CV (Canny + probabilistic Hough) when the segmenter cannot load. Tagged `hough:local`.
             It finds the strongest straight edges, not paint: a kerb's top edge, one side of a painted line,
             a median barrier. It exists so the workflow is testable without a model, and says so in its tag.
  pod path:  dispatched via the cloud seam (cloud/perception_pod.py) when models.lane.backend=pod.

Every lane on this box used to be tagged `clrernet:local` whatever drew it. CLRerNet never ran locally: the
tag was the configured model name glued to the backend, so Hough lines were labelled as a learned detector's.
The tag now names what actually drew the line.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("lane_detect")

TAG_MARKING = "mapillary-marking:local"
TAG_HOUGH = "hough:local"


def _x_at(poly: list, y: float) -> float:
    """Where a two-point segment's line crosses row y (its own x when it is horizontal)."""
    (x1, y1), (x2, y2) = poly[0], poly[-1]
    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) / 2
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


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
            cand.append((math.hypot(x2 - x1, y2 - y1), ang,
                         [[float(x1), float(y1)], [float(x2), float(y2)]]))
    cand.sort(key=lambda c: -c[0])
    kept: list[tuple[float, float, list]] = []
    for _length, ang, poly in cand:
        # One proposal per line, compared where each line meets the bottom of the frame. Comparing segment
        # midpoints, as this did, keeps two pieces of the same diagonal line as two lanes: they sit at
        # different heights, so their midpoints differ in x even though they are one line.
        xb = _x_at(poly, h)
        if all(abs(xb - kx) > w * 0.06 or abs(ang - ka) > 6 for kx, ka, _ in kept):
            kept.append((xb, ang, poly))
        if len(kept) >= max_lanes:
            break
    return [poly for _, _, poly in kept]


def _from_markings(image_bgr: np.ndarray, max_lanes: int) -> list[list[list[float]]]:
    from services.autolabel.drivable import lane_marking_mask
    from services.autolabel.lane.marking import lanes_from_marking_mask_grouped

    mask = lane_marking_mask(image_bgr)
    return lanes_from_marking_mask_grouped(mask, image_bgr.shape[1], max_lanes=max_lanes)


def _propose_pod(image_bgr: np.ndarray) -> list[list[list[float]]]:
    raise NotImplementedError(
        "CLRerNet/UFLDv2 lane proposals run on the RunPod pod via cloud/perception_pod.py. Start the pod "
        "and set models.lane.backend=pod, or use the local backend (backend=local).")


def propose_lanes_tagged(image_bgr: np.ndarray) -> tuple[list[list[list[float]]], str]:
    """Lane-line proposals as polylines [[x,y],...] in image coordinates, and the tag of what drew them."""
    cfg = get_settings().models.lane
    if cfg.backend == "pod":
        return _propose_pod(image_bgr), f"{cfg.model}:pod"
    if cfg.model != "classical":
        try:
            return _from_markings(image_bgr, cfg.max_lanes), TAG_MARKING
        except Exception as exc:  # noqa: BLE001 - the fallback is named in the tag, not hidden
            log.warning("lane_detect.marking_unavailable_using_hough", error=str(exc))
    return _classical(image_bgr, cfg.max_lanes), TAG_HOUGH


def propose_lanes(image_bgr: np.ndarray) -> list[list[list[float]]]:
    """Return lane-line proposals as polylines [[x,y],...] in image coordinates."""
    return propose_lanes_tagged(image_bgr)[0]


def model_tag() -> str:
    """The tag the configured path writes when it runs as intended."""
    cfg = get_settings().models.lane
    if cfg.backend == "pod":
        return f"{cfg.model}:pod"
    return TAG_HOUGH if cfg.model == "classical" else TAG_MARKING
