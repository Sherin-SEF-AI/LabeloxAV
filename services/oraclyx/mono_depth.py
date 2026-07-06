"""ORACLYX monocular metric-depth prior (M14). When a session has no LiDAR, ORACLYX still needs a metric depth
to place an object in 3D for the pseudo-GT. Two priors recover it from a single camera: the ground-plane
prior (a road-contact box bottom back-projects to a range through the calibrated camera height and pitch) and
the known-size prior (a class with a known real-world height fixes range from its pixel height and the focal
length). The two are fused inversely by their variances, so the more certain prior dominates and the result
carries an uncertainty the soft-target path can consume. Capability-honest: with no calibration the estimate is
refused (returns None) rather than guessed. Pure math, so the recovered depth is testable against a synthetic
scene."""

from __future__ import annotations

import math

# nominal real-world heights (metres) for the India ontology classes that carry a stable size prior
CLASS_HEIGHT_M = {
    0: 1.7,    # pedestrian
    1: 1.7,    # rider
    2: 1.5,    # motorcycle
    3: 1.5,    # bicycle
    4: 1.5,    # car
    5: 3.2,    # truck
    6: 1.6,    # autorickshaw
    7: 3.1,    # bus
    8: 1.3,    # cattle
}


def depth_from_ground(box: list[float], cam_height_m: float, focal_px: float, cy: float,
                      pitch_rad: float = 0.0) -> float | None:
    """Range from the ground-contact prior: the box bottom, back-projected through a camera at cam_height_m,
    gives the ground distance. Returns None when the contact point is at or above the horizon (no intersection)."""
    if focal_px <= 0 or cam_height_m <= 0:
        return None
    y_bottom = box[3]
    # angle below the optical axis to the contact ray, corrected for camera pitch
    theta = math.atan2(y_bottom - cy, focal_px) + pitch_rad
    if theta <= 1e-4:
        return None
    return cam_height_m / math.tan(theta)


def depth_from_size(box: list[float], class_id: int, focal_px: float) -> float | None:
    """Range from the known-size prior: real height * focal / pixel height. Returns None when the class has no
    size prior or the box is degenerate."""
    h_px = box[3] - box[1]
    real_h = CLASS_HEIGHT_M.get(class_id)
    if real_h is None or h_px <= 1 or focal_px <= 0:
        return None
    return real_h * focal_px / h_px


def metric_depth(box: list[float], class_id: int, focal_px: float, cy: float,
                 cam_height_m: float | None = None, pitch_rad: float = 0.0) -> dict | None:
    """Fuse the ground and size priors into one metric depth with an uncertainty.

    Each available prior is weighted by the inverse of a heuristic variance (ground grows uncertain far away and
    near the horizon; size grows uncertain for small pixel heights). Returns {depth_m, uncertainty, priors}
    where uncertainty is a relative sigma in [0, 1]; None when neither prior is available (no calibration and no
    size prior)."""
    estimates = []
    d_ground = depth_from_ground(box, cam_height_m, focal_px, cy, pitch_rad) if cam_height_m else None
    if d_ground is not None and d_ground > 0:
        # ground prior variance grows with range squared (contact-pixel error amplifies far away)
        var = (0.03 * d_ground) ** 2 + 1.0
        estimates.append((d_ground, var, "ground"))
    d_size = depth_from_size(box, class_id, focal_px)
    if d_size is not None and d_size > 0:
        h_px = box[3] - box[1]
        # size prior variance grows as the object shrinks in pixels and with size-prior spread
        var = (0.15 * d_size) ** 2 + (20.0 / max(h_px, 1.0)) ** 2
        estimates.append((d_size, var, "size"))
    if not estimates:
        return None

    wsum = sum(1.0 / v for _, v, _ in estimates)
    depth = sum(d / v for d, v, _ in estimates) / wsum
    fused_var = 1.0 / wsum
    rel_sigma = min(1.0, math.sqrt(fused_var) / max(depth, 1e-3))
    return {"depth_m": round(depth, 3), "uncertainty": round(rel_sigma, 4),
            "priors": [name for _, _, name in estimates]}
