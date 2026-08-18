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
from functools import lru_cache

# Nominal real-world heights (metres), BY NAME, for the classes that carry a stable size prior.
#
# This table used to be keyed by id, 0-based, against a 1-based governed ontology - latent bug #1 in
# docs/AV_ASSUMPTIONS.md, and the same defect already fixed once in services/verdyx/safety_recall.py. The
# ids did not mean what their comments said: entry 0 was dead (no such class), `pedestrian`, `rider` and
# `cattle` fell outside the table entirely and silently returned None from the size prior, and id 5 - which
# the comment called "truck" - is `delivery_rider_bike`, so a scooter with a delivery box was modelled as
# 3.2 m tall and placed at roughly twice its real distance.
#
# Names are resolved through the ontology at call time, so an id can never drift out from under this again.
# A class with no entry has no size prior and is refused (None), which is the module's stated contract.
CLASS_HEIGHT_M_BY_NAME: dict[str, float] = {
    # vru
    "pedestrian": 1.70,
    "rider": 1.70,
    "traffic_police": 1.70,
    "street_vendor": 1.70,
    "animal_handler": 1.70,
    "child": 1.20,
    # two-wheelers, measured to the top of the machine rather than the rider
    "motorcycle": 1.50,
    "scooter": 1.40,
    "moped": 1.30,
    "cycle": 1.40,
    "delivery_rider_bike": 1.50,
    # three-wheelers
    "autorickshaw": 1.60,
    "e_auto": 1.70,
    "e_rickshaw": 1.70,
    "cycle_rickshaw": 1.60,
    # four-wheelers. There is no bare "car" in this ontology; the body styles differ enough in height to
    # be worth separating, which the old single 1.5 m "car" entry could not express.
    "hatchback": 1.50,
    "sedan": 1.45,
    "suv": 1.80,
    "taxi": 1.50,
    "app_cab": 1.50,
    "pickup": 1.85,
    "minivan": 1.90,
    # heavy
    "bus": 3.10,
    "school_bus": 3.10,
    "lcv": 2.60,
    "truck": 3.20,
    "water_tanker": 3.20,
    "garbage_truck": 3.20,
    "tipper": 3.20,
    # animals
    "cattle": 1.30,
    "dog": 0.55,
    "goat": 0.65,
}


@lru_cache(maxsize=1)
def _height_by_id() -> dict[int, float]:
    """The name table resolved to the governed ontology's ids.

    Lazy and cached rather than resolved at import, so importing this module does not require an ontology -
    the same shape services/verdyx/safety_recall.py uses. Names absent from the active pack's ontology are
    skipped rather than raising, so a pack with a smaller vocabulary simply has fewer size priors.
    """
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    return {onto.by_name(n).id: h for n, h in CLASS_HEIGHT_M_BY_NAME.items() if onto.has_name(n)}


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
    real_h = _height_by_id().get(int(class_id))
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
