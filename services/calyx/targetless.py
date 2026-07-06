"""CALYX targetless calibration (M11). Calibrate from natural-scene cues so no checkerboard or field target is
needed: two orthogonal vanishing points (from lane lines and the road's cross structure) recover the focal
length, and the horizon line (the ground plane's vanishing line) recovers the camera pitch. This lets a field
session self-calibrate from the scene it already recorded.

Pure projective geometry, so it validates against a synthetic scene with known intrinsics in the test."""

from __future__ import annotations

import math

import numpy as np


def focal_from_vanishing_points(vp1, vp2, cx: float, cy: float) -> float | None:
    """Focal length from two vanishing points of orthogonal world directions. For orthogonal directions the
    principal-point-centred vanishing points satisfy (v1-c).(v2-c) = -f^2, so f = sqrt(-(v1-c).(v2-c)). Returns
    None when the geometry is inconsistent (the dot product is non-negative, i.e. the directions are not
    orthogonal in view)."""
    v1 = np.asarray(vp1, dtype=np.float64) - np.array([cx, cy])
    v2 = np.asarray(vp2, dtype=np.float64) - np.array([cx, cy])
    dot = float(v1 @ v2)
    if dot >= 0:
        return None
    return math.sqrt(-dot)


def pitch_from_horizon(horizon_y: float, cy: float, fy: float) -> float:
    """Camera pitch (degrees) from the horizon line's image row. The horizon is the ground plane's vanishing
    line; its offset from the principal point is fy*tan(pitch)."""
    return math.degrees(math.atan2(cy - horizon_y, fy))


def calibrate_targetless(vp1, vp2, horizon_y: float, cx: float, cy: float) -> dict:
    """Recover focal and pitch from natural-scene cues. Returns the estimate plus a confidence gated by the
    geometry consistency."""
    f = focal_from_vanishing_points(vp1, vp2, cx, cy)
    if f is None:
        return {"ok": False, "reason": "vanishing points not orthogonal in view", "focal": None, "pitch_deg": None}
    pitch = pitch_from_horizon(horizon_y, cy, f)
    return {"ok": True, "focal": round(f, 2), "pitch_deg": round(pitch, 3),
            "cx": cx, "cy": cy, "method": "vanishing_points+horizon"}
