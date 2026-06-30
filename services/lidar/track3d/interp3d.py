"""Keyframe cuboid interpolation along a 3D track, mirroring the 2D keyframe interpolation
(services/temporal/interpolate.py): an annotator labels a cuboid at keyframes and the full pose (centre,
dimensions, yaw) is filled in between, with the filled frames marked as interpolated. Centre and dimensions
interpolate linearly; yaw takes the shortest angular path.
"""

from __future__ import annotations

import math


def _angle_lerp(a: float, b: float, t: float) -> float:
    """Interpolate an angle along the shortest path and wrap to (-pi, pi]."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return math.atan2(math.sin(a + t * d), math.cos(a + t * d))


def _lerp_vec(a, b, t):
    return [float(x + (y - x) * t) for x, y in zip(a, b, strict=False)]


def interpolate_cuboids(keyframes: list[dict], ts_list: list[int]) -> list[dict]:
    """Fill cuboids at each ts in ts_list that lies strictly between two keyframes. keyframes is a list of
    {ts_ns, center, dims, yaw} sorted by ts_ns; returns one interpolated cuboid per fillable ts, marked
    interp_source=linear. Timestamps outside the keyframe span (or already on a keyframe) are skipped."""
    kf = sorted(keyframes, key=lambda k: k["ts_ns"])
    if len(kf) < 2:
        return []
    kf_ts = {k["ts_ns"] for k in kf}
    out = []
    for ts in sorted(ts_list):
        if ts in kf_ts or ts < kf[0]["ts_ns"] or ts > kf[-1]["ts_ns"]:
            continue
        j = next(i for i in range(len(kf) - 1) if kf[i]["ts_ns"] <= ts <= kf[i + 1]["ts_ns"])
        a, b = kf[j], kf[j + 1]
        span = b["ts_ns"] - a["ts_ns"]
        t = (ts - a["ts_ns"]) / span if span else 0.0
        out.append({"ts_ns": ts, "center": _lerp_vec(a["center"], b["center"], t),
                    "dims": _lerp_vec(a["dims"], b["dims"], t),
                    "yaw": round(_angle_lerp(a["yaw"], b["yaw"], t), 5), "interp_source": "linear"})
    return out
