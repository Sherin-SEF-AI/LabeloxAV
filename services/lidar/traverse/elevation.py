"""Elevation profile from the ground points: slope, ramp, bridge, and flyover. Relevant for Indian flyovers
and ramps. The ground height is sampled per forward distance bin; a sustained climb is a ramp, and a raised
section that returns to grade is a bridge or flyover.
"""

from __future__ import annotations

import numpy as np

from services.lidar.extract.common import height_above_plane
from services.lidar.ingest.normalize import Cloud


def elevation_profile(cloud: Cloud, plane: list[float], bin_m: float = 5.0,
                      x_range: tuple[float, float] = (0.0, 80.0)) -> dict:
    """Ground elevation per forward bin, the max slope, and any ramp / flyover feature detected."""
    near_ground = np.abs(height_above_plane(cloud.xyz, plane)) < 0.4
    g = cloud.xyz[near_ground]
    if len(g) < 30:
        return {"profile": [], "max_slope": 0.0, "feature": "flat", "n_points": int(len(g))}

    bins = np.arange(x_range[0], x_range[1] + bin_m, bin_m)
    profile = []
    for i in range(len(bins) - 1):
        m = (g[:, 0] >= bins[i]) & (g[:, 0] < bins[i + 1])
        if m.sum() < 3:
            continue
        profile.append([round(float((bins[i] + bins[i + 1]) / 2.0), 1), round(float(np.median(g[m, 2])), 3)])

    if len(profile) < 2:
        return {"profile": profile, "max_slope": 0.0, "feature": "flat", "n_points": int(len(g))}

    xs = np.array([p[0] for p in profile])
    zs = np.array([p[1] for p in profile])
    slopes = np.diff(zs) / np.clip(np.diff(xs), 1e-3, None)
    max_slope = float(np.max(np.abs(slopes)))
    rise = float(zs.max() - zs.min())

    feature = "flat"
    if max_slope > 0.04 and rise > 1.5:
        # a section that climbs and then returns toward grade is a flyover/bridge; a one-way climb is a ramp
        feature = "flyover" if (zs[-1] - zs.min() < rise * 0.5) else "ramp"
    elif max_slope > 0.02:
        feature = "incline"
    return {"profile": profile, "max_slope": round(max_slope, 4), "rise_m": round(rise, 2),
            "feature": feature, "n_points": int(len(g))}
