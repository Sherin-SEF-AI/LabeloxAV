"""Auto-computed properties for a 3D object: distance, heading, velocity, and acceleration from the cuboid,
the 3D track, and ego motion, plus occlusion from camera visibility and point density. Reuses the 3D track
trajectory (M-L2.2) and the ego speed; the velocity is ego-motion compensated so a parked object reads zero.
"""

from __future__ import annotations

import math


def _world_speed_series(traj: list[dict], ego_speed: float) -> list[float]:
    out = []
    for a, b in zip(traj[:-1], traj[1:], strict=False):
        dt = (b["ts_ns"] - a["ts_ns"]) / 1e9
        if dt <= 0:
            continue
        vx = (b["center"][0] - a["center"][0]) / dt + ego_speed   # ego-compensated forward velocity
        vy = (b["center"][1] - a["center"][1]) / dt
        out.append(math.hypot(vx, vy))
    return out


def compute_object_properties(center, dims, yaw: float, *, trajectory: list[dict] | None = None,
                              ego_speed: float = 0.0, in_image_frac: float | None = None,
                              points_in_box: int | None = None) -> dict:
    """Return the auto-computed properties for one 3D object, suitable for object_3d.attrs."""
    distance = math.hypot(center[0], center[1])
    heading_deg = math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))

    velocity, acceleration = 0.0, 0.0
    if trajectory and len(trajectory) >= 2:
        speeds = _world_speed_series(sorted(trajectory, key=lambda p: p["ts_ns"]), ego_speed)
        if speeds:
            velocity = speeds[-1]
            if len(speeds) >= 2:
                # acceleration over the last interval (uses the trajectory timestamps for dt)
                ts = sorted(p["ts_ns"] for p in trajectory)
                dt = (ts[-1] - ts[-2]) / 1e9
                acceleration = (speeds[-1] - speeds[-2]) / dt if dt > 0 else 0.0

    # occlusion: how much of the box is out of camera view, plus a sparsity penalty from point density
    occlusion = 0.0
    if in_image_frac is not None:
        occlusion = 1.0 - max(0.0, min(1.0, in_image_frac))
    if points_in_box is not None:
        expected = max(20.0, 400.0 / max(distance, 1.0))   # nearer boxes should have more points
        sparsity = 1.0 - min(1.0, points_in_box / expected)
        occlusion = max(occlusion, 0.5 * sparsity)

    return {"distance_m": round(distance, 2), "heading_deg": round(heading_deg, 1),
            "velocity_mps": round(velocity, 2), "acceleration_mps2": round(acceleration, 2),
            "occlusion": round(occlusion, 3),
            "points_in_box": int(points_in_box) if points_in_box is not None else None}
