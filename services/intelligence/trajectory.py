"""Trajectories: project tracks into motion over time. Position uses the box bottom-center as a
ground proxy; velocity/acceleration/heading are derived. Ego-motion (CAN ego_speed, later IMU/GNSS)
is attached so downstream events reason in the road frame rather than the pixel frame.

Full road-frame projection needs camera intrinsics/extrinsics (the calibration seam); until those
land, motion is image-space with ego-speed context, which is enough for the rule-based detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.intelligence.tracking import TrackResult

NS_PER_S = 1_000_000_000


@dataclass
class FrameCtx:
    width: int
    height: int
    ego_speed: float | None
    lat: float | None
    lon: float | None


@dataclass
class Trajectory:
    track_id: str
    points: list[dict]
    summary: dict


def build_trajectory(track: TrackResult, frame_ctx: dict) -> Trajectory:
    pts: list[dict] = []
    prev = None
    for d in track.members:
        ctx: FrameCtx = frame_ctx[d.frame_id]
        x1, y1, x2, y2 = d.bbox
        cx = (x1 + x2) / 2.0
        by = y2  # bottom edge = ground contact proxy
        area = max(1.0, (x2 - x1) * (y2 - y1))
        p = {
            "ts_ns": d.ts_ns,
            "cx": cx,
            "by": by,
            "area": area,
            "ego_speed": ctx.ego_speed,
            "vx": 0.0,
            "vy": 0.0,
            "speed_px": 0.0,
        }
        if prev is not None:
            dt = (d.ts_ns - prev["ts_ns"]) / NS_PER_S
            if dt > 0:
                p["vx"] = (cx - prev["cx"]) / dt
                p["vy"] = (by - prev["by"]) / dt
                p["speed_px"] = (p["vx"] ** 2 + p["vy"] ** 2) ** 0.5
        pts.append(p)
        prev = p

    if not pts:
        return Trajectory(track_id=str(track.track_id), points=[], summary={})

    first, last = pts[0], pts[-1]
    any_ctx: FrameCtx = frame_ctx[track.members[0].frame_id]
    w = max(1, any_ctx.width)
    x_drift = (last["cx"] - first["cx"]) / w
    area_growth = last["area"] / first["area"]
    mean_speed = sum(p["speed_px"] for p in pts) / len(pts)
    net_disp = ((last["cx"] - first["cx"]) ** 2 + (last["by"] - first["by"]) ** 2) ** 0.5

    summary = {
        "n": len(pts),
        "x_drift_frac": round(x_drift, 4),         # +right / -left across the image
        "area_growth": round(area_growth, 4),      # >1 closing toward ego
        "mean_speed_px": round(mean_speed, 3),
        "net_disp_px": round(net_disp, 2),
        "duration_ns": last["ts_ns"] - first["ts_ns"],
        "approaching": area_growth > 1.15,
    }
    return Trajectory(track_id=str(track.track_id), points=pts, summary=summary)
