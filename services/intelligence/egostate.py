"""Derived ego-state (M-IMU.1), the foundation of the Plane-4 inertial timeline.

A measured IMU is not landed today (the dashcam freeGPS telemetry blocks are present but empty: no GPS or
G-sensor lock in the current footage), but a real ego-motion signal IS derivable from the GNSS track and CAN
speed already carried on each frame: heading from consecutive fixes, yaw rate from the heading change,
longitudinal acceleration from the speed change, lateral (centripetal) acceleration from speed * yaw_rate,
and jerk from the acceleration change. This is the honest signal (source=derived) the inertial timeline,
event tagging, and anomaly pre-marking ride on; a measured IMU, once ingested, supersedes it the same way
real calibration supersedes nominal.
"""

from __future__ import annotations

import math

from services.hdmap.georef import bearing

_R_EARTH_M = 6371000.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_EARTH_M * math.asin(min(1.0, math.sqrt(a)))


def _wrap_rad(r: float) -> float:
    return (r + math.pi) % (2 * math.pi) - math.pi


def derive_ego_state(samples: list[tuple]) -> list[dict]:
    """Per-sample ego-state from GNSS+speed. samples: [(ts_ns, lat|None, lon|None, speed_mps|None)] sorted by
    ts_ns. Each output row carries speed, heading, yaw_rate (rad/s), longitudinal + lateral acceleration
    (m/s^2) and jerk (m/s^3); fields are None until enough history exists to derive them."""
    out: list[dict] = []
    prev = None
    prev_heading = prev_accel = prev_speed = None
    for ts, lat, lon, speed in samples:
        dt = heading = None
        if prev is not None:
            pts, plat, plon, _ = prev
            dt = (ts - pts) / 1e9
            if dt and dt > 0 and lat is not None and plat is not None and (lat != plat or lon != plon):
                heading = bearing(plat, plon, lat, lon)
                if speed is None:
                    speed = _haversine_m(plat, plon, lat, lon) / dt

        yaw_rate = None
        if heading is not None and prev_heading is not None and dt and dt > 0:
            yaw_rate = _wrap_rad(heading - prev_heading) / dt   # bearing() returns radians
        long_accel = None
        if speed is not None and prev_speed is not None and dt and dt > 0:
            long_accel = (speed - prev_speed) / dt
        lat_accel = speed * yaw_rate if (speed is not None and yaw_rate is not None) else None
        jerk = None
        if long_accel is not None and prev_accel is not None and dt and dt > 0:
            jerk = (long_accel - prev_accel) / dt

        out.append({
            "ts_ns": int(ts),
            "speed_mps": round(speed, 3) if speed is not None else None,
            "heading_deg": round(math.degrees(heading), 2) if heading is not None else None,
            "yaw_rate": round(yaw_rate, 4) if yaw_rate is not None else None,
            "long_accel": round(long_accel, 3) if long_accel is not None else None,
            "lat_accel": round(lat_accel, 3) if lat_accel is not None else None,
            "jerk": round(jerk, 3) if jerk is not None else None,
        })
        prev = (ts, lat, lon, speed)
        if heading is not None:
            prev_heading = heading
        if long_accel is not None:
            prev_accel = long_accel
        if speed is not None:
            prev_speed = speed
    return out


async def session_ego_state(session_id) -> dict:
    """The derived ego-state series for a session, from its frames' GNSS + CAN speed."""
    from geoalchemy2 import Geometry
    from sqlalchemy import cast, func, select

    from db.models import Frame
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        geom = cast(Frame.gnss, Geometry)
        rows = (await db.execute(
            select(Frame.ts_ns, func.ST_Y(geom), func.ST_X(geom), Frame.ego_speed)
            .where(Frame.session_id == session_id).order_by(Frame.ts_ns))).all()
    samples = [(int(ts), lat, lon, sp) for ts, lat, lon, sp in rows]
    series = derive_ego_state(samples)
    n_motion = sum(1 for s in series if s["yaw_rate"] is not None or s["long_accel"] is not None)
    return {"session_id": str(session_id), "source": "derived", "n_samples": len(series),
            "n_with_motion": n_motion, "series": series}
