"""Geo-reference image-space annotations to world (M3.3): lift M2.1 lanes (and M2.3 signs) from pixels to
WGS84 via inverse perspective mapping, the flat road-plane homography from the validated intrinsics +
camera height, composed with the frame's GNSS position and a GNSS-track-derived heading. Writes
map_element rows with full provenance (source frames, source sessions, calibration version). Gated on
calibration: a session failing M3.0 is excluded. Lanes lie on the road plane so IPM is exact; signs are
lifted by their bbox base as an approximation (cross-frame triangulation is the upgrade seam).
"""

from __future__ import annotations

import math
from uuid import UUID

from geoalchemy2.elements import WKTElement
from sqlalchemy import cast, delete, func, select
from geoalchemy2 import Geometry

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Lane, MapElement, Object
from db.session import get_sessionmaker
from services.calibration.report import session_calibrated

log = get_logger("hdmap_georef")

CALIB_VERSION = "labelox-calib-0.1"
_LANE_CONF = {"human": 0.95, "propagated": 0.7, "proposed": 0.6}


def ipm_pixel_to_vehicle(u: float, v: float, fx: float, fy: float, cx: float, cy: float,
                         height_m: float, pitch_rad: float = 0.0) -> tuple[float, float] | None:
    """Flat-ground IPM: a pixel to (forward, lateral) metres in the vehicle frame. None above the horizon.
    Camera frame x-right, y-down, z-forward; the road plane is height_m below the camera."""
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    if pitch_rad:  # rotate the ray about the camera x-axis (downward pitch raises the horizon)
        cy_, sy_ = math.cos(pitch_rad), math.sin(pitch_rad)
        y, z = y * cy_ - z * sy_, y * sy_ + z * cy_
    if y <= 1e-6:  # at or above the horizon, no ground intersection
        return None
    s = height_m / y
    forward, lateral = s * z, s * x
    if forward <= 0:
        return None
    return forward, lateral


def vehicle_to_world(forward: float, lateral: float, lat: float, lon: float,
                     heading_rad: float) -> tuple[float, float]:
    """Vehicle-frame (forward, lateral) to world (lat, lon) given the GNSS pos + heading (rad from north,
    clockwise). Lateral is +right of travel."""
    east = forward * math.sin(heading_rad) + lateral * math.cos(heading_rad)
    north = forward * math.cos(heading_rad) - lateral * math.sin(heading_rad)
    return lat + north / 111320.0, lon + east / (111320.0 * math.cos(math.radians(lat)))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    return math.atan2(y, x)


async def georef_session(session_id: UUID, height_m: float | None = None) -> dict:
    cfg = get_settings()
    if not await session_calibrated(session_id):
        return {"elements": 0, "reason": "session not calibrated (run /api/calibration/validate first)"}
    height_m = height_m if height_m is not None else cfg.spatial.camera_height_m
    pitch = math.radians(cfg.spatial.camera_pitch_deg)

    maker = get_sessionmaker()
    async with maker() as db:
        geom = cast(Frame.gnss, Geometry)
        frames = (await db.execute(
            select(Frame.frame_id, Frame.cam_id, func.ST_Y(geom), func.ST_X(geom), Frame.width, Frame.height)
            .where(Frame.session_id == session_id, Frame.gnss.isnot(None)).order_by(Frame.ts_ns))).all()
        if not frames:
            return {"elements": 0, "reason": "no GNSS frames in this session"}
        pts = [(fid, cam, float(lat), float(lon), w or 1920, h or 1080) for fid, cam, lat, lon, w, h in frames]

        # re-georef cleanly: drop prior elements sourced from this session
        await db.execute(delete(MapElement).where(MapElement.source_sessions.any(str(session_id))))

        n_lane, n_sign = 0, 0
        for i, (fid, cam, lat, lon, w, h) in enumerate(pts):
            if i + 1 < len(pts):
                hd = bearing(lat, lon, pts[i + 1][2], pts[i + 1][3])
            elif i > 0:
                hd = bearing(pts[i - 1][2], pts[i - 1][3], lat, lon)
            else:
                hd = 0.0
            lens = cfg.rig.camera_lens.get(cam, "narrow")
            K = cfg.rig.lenses[lens]
            # scale nominal intrinsics (defined at ref_width) to this frame's resolution; principal point
            # at the image centre (real per-camera intrinsics, when ingested, override this).
            scale = w / cfg.rig.ref_width
            fx, fy, cx, cy = K.fx * scale, K.fy * scale, w / 2.0, h / 2.0

            for lane in (await db.execute(select(Lane).where(Lane.frame_id == fid))).scalars().all():
                world = []
                for pt in lane.control_points:
                    fl = ipm_pixel_to_vehicle(pt[0], pt[1], fx, fy, cx, cy, height_m, pitch)
                    if fl is None:
                        continue
                    wlat, wlon = vehicle_to_world(fl[0], fl[1], lat, lon, hd)
                    world.append((wlon, wlat))
                if len(world) < 2:
                    continue
                wkt = "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in world) + ")"
                db.add(MapElement(kind="lane", geometry=WKTElement(wkt, srid=4326),
                                  attrs={"lane_type": lane.lane_type, "is_ego": lane.is_ego, "source": lane.source},
                                  source_frames=[str(fid)], source_sessions=[str(session_id)],
                                  calibration_version=CALIB_VERSION, confidence=_LANE_CONF.get(lane.source, 0.6)))
                n_lane += 1

            for s in (await db.execute(
                    select(Object).where(Object.frame_id == fid, Object.sign_type.isnot(None)))).scalars().all():
                bb = s.bbox
                fl = ipm_pixel_to_vehicle((bb[0] + bb[2]) / 2.0, bb[3], fx, fy, cx, cy, height_m, pitch)
                if fl is None:
                    continue
                wlat, wlon = vehicle_to_world(fl[0], fl[1], lat, lon, hd)
                db.add(MapElement(kind="sign", geometry=WKTElement(f"POINT({wlon} {wlat})", srid=4326),
                                  attrs={"sign_type": s.sign_type, "sign_category": s.sign_category, "approx": True},
                                  source_frames=[str(fid)], source_sessions=[str(session_id)],
                                  calibration_version=CALIB_VERSION, confidence=float(s.conf or 0.5)))
                n_sign += 1
        await db.commit()

    out = {"session_id": str(session_id), "elements": n_lane + n_sign, "lanes": n_lane, "signs": n_sign,
           "frames": len(pts)}
    log.info("hdmap.georef", **{k: out[k] for k in ("elements", "lanes", "signs")})
    return out
