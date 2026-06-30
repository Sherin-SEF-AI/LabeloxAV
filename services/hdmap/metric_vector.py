"""Milestone F: 3D metric polylines and polygons (lane boundaries, stop lines, crosswalks) drawn in the
metric ego/BEV frame and georeferenced into world space, reusing the existing HD-map store. No new model: a
metric polyline IS a MapElement LineString, georeferenced through the existing vehicle_to_world. The ego
frame is x forward, y left; vehicle_to_world takes lateral as +right, so lateral = -y.
"""

from __future__ import annotations

import math

from core.logging import get_logger
from services.hdmap.georef import vehicle_to_world

log = get_logger("metric_vector")

_KINDS = {"lane", "road_edge", "stop_line", "crosswalk", "crossing"}


def polyline_length_m(points_ego: list) -> float:
    """Metric length of an ego-frame polyline [[x, y], ...] in metres."""
    if len(points_ego) < 2:
        return 0.0
    return sum(math.dist(points_ego[i], points_ego[i + 1]) for i in range(len(points_ego) - 1))


def ego_polyline_to_world(points_ego: list, lat: float, lon: float, heading_rad: float) -> list[tuple]:
    """Each ego point (x forward, y left) to world, returned as (lon, lat) for WKT. lateral = -y because
    vehicle_to_world measures lateral to the right."""
    out = []
    for x, y in points_ego:
        wlat, wlon = vehicle_to_world(x, -y, lat, lon, heading_rad)
        out.append((wlon, wlat))
    return out


async def create_metric_element(session_id, kind: str, points_ego: list, ref_lat: float, ref_lon: float,
                                heading_rad: float, confidence: float = 1.0) -> dict:
    """Georeference a metric ego-frame polyline and store it as a MapElement LineString in world space."""
    from sqlalchemy import func

    from db.models import MapElement
    from db.session import get_sessionmaker
    if kind not in _KINDS:
        return {"error": f"unknown map element kind {kind}"}
    if len(points_ego) < 2:
        return {"error": "a polyline needs at least two points"}
    world = ego_polyline_to_world(points_ego, ref_lat, ref_lon, heading_rad)
    wkt = "LINESTRING(" + ", ".join(f"{lon} {lat}" for lon, lat in world) + ")"
    length = round(polyline_length_m(points_ego), 3)
    async with get_sessionmaker()() as db:
        el = MapElement(kind=kind, geometry=func.ST_GeogFromText(wkt),
                        attrs={"metric_length_m": length, "frame": "ego_metric", "n_points": len(points_ego)},
                        source_sessions=[str(session_id)], confidence=confidence)
        db.add(el)
        await db.commit()
        await db.refresh(el)
        eid = str(el.element_id)
    log.info("metric_vector.created", kind=kind, length_m=length, element=eid)
    return {"element_id": eid, "kind": kind, "metric_length_m": length, "points": len(points_ego)}
