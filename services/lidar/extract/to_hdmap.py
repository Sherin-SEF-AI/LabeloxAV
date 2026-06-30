"""Geo-reference extracted static elements into world space and feed them to the existing HD map pipeline.
Each element's ego-frame geometry is lifted to WGS84 with the cloud's GNSS position and heading (reusing the
Phase 3 georef.vehicle_to_world), written as a static_element, and mirrored into a MapElement so LiDAR
enriches the same HD map the camera-only pipeline builds. Provenance: the source cloud, the method, and the
calibration. Raw is never mutated.
"""

from __future__ import annotations

import uuid

from geoalchemy2.elements import WKTElement
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import MapElement, StaticElement
from services.hdmap.georef import vehicle_to_world

log = get_logger("lidar_to_hdmap")


def _ego_to_world(x: float, y: float, lat: float, lon: float, heading: float) -> tuple[float, float]:
    # ego y is +left; vehicle_to_world wants lateral +right. Returns (lon, lat) for WKT order.
    wlat, wlon = vehicle_to_world(x, -y, lat, lon, heading)
    return (wlon, wlat)


def _geometry_wkt(el: dict, lat: float, lon: float, heading: float) -> str | None:
    if "line" in el and len(el["line"]) >= 2:
        pts = [_ego_to_world(p[0], p[1], lat, lon, heading) for p in el["line"]]
        return "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in pts) + ")"
    if "footprint" in el and len(el["footprint"]) >= 2:
        pts = [_ego_to_world(p[0], p[1], lat, lon, heading) for p in el["footprint"]]
        return "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in pts) + ")"
    if "position" in el:
        wx, wy = _ego_to_world(el["position"][0], el["position"][1], lat, lon, heading)
        return f"POINT({wx} {wy})"
    return None


async def store_static_elements(db: AsyncSession, session_id: uuid.UUID, cloud_id: uuid.UUID,
                                elements: list[dict], lat: float | None, lon: float | None,
                                heading: float | None, calibration_version: str,
                                source_frames: list[str] | None = None) -> dict:
    """Write static_element rows and, when geo-referenced, mirror each into a MapElement HD map candidate. The
    MapElement carries source_frames so a LiDAR element has the same frame-level provenance as a 2D element."""
    n_static, n_map = 0, 0
    for el in elements:
        wkt = _geometry_wkt(el, lat, lon, heading) if lat is not None and lon is not None else None
        geom = WKTElement(wkt, srid=4326) if wkt else None
        conf = float(el.get("confidence", 0.6))
        map_id = None
        if geom is not None:
            me = MapElement(kind=el["kind"][:16], geometry=geom, attrs=dict(el),
                            source_frames=source_frames or None, source_sessions=[str(session_id)],
                            calibration_version=calibration_version, confidence=conf)
            db.add(me)
            await db.flush()
            map_id = me.element_id
            n_map += 1
        db.add(StaticElement(session_id=session_id, kind=el["kind"], geometry=geom, attrs=dict(el),
                             source_clouds=[cloud_id], method=el.get("method"), confidence=conf,
                             calibration_version=calibration_version, map_element_id=map_id))
        n_static += 1
    log.info("lidar.to_hdmap", session=str(session_id), static=n_static, map_elements=n_map)
    return {"static_elements": n_static, "map_elements": n_map}
