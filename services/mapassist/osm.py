"""OSM road-network ingest (M3.2): read the local Bangalore extract via osmium into shapely road
LineStrings (lon,lat) with their tags (highway class, lane count, speed limit), indexed by an STRtree for
fast nearest-road queries during map-matching."""

from __future__ import annotations

import functools
import math

import osmium
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from core.config import get_settings
from core.logging import get_logger

log = get_logger("osm")


def _parse_speed(v) -> int | None:
    if not v:
        return None
    digits = "".join(c for c in str(v) if c.isdigit())
    return int(digits) if digits else None


def _parse_lanes(v) -> int | None:
    if v and str(v).isdigit():
        return int(v)
    return None


# Vehicle-drivable highway classes (exclude footway, steps, path, cycleway, pedestrian).
DRIVABLE = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "service", "living_street", "road",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}


class _RoadHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.roads: list[dict] = []

    def way(self, w) -> None:
        if w.tags.get("highway") not in DRIVABLE:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except Exception:  # noqa: BLE001
            return
        if len(coords) < 2:
            return
        self.roads.append({
            "id": str(w.id), "highway": w.tags.get("highway"),
            "lanes": _parse_lanes(w.tags.get("lanes")), "maxspeed": _parse_speed(w.tags.get("maxspeed")),
            "name": w.tags.get("name"), "geom": LineString(coords),
        })


@functools.lru_cache(maxsize=1)
def load_roads() -> dict:
    """Load the OSM extract into {tree, geoms, data}. Empty + graceful if the extract is missing."""
    path = get_settings().spatial.osm_extract_path
    h = _RoadHandler()
    try:
        h.apply_file(path, locations=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("osm.load_failed", path=path, error=str(exc))
        return {"tree": None, "geoms": [], "data": []}
    geoms = [r["geom"] for r in h.roads]
    tree = STRtree(geoms) if geoms else None
    log.info("osm.loaded", roads=len(geoms), path=path)
    return {"tree": tree, "geoms": geoms, "data": h.roads}


def nearest_road(roads: dict, lon: float, lat: float, max_dist_m: float) -> dict | None:
    """Nearest road within max_dist_m of (lon, lat), or None (graceful where OSM is sparse)."""
    tree, geoms, data = roads["tree"], roads["geoms"], roads["data"]
    if tree is None or not geoms:
        return None
    pt = Point(lon, lat)
    idx = tree.query_nearest(pt)
    if idx is None or len(idx) == 0:
        return None
    i = int(idx[0])
    near = nearest_points(geoms[i], pt)[0]
    d_m = math.hypot((pt.y - near.y) * 111320.0, (pt.x - near.x) * 111320.0 * math.cos(math.radians(lat)))
    return data[i] if d_m <= max_dist_m else None
