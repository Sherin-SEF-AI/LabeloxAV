"""GNSS-to-road map matching (M3.2): for each frame's GNSS fix, find the nearest OSM road within a gate
and write road_segment_id / road_class / lane_count / speed_limit. A nearest-road matcher (the HMM
upgrade is a clean seam). India reality: where no road is near, the fields stay null (graceful)."""

from __future__ import annotations

from uuid import UUID

from geoalchemy2 import Geometry
from sqlalchemy import cast, func, select, update

from core.logging import get_logger
from db.models import Frame
from db.session import get_sessionmaker
from services.mapassist.osm import load_roads, nearest_road

log = get_logger("map_match")


async def match_session(session_id: UUID, max_dist_m: float = 30.0) -> dict:
    roads = load_roads()
    maker = get_sessionmaker()
    async with maker() as db:
        geom = cast(Frame.gnss, Geometry)
        rows = (await db.execute(
            select(Frame.frame_id, func.ST_Y(geom), func.ST_X(geom))
            .where(Frame.session_id == session_id, Frame.gnss.isnot(None)))).all()
        matched, no_road, classes = 0, 0, {}
        for fid, lat, lon in rows:
            r = nearest_road(roads, float(lon), float(lat), max_dist_m)
            if r is None:
                no_road += 1
                continue
            await db.execute(update(Frame).where(Frame.frame_id == fid).values(
                road_segment_id=r["id"], road_class=r["highway"], lane_count=r["lanes"], speed_limit=r["maxspeed"]))
            matched += 1
            classes[r["highway"]] = classes.get(r["highway"], 0) + 1
        await db.commit()

    out = {"session_id": str(session_id), "frames_with_gnss": len(rows), "matched": matched,
           "no_road": no_road, "road_classes": classes, "osm_roads": len(roads["data"])}
    log.info("map_match.done", **{k: out[k] for k in ("matched", "no_road", "osm_roads")})
    return out
