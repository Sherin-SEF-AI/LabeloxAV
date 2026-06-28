"""Orchestrate static scene element extraction for a cloud: run every extractor on the cloud, its Phase 2
segmentation, and the ground plane, geo-reference the results with the cloud's GNSS pose, and feed them to the
HD map. Writes static_element rows; mirrors geo-referenced elements into MapElement candidates.
"""

from __future__ import annotations

import uuid
from collections import Counter

from geoalchemy2 import Geometry
from sqlalchemy import cast, func, select

from core.logging import get_logger
from db.models import Frame, PointCloud
from db.session import get_sessionmaker
from services.hdmap.georef import bearing
from services.lidar.extract.buildings import extract_buildings
from services.lidar.extract.common import load_for_extraction
from services.lidar.extract.markings import extract_markings
from services.lidar.extract.poles import extract_poles
from services.lidar.extract.roadedge import extract_road_edges
from services.lidar.extract.to_hdmap import store_static_elements
from services.lidar.extract.vegetation import extract_vegetation
from services.lidar.segment3d.semantic import road_class_id

log = get_logger("lidar_extract_run")
CALIB_VERSION = "labelox-calib-0.1"


async def _cloud_pose(session_id: uuid.UUID, cloud_id: uuid.UUID) -> tuple[float | None, float | None, float]:
    """The cloud's GNSS position and heading, from the synchronized frames' GNSS track."""
    async with get_sessionmaker()() as db:
        pc = await db.get(PointCloud, cloud_id)
        ts = pc.ts_ns if pc else None
        geom = cast(Frame.gnss, Geometry)
        rows = (await db.execute(select(Frame.ts_ns, func.ST_Y(geom), func.ST_X(geom))
                .where(Frame.session_id == session_id, Frame.gnss.isnot(None)).order_by(Frame.ts_ns))).all()
    pts = [(int(t), float(la), float(lo)) for t, la, lo in rows]
    if not pts or ts is None:
        return None, None, 0.0
    i = min(range(len(pts)), key=lambda k: abs(pts[k][0] - ts))
    lat, lon = pts[i][1], pts[i][2]
    if i + 1 < len(pts):
        h = bearing(lat, lon, pts[i + 1][1], pts[i + 1][2])
    elif i > 0:
        h = bearing(pts[i - 1][1], pts[i - 1][2], lat, lon)
    else:
        h = 0.0
    return lat, lon, h


async def extract_cloud(cloud_id: uuid.UUID) -> dict:
    """Extract poles, road edges, buildings, vegetation, and markings from a cloud and feed the HD map."""
    data = await load_for_extraction(cloud_id)
    if data is None:
        return {"error": "cloud not found"}
    cloud, plane, semantic, session_id = data["cloud"], data["plane"], data["semantic"], data["session_id"]
    road_id = road_class_id()

    elements: list[dict] = []
    elements += extract_poles(cloud, plane)
    elements += extract_road_edges(cloud, semantic, road_id, plane)
    elements += extract_buildings(cloud, plane)
    elements += extract_vegetation(cloud, semantic, plane)
    elements += extract_markings(cloud, semantic, road_id, plane)

    lat, lon, heading = await _cloud_pose(session_id, cloud_id)
    async with get_sessionmaker()() as db:
        res = await store_static_elements(db, session_id, cloud_id, elements, lat, lon, heading, CALIB_VERSION)
        await db.commit()

    by_kind = dict(Counter(e["kind"] for e in elements))
    log.info("lidar.extract_cloud", cloud=str(cloud_id), elements=len(elements), by_kind=by_kind,
             geo=lat is not None)
    return {"cloud_id": str(cloud_id), "session_id": str(session_id), "elements": len(elements),
            "by_kind": by_kind, "geo_referenced": lat is not None, **res}
