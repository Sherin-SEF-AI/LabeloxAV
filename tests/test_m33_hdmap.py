"""M3.3 HD map generation: IPM lifts an image-space lane to plausible world coordinates near the frame's
GNSS, and a multi-drive fusion seals a map_commit with Lanelet2 + OpenDRIVE exports. The RANSAC ground
plane is a pure unit. georef/fusion run on a constructed session (one frame, GNSS, a lane, calibration
pass). Single asyncio.run binds the cached engine to one loop."""

from __future__ import annotations

import asyncio
import math
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.hdmap.elevation import elevation_at, ransac_ground_plane
from services.hdmap.export import parse_wkt, seal_map_commit_id, to_lanelet2_osm, to_opendrive
from services.hdmap.georef import ipm_pixel_to_vehicle, vehicle_to_world


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def test_ransac_ground_plane_recovers_flat():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-5, 5, 200), rng.uniform(-5, 5, 200), np.full(200, 2.0)])
    pts[:5, 2] += 50.0  # outliers
    plane = ransac_ground_plane(pts)
    assert plane is not None and abs(elevation_at(plane, 1.0, 1.0) - 2.0) < 0.05


def test_ipm_forward_distance_and_world_offset():
    # a pixel below the principal point projects ahead on the road; nearer the horizon means farther
    near = ipm_pixel_to_vehicle(320, 460, fx=957, fy=957, cx=320, cy=240, height_m=1.5)
    far = ipm_pixel_to_vehicle(320, 300, fx=957, fy=957, cx=320, cy=240, height_m=1.5)
    assert near and far and far[0] > near[0] > 0  # farther forward as v approaches the horizon
    # a point straight ahead moves north when heading is north
    lat, lon = vehicle_to_world(10.0, 0.0, 12.97, 77.59, 0.0)
    assert lat > 12.97 and abs(lon - 77.59) < 1e-4


def test_exporters_emit_wellformed_xml():
    import xml.dom.minidom as minidom

    fused = [{"kind": "lane", "wkt": "LINESTRING(77.59 12.97, 77.591 12.971)",
              "attrs": {"lane_type": "solid"}, "confidence": 0.9, "frames": ["f"], "sessions": ["s"]}]
    minidom.parseString(to_lanelet2_osm(fused))   # raises if malformed
    minidom.parseString(to_opendrive(fused, (12.97, 77.59)))
    assert seal_map_commit_id(["s"], fused, "c").startswith("map-")
    assert parse_wkt("POINT(77.59 12.97)")[0] == "point"


@requires_infra
def test_georef_and_fuse_seal_a_commit():
    from db.models import CalibrationValidation, Frame, Lane, MapCommit, MapElement
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.hdmap.georef import georef_session
    from services.hdmap.run import start_map_fusion

    sid = uuid.uuid4()

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="MAP-TEST", start_ts_ns=0, end_ts_ns=2,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            # two frames a few metres apart so a heading can be derived
            f1 = Frame(session_id=sid, ts_ns=1000, cam_id="cam_f", img_uri="s3://x/1.jpg", width=640, height=480,
                       gnss="SRID=4326;POINT(77.5900 12.9700)")
            f2 = Frame(session_id=sid, ts_ns=2000, cam_id="cam_f", img_uri="s3://x/2.jpg", width=640, height=480,
                       gnss="SRID=4326;POINT(77.5901 12.9701)")
            db.add_all([f1, f2])
            await db.flush()
            db.add(Lane(frame_id=f1.frame_id, session_id=sid, control_points=[[100, 460], [220, 360], [320, 300]],
                        lane_type="solid", is_ego=True, source="human"))
            db.add(CalibrationValidation(session_id=sid, cam_id="cam_f", model="pinhole", status="pass"))
            await db.commit()

        g = await georef_session(sid)
        assert g["lanes"] >= 1

        async with maker() as db:
            els = (await db.execute(MapElement.__table__.select().where(
                MapElement.source_sessions.any(str(sid))))).all()
            assert els  # a lane element near Bangalore was written
        fr = await start_map_fusion([str(sid)], compute_target="local")
        assert fr.get("commit_id", "").startswith("map-")
        assert fr["formats"]["lanelet2"].endswith(".osm") and fr["formats"]["opendrive"].endswith(".xodr")

        async with maker() as db:
            commit = await db.get(MapCommit, fr["commit_id"])
            assert commit is not None and commit.element_count >= 1
            await db.delete(await db.get(DbSession, sid))  # cascades frames/lanes
            await db.execute(MapElement.__table__.delete().where(MapElement.source_sessions.any(str(sid))))
            await db.delete(commit)
            await db.commit()

    asyncio.run(run())
