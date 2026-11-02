"""Dashcam telemetry stamps read back into position and speed.

The stamp is OCR'd, so the parser must survive the confusions this font produces (taken from real reads on
the corpus), the fix updates every few seconds and is held in between, and a misread digit must not become a
teleport. Sensor GNSS, where a session has it, is never overwritten.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest

from core.timebase import now_ns, seconds_to_ns
from services.ingest import overlay_telemetry as ot


def test_the_parser_reads_real_stamps_including_the_ocr_confusions():
    t = ot.parse_stamp("2026/05/3115:23:08 076KM/HN:13.0583 E:77.4684 F7N T0UCH")
    assert t == ot.Telemetry(datetime(2026, 5, 31, 15, 23, 8), 76.0, 13.0583, 77.4684)
    # '3L' for 31 and 'T5' for 15, as read on DASHCAM-01 frames.
    t = ot.parse_stamp("2026/05/3L T5:23:37 094KM/HN:13.0570 E:77.4733F7N TOUCH")
    assert t is not None and t.when == datetime(2026, 5, 31, 15, 23, 37) and t.speed_kmh == 94.0


def test_hemispheres_and_nonsense():
    t = ot.parse_stamp("2026/01/02 03:04:05 050KM/H S:33.8688 W:151.2093")
    assert t is not None and t.lat == -33.8688 and t.lon == -151.2093
    assert ot.parse_stamp("2026/01/02 03:04:05 050KM/H N:95.0000 E:77.0000") is None   # not on Earth
    assert ot.parse_stamp("2026/01/02 03:04:05 900KM/H N:13.0000 E:77.0000") is None   # not a car
    assert ot.parse_stamp("") is None and ot.parse_stamp("SHOP SIGN 2026") is None


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


@pytest.mark.db
def test_held_fixes_are_interpolated_a_jump_is_dropped_and_sensor_gnss_is_kept(monkeypatch):
    from geoalchemy2 import Geometry
    from geoalchemy2.elements import WKTElement
    from sqlalchemy import cast, func, select

    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    # 3 frames a second over 4 s; the fix changes at 0 s and 3 s (about 50 m apart) and is held between.
    # Frame 6 is misread 5 km away. Frame 11 carries a sensor fix that must survive.
    base = datetime(2026, 5, 31, 15, 23, 0)
    stamps = {}
    for i in range(12):
        held = (13.0583, 77.4684) if i < 9 else (13.0583, 77.46886)
        if i == 6:
            held = (13.1000, 77.5000)
        stamps[f"s3://x/ot/{i}.jpg"] = ot.Telemetry(base + timedelta(seconds=i // 3), 60.0, *held)
    monkeypatch.setattr(ot, "available", lambda: True)
    # The "image" is the stamp itself; reading it back is the identity.
    monkeypatch.setattr(ot, "load_image", lambda store, uri: stamps[uri])
    monkeypatch.setattr(ot, "read_overlay", lambda img: img)

    class _Store:
        pass

    async def _go():
        onto = get_ontology()
        sid, ts = uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="OT", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(5), city="BLR",
                             sensors={}, ontology_version=onto.version))
            await db.flush()
            for i in range(12):
                db.add(Frame(frame_id=uuid.uuid4(), session_id=sid, ts_ns=ts + seconds_to_ns(i) // 3, cam_id="cam_f",
                             img_uri=f"s3://x/ot/{i}.jpg", width=1920, height=1080, quality=0.9, scene={},
                             gnss=WKTElement("POINT(1.0 2.0)", srid=4326) if i == 11 else None))
            await db.commit()
            res = await ot.backfill_session(db, _Store(), sid)
            geom = cast(Frame.gnss, Geometry)
            rows = (await db.execute(select(Frame.ts_ns, func.ST_Y(geom), func.ST_X(geom), Frame.ego_speed)
                                     .where(Frame.session_id == sid).order_by(Frame.ts_ns))).all()
        return res, rows

    res, rows = _run(_go())
    assert res["rejected_jumps"] == 1
    lons = [r[2] for r in rows]
    # Between the two fixes the longitude moves steadily instead of holding and then jumping.
    assert lons[0] == pytest.approx(77.4684) and lons[9] == pytest.approx(77.46886)
    assert lons[0] < lons[3] < lons[6] < lons[9]
    assert rows[11][1] == pytest.approx(2.0) and rows[11][2] == pytest.approx(1.0), "sensor GNSS was overwritten"
    assert rows[0][3] == pytest.approx(60.0 / 3.6)
