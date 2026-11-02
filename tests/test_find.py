"""Finding a session by its name, and a frame, object or track by the first characters of its id."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns, seconds_to_ns
from db.models import Frame, Object, OntologyClass, Track
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.api.routers.find import _id_prefix, _uuid_range, find

pytestmark = pytest.mark.db


async def _seed():
    sid, fid, oid, tid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    vehicle = f"FIND-{uuid.uuid4().hex[:6].upper()}"
    start = now_ns()
    async with get_sessionmaker()() as db:
        cid = (await db.execute(select(OntologyClass.id).limit(1))).scalar_one()
        db.add(DbSession(session_id=sid, vehicle_id=vehicle, start_ts_ns=start, end_ts_ns=start + seconds_to_ns(1),
                         city="Kochi", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri="s3://labeloxav/x.jpg",
                     width=640, height=480, quality=0.9))
        db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=start, last_ts_ns=start))
        await db.flush()
        db.add(Object(object_id=oid, frame_id=fid, track_id=tid, class_id=cid, bbox=[1, 1, 9, 9], conf=0.5))
        await db.commit()
    return vehicle, sid, fid, oid, tid


async def _find(q):
    async with get_sessionmaker()() as db:
        return (await find(q, db))["results"]


async def test_a_session_is_found_by_its_vehicle_and_opens_its_first_frame():
    vehicle, sid, fid, *_ = await _seed()
    hits = [r for r in await _find(vehicle.lower()) if r["kind"] == "session"]
    assert hits and hits[0]["id"] == str(sid)
    assert hits[0]["href"] == f"/frame/{fid}"


async def test_each_kind_is_found_by_its_id_prefix_and_opens_the_right_page():
    _, sid, fid, oid, tid = await _seed()
    assert any(r["kind"] == "session" and r["id"] == str(sid) for r in await _find(str(sid)[:8]))
    assert any(r["kind"] == "frame" and r["href"] == f"/frame/{fid}" for r in await _find(str(fid)[:8]))
    obj = [r for r in await _find(str(oid)[:10]) if r["kind"] == "object"]
    assert obj and obj[0]["href"] == f"/frame/{fid}?focus={oid}"
    trk = [r for r in await _find(str(tid)[:8]) if r["kind"] == "track"]
    assert trk and trk[0]["href"] == f"/track/{tid}" and trk[0]["detail"] == "1 objects"


async def test_a_word_never_triggers_an_id_scan():
    assert _id_prefix("pedestrian") is None
    assert _id_prefix("abc12") is None          # too short to narrow anything
    assert _id_prefix("5901b4fa") == "5901b4fa"


def test_a_prefix_becomes_the_id_range_it_covers():
    lo, hi = _uuid_range("5901b4fa-69")
    assert str(lo) == "5901b4fa-6900-0000-0000-000000000000"
    assert str(hi) == "5901b4fa-69ff-ffff-ffff-ffffffffffff"


def test_it_needs_a_signed_in_person():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    with TestClient(app) as c:
        assert c.get("/api/find?q=kochi").status_code == 401
        assert c.get("/api/find?q=kochi", headers=auth_headers("annotator")).status_code == 200
        assert c.get("/api/find?q=k", headers=auth_headers("annotator")).status_code == 422
