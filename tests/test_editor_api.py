"""Editor endpoint tests: create object, save mask, delete, frame meta + nav. Sync TestClient + the
run_async engine-cache-clear pattern (same as test_m6_api). Requires infra (DB + MinIO)."""

from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _seed_two_frames():
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid, f1, f2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    start = now_ns()
    img = np.random.default_rng(7).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{start}.jpg", buf.tobytes(), "image/jpeg")
    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="EDIT-01", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=f1, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri=uri, width=640, height=480, quality=0.9))
        db.add(Frame(frame_id=f2, session_id=sid, ts_ns=start + seconds_to_ns(1), cam_id="cam_f", img_uri=uri, width=640, height=480, quality=0.9))
        await db.commit()
    return sid, f1, f2


def _client():
    from fastapi.testclient import TestClient

    from services.api.main import app

    _clear_db_cache()
    return TestClient(app)


@requires_infra
def test_create_mask_and_delete_object():
    sid, f1, f2 = run_async(_seed_two_frames())
    poly = [[10, 10, 60, 10, 60, 50, 10, 50]]
    with _client() as c:
        # create a human-drawn object
        created = c.post(f"/api/frames/{f1}/objects", json={
            "class_name": "pedestrian", "bbox": [10, 10, 60, 50], "attrs": {}}).json()
        oid = created["object_id"]
        assert created["class_name"] == "pedestrian" and created["source"] == "human" and created["state"] == "accepted"

        # it shows up in the frame's object list
        objs = c.get(f"/api/frames/{f1}/objects").json()
        assert any(o["object_id"] == oid for o in objs)

        # save a mask, then read it back via object detail
        r = c.put(f"/api/objects/{oid}/mask", json={"polygons": poly})
        assert r.status_code == 200
        detail = c.get(f"/api/objects/{oid}").json()
        assert detail["mask_polygons"] == poly

        # bad class is rejected
        assert c.post(f"/api/frames/{f1}/objects", json={"class_name": "not_a_class", "bbox": [0, 0, 1, 1]}).status_code == 400

        # delete removes it
        assert c.delete(f"/api/objects/{oid}").status_code == 200
        assert c.get(f"/api/objects/{oid}").status_code == 404
        objs2 = c.get(f"/api/frames/{f1}/objects").json()
        assert not any(o["object_id"] == oid for o in objs2)


@requires_infra
def test_track_view_and_relabel():
    # seed a track whose two objects disagree on class, then relabel the whole track in one call
    sid, f1, f2 = run_async(_seed_two_frames())
    tid = uuid.uuid4()

    async def _seed_track():
        from db.models import Object, Track
        from db.session import get_sessionmaker
        async with get_sessionmaker()() as db:
            db.add(Track(track_id=tid, session_id=sid, class_id=6, first_ts_ns=1, last_ts_ns=2, trajectory=None))
            await db.flush()  # ensure the track row exists before the objects that FK to it
            db.add(Object(frame_id=f1, track_id=tid, class_id=6, bbox=[10, 10, 40, 40], conf=0.5,
                          attrs={}, source="fused", state="review", provenance={}))
            db.add(Object(frame_id=f2, track_id=tid, class_id=7, bbox=[12, 12, 42, 42], conf=0.5,
                          attrs={}, source="fused", state="review", provenance={}))
            await db.commit()

    run_async(_seed_track())
    with _client() as c:
        t = c.get(f"/api/tracks/{tid}").json()
        assert t["n_frames"] == 2 and t["flips"] is True
        r = c.post(f"/api/tracks/{tid}/relabel", json={"class_name": "pedestrian"})
        assert r.json()["relabeled"] == 2
        t2 = c.get(f"/api/tracks/{tid}").json()
        assert t2["flips"] is False and t2["dominant"] == "pedestrian"
        assert all(it["state"] == "accepted" for it in t2["items"])  # confirmed as human gold


@requires_infra
def test_bulk_review_and_jobs():
    sid, f1, f2 = run_async(_seed_two_frames())
    with _client() as c:
        o1 = c.post(f"/api/frames/{f1}/objects", json={"class_name": "sedan", "bbox": [1, 1, 5, 5], "state": "review"}).json()["object_id"]
        o2 = c.post(f"/api/frames/{f1}/objects", json={"class_name": "sedan", "bbox": [6, 6, 9, 9], "state": "review"}).json()["object_id"]
        r = c.post("/api/objects/bulk-review", json={"object_ids": [o1, o2], "action": "reject"})
        assert r.json()["updated"] == 2
        for oid in (o1, o2):
            assert c.get(f"/api/objects/{oid}").json()["state"] == "rejected"
        # unified jobs endpoint returns a list (kinds present even if empty)
        assert isinstance(c.get("/api/jobs").json(), list)


@requires_infra
def test_users_and_attribution():
    sid, f1, f2 = run_async(_seed_two_frames())
    uname = f"u-{uuid.uuid4().hex[:8]}"
    with _client() as c:
        u = c.post("/api/users", json={"name": uname, "role": "reviewer"}).json()
        assert u["role"] == "reviewer"
        assert c.post("/api/users", json={"name": uname}).status_code == 409  # duplicate name
        oid = c.post(f"/api/frames/{f1}/objects", json={"class_name": "sedan", "bbox": [1, 1, 5, 5]}).json()["object_id"]
        # review AS this user via the header -> attributed to them
        r = c.post(f"/api/objects/{oid}/review", json={"action": "reject"}, headers={"X-Lbx-User-Id": u["user_id"]})
        assert r.status_code == 200
        users = {x["name"]: x for x in c.get("/api/users").json()}
        assert users[uname]["reviews"] >= 1


@requires_infra
def test_intelligence_endpoints():
    """Data Intelligence Layer endpoints return well-formed shapes."""
    with _client() as c:
        sem = c.get("/api/search/semantic?q=night street&k=4").json()
        assert {"query", "filters", "classes", "results"} <= set(sem)
        assert c.get("/api/discovery/queue?state=pending&limit=3").status_code == 200
        ss = c.get("/api/analytics/scene-splits").json()
        assert {"weather", "time_of_day", "road_type", "density"} <= set(ss)
        dr = c.get("/api/analytics/dedup-rate").json()
        assert {"total", "redundant", "rate"} <= set(dr)
        rep = c.get("/api/analytics/report").json()
        assert "scene_splits" in rep and "dedup_rate" in rep


@requires_infra
def test_dedup_endpoint_shape():
    """Dedup returns a well-formed report; an empty/unknown session yields zeros, not a 500."""
    with _client() as c:
        r = c.post(f"/api/curation/dedup?session_id={uuid.uuid4()}")
        assert r.status_code == 200
        d = r.json()
        assert {"frames", "dup_groups", "redundant", "duplicate_rate"} <= set(d) and d["frames"] == 0


@requires_infra
def test_corrections_endpoints():
    """Interactive AI correction: coverage + confusions shape, and suggest's no-embedding path."""
    with _client() as c:
        cov = c.get("/api/corrections/coverage").json()
        assert {"embedded", "total", "pct"} <= set(cov)
        conf = c.get("/api/corrections/confusions?by=class").json()
        assert "confusions" in conf and "total_corrections" in conf
        # suggest on a non-embedded (random) object returns a valid empty shape, not a 500
        r = c.post("/api/corrections/suggest", json={
            "object_id": str(uuid.uuid4()), "kind": "class",
            "old_class_name": "truck", "new_class_name": "bus"})
        assert r.status_code == 200 and "candidates" in r.json()


@requires_infra
def test_autolabel_cloud_routing():
    """compute_target='cloud' parks the job for the A100 and never runs it locally (no GPU touched)."""
    with _client() as c:
        sessions = c.get("/api/sessions").json()
        if not sessions:
            return
        sid = sessions[0]["session_id"]
        r = c.post("/api/autolabel/start", json={"session_id": sid, "compute_target": "cloud"})
        assert r.status_code == 200 and r.json()["status"] == "queued-cloud"
        j = c.get(f"/api/autolabel/{r.json()['job_id']}").json()
        assert j["counts"]["compute_target"] == "cloud" and j["status"] == "pending"


@requires_infra
def test_datasets_list_and_detail():
    with _client() as c:
        ds = c.get("/api/datasets").json()
        assert isinstance(ds, list)
        if ds:
            d = c.get(f"/api/datasets/{ds[0]['commit_id']}").json()
            assert d["commit_id"] == ds[0]["commit_id"] and "files" in d
        assert c.get("/api/datasets/does-not-exist").status_code == 404


@requires_infra
def test_frame_meta_and_nav():
    sid, f1, f2 = run_async(_seed_two_frames())
    with _client() as c:
        m1 = c.get(f"/api/frames/{f1}").json()
        assert m1["width"] == 640 and m1["height"] == 480
        assert m1["next_frame_id"] == str(f2) and m1["prev_frame_id"] is None
        m2 = c.get(f"/api/frames/{f2}").json()
        assert m2["prev_frame_id"] == str(f1) and m2["next_frame_id"] is None
