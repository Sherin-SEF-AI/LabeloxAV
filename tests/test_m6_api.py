"""M6 API tests: ontology, triage ranking, object detail, review persistence, image proxy, export.
Requires infra (DB + MinIO).

These are sync tests: TestClient drives the ASGI app in its own event loop, while DB seeding and
assertions use asyncio.run. The app caches one engine per process, so we clear that cache around
each loop boundary (run_async) to avoid binding an engine to a closed/foreign loop.
"""

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


async def _seed_coro():
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid, fid, oid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    start = now_ns()
    img = np.random.default_rng(2).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    img_uri = store.put_bytes(f"frames/{sid}/cam_f/{start}.jpg", buf.tobytes(), "image/jpeg")

    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-07", start_ts_ns=start, end_ts_ns=start + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri=img_uri,
                     width=640, height=480, quality=0.9))
        db.add(Object(object_id=oid, frame_id=fid, class_id=6, bbox=[100, 100, 200, 200], conf=0.41,
                      attrs={}, source="fused", state="annotate",
                      provenance={"proposals": [{"path": "path_b_sam3", "model_version": "world+sam", "verdict": "agree"}],
                                  "agreement": False, "mask_box_disagree": True}))
        await db.commit()
    return sid, fid, oid


def _seed():
    return run_async(_seed_coro())


def _client():
    from fastapi.testclient import TestClient

    from services.api.main import app

    _clear_db_cache()
    return TestClient(app)


@requires_infra
def test_ontology_and_sessions_endpoints():
    sid, _, _ = _seed()
    with _client() as c:
        onto = c.get("/api/ontology").json()
        assert onto["version"] == "labelox-in-0.1.0"
        assert len(onto["classes"]) >= 166  # additive ontology (P1 governed floor; customs may add more)
        sessions = c.get("/api/sessions").json()
        assert any(s["session_id"] == str(sid) for s in sessions)


@requires_infra
def test_triage_surfaces_rare_lowconf_with_reason():
    sid, _, oid = _seed()
    with _client() as c:
        rows = c.get(f"/api/triage?session_id={sid}").json()
    assert rows
    row = next(r for r in rows if r["object_id"] == str(oid))
    assert row["state"] == "annotate"
    assert "rare class" in row["why"] or "mask != box" in row["why"]
    assert rows == sorted(rows, key=lambda r: r["priority"], reverse=True)


@requires_infra
def test_session_stats_and_first_frame():
    sid, fid, oid = _seed()
    with _client() as c:
        stats = c.get(f"/api/sessions/{sid}/stats").json()
        assert stats["frames"] >= 1 and "by_state" in stats and 0.0 <= stats["progress"] <= 1.0
        ff = c.get(f"/api/sessions/{sid}/first-frame").json()
        assert ff["frame_id"] == str(fid)


@requires_infra
def test_object_detail_and_image_proxy():
    sid, fid, oid = _seed()
    with _client() as c:
        detail = c.get(f"/api/objects/{oid}").json()
        assert detail["class_name"] == "autorickshaw"
        assert detail["image_url"] == f"/api/frames/{fid}/image"
        img = c.get(detail["image_url"])
        assert img.status_code == 200
        assert img.headers["content-type"] == "image/jpeg"
        assert len(img.content) > 0


@requires_infra
def test_review_persists_correction_and_writes_review_row():
    from sqlalchemy import func, select

    sid, fid, oid = _seed()
    with _client() as c:
        resp = c.post(
            f"/api/objects/{oid}/review",
            json={"reviewer": "sherin", "action": "reclassify", "class_name": "e_auto",
                  "attrs": {"overload": True}, "state": "accepted", "time_spent_ms": 1500},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["class_name"] == "e_auto"
        assert body["state"] == "accepted"
        assert body["source"] == "human"

    async def _check():
        from db.models import Object, Review
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            obj = await db.get(Object, oid)
            n = (await db.execute(select(func.count()).select_from(Review).where(Review.object_id == oid))).scalar_one()
            return obj.state, obj.source, obj.attrs.get("overload"), n

    state, source, overload, n = run_async(_check())
    assert (state, source, overload, n) == ("accepted", "human", True, 1)


@requires_infra
def test_review_optimistic_lock_rejects_stale_write():
    """R3: a review carrying a stale expected_version is rejected (409); the up-to-date one succeeds."""
    sid, fid, oid = _seed()
    with _client() as c:
        v0 = c.get(f"/api/objects/{oid}").json()["version"]
        # first edit at the loaded version succeeds and advances the version
        r1 = c.post(f"/api/objects/{oid}/review",
                    json={"action": "confirm", "state": "accepted", "expected_version": v0})
        assert r1.status_code == 200 and r1.json()["version"] == v0 + 1
        # a second editor still on the old version is refused
        r2 = c.post(f"/api/objects/{oid}/review",
                    json={"action": "confirm", "state": "accepted", "expected_version": v0})
        assert r2.status_code == 409
        # without a version (legacy clients) the write still goes through
        assert c.post(f"/api/objects/{oid}/review", json={"action": "confirm", "state": "accepted"}).status_code == 200


@requires_infra
def test_review_rejects_invalid_class_and_attrs():
    _, _, oid = _seed()
    with _client() as c:
        assert c.post(f"/api/objects/{oid}/review", json={"action": "reclassify", "class_name": "not_a_class"}).status_code == 400
        assert c.post(f"/api/objects/{oid}/review", json={"action": "confirm", "attrs": {"occlusion": 33}}).status_code == 400


@requires_infra
def test_export_endpoint_seals_and_sanity_checks():
    sid, fid, oid = _seed()
    with _client() as c:
        c.post(f"/api/objects/{oid}/review", json={"action": "confirm", "state": "accepted"})
        resp = c.post("/api/export", json={"name": "api-demo", "states": ["accepted"], "session_id": str(sid),
                                           "formats": ["coco", "parquet"]})
        assert resp.status_code == 200, resp.text
        result = resp.json()
    assert result["object_count"] >= 1
    assert result["commit_id"].startswith("lbx-")
    assert result["reimport_sanity"]["ok"] is True
