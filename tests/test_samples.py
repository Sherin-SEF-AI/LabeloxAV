"""Sample drives: listed with their licence and credit, loaded once, and only ever the exact recorded bytes.

The download itself is exercised against Wikimedia Commons by hand (see the commit); these run offline.
"""

from __future__ import annotations

import hashlib
import uuid

import httpx
import pytest

from core.timebase import now_ns, seconds_to_ns
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.samples import SAMPLES, Sample, SampleError, fetch_to_store, public, sample_for_route

pytestmark = pytest.mark.db


def test_every_sample_carries_its_licence_and_credit():
    for s in SAMPLES:
        p = public(s)
        assert p["license"] == "CC BY-SA 3.0" and p["license_url"].startswith("https://creativecommons.org/")
        assert s.author in p["credit"] and "Wikimedia Commons" in p["credit"]
        assert p["route"] == f"sample:{s.id}" and sample_for_route(p["route"]) == s
        assert len(s.sha256) == 64 and s.size_bytes > 0


def test_a_teams_own_recording_is_not_a_sample():
    assert sample_for_route("import:video") is None and sample_for_route(None) is None


class _Store:
    bucket = "b"

    def __init__(self):
        self.put: dict[str, bytes] = {}

        class _C:
            def head_object(_, **kw):  # noqa: N805 - a stand-in for boto3's client
                raise FileNotFoundError

        self._client = _C()

    def uri(self, key):
        return f"s3://b/{key}"

    def put_bytes(self, key, data, content_type):
        self.put[key] = data
        return self.uri(key)


def _serve(monkeypatch, body: bytes, status: int = 200):
    real = httpx.AsyncClient

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(lambda req: httpx.Response(status, content=body))
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", client)


def _sample(body: bytes) -> Sample:
    return Sample(id="t", title="T", description="", url="https://example.invalid/t.webm",
                  sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body), license="CC BY-SA 3.0",
                  license_url="https://creativecommons.org/licenses/by-sa/3.0", author="A", source_page="",
                  vehicle_id="SAMPLE-T", city="X")


async def test_the_recorded_bytes_are_stored(monkeypatch):
    body = b"\x1a\x45\xdf\xa3 a small webm-shaped payload"
    _serve(monkeypatch, body)
    store = _Store()
    assert await fetch_to_store(_sample(body), store) == "s3://b/samples/t.webm"
    assert store.put["samples/t.webm"] == body


async def test_different_bytes_are_refused_and_not_stored(monkeypatch):
    _serve(monkeypatch, b"not the file whose licence was checked")
    store = _Store()
    with pytest.raises(SampleError, match="does not match"):
        await fetch_to_store(_sample(b"the real file"), store)
    assert store.put == {}


async def test_a_server_error_is_said_plainly(monkeypatch):
    _serve(monkeypatch, b"", status=503)
    with pytest.raises(SampleError, match="503"):
        await fetch_to_store(_sample(b"x"), _Store())


def test_listing_and_loading_through_the_api():
    import asyncio

    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from db.session import get_engine
    from services.api.main import app

    s = SAMPLES[0]
    sid = uuid.uuid4()

    async def seed():
        async with get_sessionmaker()() as db:
            from sqlalchemy import delete
            await db.execute(delete(DbSession).where(DbSession.route == f"sample:{s.id}"))
            start = now_ns()
            db.add(DbSession(session_id=sid, vehicle_id=s.vehicle_id, start_ts_ns=start,
                             end_ts_ns=start + seconds_to_ns(1), city=s.city, sensors={},
                             ontology_version="labelox-in-0.1.0", route=f"sample:{s.id}"))
            await db.commit()

    asyncio.run(seed())
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    with TestClient(app) as c:
        h = auth_headers("annotator")
        listed = {x["id"]: x for x in c.get("/api/samples", headers=h).json()["samples"]}
        assert listed[s.id]["session_id"] == str(sid) and listed[s.id]["credit"]
        # Already loaded: not fetched or imported a second time.
        r = c.post(f"/api/samples/{s.id}/load", headers=h).json()
        assert r == {"status": "already_loaded", "session_id": str(sid)}
        assert c.post("/api/samples/nope/load", headers=h).status_code == 404
        assert c.get("/api/samples").status_code == 401


def test_an_import_in_flight_is_reported_and_not_started_twice():
    """A page reloaded mid-import offered "Load sample" again, and a second click started a second download."""
    import asyncio

    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from db.models import ImportJob
    from db.session import get_engine
    from services.api.main import app

    s = SAMPLES[1]
    jid = uuid.uuid4()

    async def seed():
        async with get_sessionmaker()() as db:
            from sqlalchemy import delete
            await db.execute(delete(DbSession).where(DbSession.route == f"sample:{s.id}"))
            db.add(ImportJob(job_id=jid, status="running", format="video", source_uri=s.url,
                             target_vehicle=s.vehicle_id, city=s.city, progress=0.3, counts={"stage": "importing"}))
            await db.commit()

    async def jobs_for_sample() -> int:
        from sqlalchemy import func, select
        async with get_sessionmaker()() as db:
            return (await db.execute(select(func.count()).select_from(ImportJob)
                                     .where(ImportJob.source_uri == s.url))).scalar_one()

    asyncio.run(seed())
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    with TestClient(app) as c:
        h = auth_headers("annotator")
        listed = {x["id"]: x for x in c.get("/api/samples", headers=h).json()["samples"]}
        assert listed[s.id]["session_id"] is None and listed[s.id]["job_id"] == str(jid)
        assert c.post(f"/api/samples/{s.id}/load", headers=h).json() == {"status": "pending", "job_id": str(jid)}
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    assert asyncio.run(jobs_for_sample()) == 1
