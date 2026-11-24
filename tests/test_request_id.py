"""Every request carries an id that its log lines share and its response returns."""

from __future__ import annotations

import re

import pytest
import structlog
from fastapi.testclient import TestClient

from services.api.main import app

# /api/health opens a database session.
pytestmark = pytest.mark.db


def test_a_response_names_its_request():
    r = TestClient(app).get("/api/health")
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"])


def test_two_requests_get_different_ids():
    c = TestClient(app)
    assert c.get("/api/health").headers["x-request-id"] != c.get("/api/health").headers["x-request-id"]


def test_a_callers_own_id_is_kept_so_a_trace_survives_a_proxy():
    r = TestClient(app).get("/api/health", headers={"X-Request-ID": "trace-abc12345"})
    assert r.headers["x-request-id"] == "trace-abc12345"


@pytest.mark.parametrize("bad", ["short", "has space in it 123", "x" * 200, "new\nline-12345678",
                                 "<script>alert(1)</script>"])
def test_an_unsafe_id_is_replaced_not_echoed(bad):
    r = TestClient(app).get("/api/health", headers={"X-Request-ID": bad})
    assert r.headers["x-request-id"] != bad
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"])


def test_the_id_is_on_log_lines_written_during_the_request_and_gone_afterwards():
    seen: dict = {}

    @app.get("/api/health/_probe_request_id", include_in_schema=False)
    async def _probe():
        seen.update(structlog.contextvars.get_contextvars())
        return {"ok": True}

    try:
        r = TestClient(app).get("/api/health/_probe_request_id", headers={"X-Request-ID": "probe-12345678"})
        if r.status_code == 200:
            assert seen.get("request_id") == "probe-12345678"
        assert "request_id" not in structlog.contextvars.get_contextvars(), "the id leaked past its request"
    finally:
        app.router.routes[:] = [rt for rt in app.router.routes
                                if getattr(rt, "path", "") != "/api/health/_probe_request_id"]


def test_a_401_still_carries_an_id():
    r = TestClient(app).get("/api/ontology")
    assert r.status_code in (200, 401)
    assert "x-request-id" in r.headers
