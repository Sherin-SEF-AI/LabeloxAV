"""API auth gate (R1.1): deny-by-default for mutating routes, role floors, reads stay open.
Requires infra (DB). The rest of the suite runs with auth disabled (see conftest); this file turns it
on explicitly to exercise the middleware."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings


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


@pytest.fixture
def auth_on():
    s = get_settings()
    prev = s.auth.enabled
    s.auth.enabled = True
    yield
    s.auth.enabled = prev


async def _seed_users_coro():
    from db.models import User
    from db.session import get_sessionmaker

    admin = User(name=f"admin-{uuid.uuid4().hex[:8]}", role="admin")
    rev = User(name=f"rev-{uuid.uuid4().hex[:8]}", role="reviewer")
    ann = User(name=f"ann-{uuid.uuid4().hex[:8]}", role="annotator")
    async with get_sessionmaker()() as db:
        db.add_all([admin, rev, ann])
        await db.commit()
        return str(admin.user_id), str(rev.user_id), str(ann.user_id)


def _client():
    from fastapi.testclient import TestClient

    from services.api.main import app

    _clear_db_cache()
    return TestClient(app)


@requires_infra
def test_auth_gate(auth_on):
    admin_id, rev_id, ann_id = run_async(_seed_users_coro())
    with _client() as c:
        # reads stay open even with auth on
        assert c.get("/api/ontology").status_code == 200

        # mutating route with no identity -> 401
        assert c.post("/api/govern/controller/tick").status_code == 401

        # annotator cannot reach an admin route -> 403
        r = c.post("/api/govern/controller/tick", headers={"X-Lbx-User-Id": ann_id})
        assert r.status_code == 403

        # admin clears the auth gate on the admin route (not 401/403; may 200/5xx on logic)
        r = c.post("/api/govern/controller/tick", headers={"X-Lbx-User-Id": admin_id})
        assert r.status_code not in (401, 403)

        # reviewer floor: annotator blocked on a reviewer route, reviewer passes the gate
        assert c.post("/api/export", json={"name": "x", "states": ["accepted"]},
                      headers={"X-Lbx-User-Id": ann_id}).status_code == 403
        assert c.post("/api/export", json={"name": "x", "states": ["accepted"]},
                      headers={"X-Lbx-User-Id": rev_id}).status_code not in (401, 403)

        # an unknown user id is treated as unauthenticated -> 401
        assert c.post("/api/govern/controller/tick",
                      headers={"X-Lbx-User-Id": str(uuid.uuid4())}).status_code == 401
