"""API auth gate (R1.1, R2.2): deny-by-default for reads and writes, role floors, only a tiny allowlist of
public reads. Requires infra (DB). The rest of the suite runs with auth disabled (see conftest); this file
turns it on explicitly to exercise the middleware."""

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


def _bearer(user_id: str) -> dict:
    from core.config import get_settings
    from services.api.auth_token import mint_token

    return {"Authorization": f"Bearer {mint_token(user_id, get_settings().auth.signing_key)}"}


@requires_infra
def test_auth_gate(auth_on):
    admin_id, rev_id, ann_id = run_async(_seed_users_coro())
    with _client() as c:
        # reads now fail closed: an unauthenticated data read is 401, not open (R2.2 allowlist inversion)
        assert c.get("/api/ontology").status_code == 401
        # a signed-in user reads fine (the frontend rides the token on every GET)
        assert c.get("/api/ontology", headers=_bearer(rev_id)).status_code == 200
        # health stays public: it is the load-balancer liveness probe
        assert c.get("/api/health").status_code == 200

        # mutating route with no identity -> 401
        assert c.post("/api/govern/controller/tick").status_code == 401

        # a plaintext user id (the old, forgeable credential) no longer authenticates -> 401
        assert c.post("/api/govern/controller/tick",
                      headers={"X-Lbx-User-Id": admin_id}).status_code == 401

        # annotator (signed token) cannot reach an admin route -> 403
        r = c.post("/api/govern/controller/tick", headers=_bearer(ann_id))
        assert r.status_code == 403

        # admin clears the auth gate on the admin route (not 401/403; may 200/5xx on logic)
        r = c.post("/api/govern/controller/tick", headers=_bearer(admin_id))
        assert r.status_code not in (401, 403)

        # reviewer floor: annotator blocked on a reviewer route, reviewer passes the gate
        assert c.post("/api/export", json={"name": "x", "states": ["accepted"]},
                      headers=_bearer(ann_id)).status_code == 403
        assert c.post("/api/export", json={"name": "x", "states": ["accepted"]},
                      headers=_bearer(rev_id)).status_code not in (401, 403)

        # an unknown (but validly-signed) user id is treated as unauthenticated -> 401
        assert c.post("/api/govern/controller/tick",
                      headers=_bearer(str(uuid.uuid4()))).status_code == 401

        # a tampered/garbage token -> 401
        assert c.post("/api/govern/controller/tick",
                      headers={"Authorization": "Bearer lbx1.deadbeef.deadbeef"}).status_code == 401


@requires_infra
def test_dev_login_hands_out_an_admin_token_locally(auth_on):
    """dev-login is reachable without a token (it exists to hand out the first one), and the token it returns
    clears the admin gate. This is the bootstrap that keeps a fresh browser from being locked out."""
    s = get_settings()
    assert s.env == "local"  # the tests run local; the route's guard is env == "local"
    with _client() as c:
        r = c.post("/api/auth/dev-login")
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "admin" and body["token"].startswith("lbx2.")
        # the handed-out token actually authenticates against a gated read and an admin write
        assert c.get("/api/users", headers={"Authorization": f"Bearer {body['token']}"}).status_code == 200
        assert c.post("/api/govern/controller/tick",
                      headers={"Authorization": f"Bearer {body['token']}"}).status_code not in (401, 403)


@requires_infra
def test_dev_login_is_invisible_when_not_local(auth_on):
    """The env != local guard makes the route 404, so it cannot mint a token on a real deployment. Flipping
    the flag off has the same effect. Both are restored after."""
    s = get_settings()
    with _client() as c:
        prev_env = s.env
        s.env = "staging"
        try:
            assert c.post("/api/auth/dev-login").status_code == 404
        finally:
            s.env = prev_env

        prev_flag = s.auth.dev_login
        s.auth.dev_login = False
        try:
            assert c.post("/api/auth/dev-login").status_code == 404
        finally:
            s.auth.dev_login = prev_flag
