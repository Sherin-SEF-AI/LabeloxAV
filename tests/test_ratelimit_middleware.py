"""The limiter in the request path, not just the arithmetic.

`tests/test_ratelimit.py` covers the bucket, the classification and the eviction as pure functions. What
those cannot show is whether the thing is actually wired: a limiter that is correct and unreachable is the
same as no limiter, and this codebase has already found several capabilities that were built and never
called.

The suite disables rate limiting globally (a limiter that throttles the tests hides real failures behind
429s), so these turn it back on for the duration, which is the same shape as
`tests/test_webhook_hardening.py` re-enabling the SSRF guard it needs.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config import get_settings
from services.api.ratelimit import Budget, MemoryLimiter
from services.api.ratelimit_middleware import RateLimitMiddleware


@pytest.fixture
def limiting():
    """Rate limiting on for this test, restored afterwards."""
    s = get_settings()
    before = s.ratelimit.enabled
    s.ratelimit.enabled = True
    yield
    s.ratelimit.enabled = before


def _app(limiter: MemoryLimiter) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(RateLimitMiddleware(limiter=limiter))

    @app.get("/api/frames/{fid}/image")
    async def image(fid: str):
        return {"fid": fid}

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.post("/api/upload/presign-put")
    async def presign():
        return {"url": "s3://signed"}

    return app


@pytest.fixture
def one_request_budget(monkeypatch):
    """Shrink the media budget to a single request for the duration of a test.

    The first version of these tests pre-spent a real bucket at `now=0.0` and relied on the refill between
    then and `time.monotonic()` being capped at the burst. That silently depends on the machine's uptime:
    large on a workstation that has been on for days, near zero on a fresh CI runner, where the bucket had
    not refilled and even the FIRST request was refused. Shrinking the budget instead is deterministic
    everywhere.
    """
    from services.api import ratelimit_middleware as mw

    tiny = Budget(rate_per_s=0.001, burst=1.0)
    monkeypatch.setattr(mw, "classify", lambda path, method: ("media", tiny, 1.0))
    return tiny


class TestItIsActuallyWired:
    def test_a_caller_over_budget_gets_429(self, limiting, one_request_budget):
        client = TestClient(_app(MemoryLimiter()))

        first = client.get("/api/frames/abc/image")
        second = client.get("/api/frames/abc/image")
        assert first.status_code == 200
        assert second.status_code == 429, "the limiter is not in the request path"

    def test_a_refusal_says_when_to_come_back(self, limiting, one_request_budget):
        """A 429 without Retry-After is an invitation to retry immediately, which turns a limit into a
        busy loop."""
        client = TestClient(_app(MemoryLimiter()))
        client.get("/api/frames/abc/image")
        r = client.get("/api/frames/abc/image")

        assert r.status_code == 429
        assert int(r.headers["Retry-After"]) >= 1
        assert r.headers["X-RateLimit-Class"] == "media"
        assert "retry in" in r.json()["detail"]

    def test_an_allowed_request_says_which_budget_it_drew_on(self, limiting):
        client = TestClient(_app(MemoryLimiter()))
        r = client.get("/api/frames/abc/image")
        assert r.status_code == 200
        assert r.headers["X-RateLimit-Class"] == "media"
        # Which key was used changes the meaning of the limit entirely behind a proxy, so it is stated
        # rather than left for an operator to infer.
        assert r.headers["X-RateLimit-Keyed-By"] in ("principal", "address")


class TestWhatItMustNotThrottle:
    def test_health_is_never_limited(self, limiting):
        """A health check that gets a 429 takes an instance out of a load balancer."""
        lim = MemoryLimiter()
        client = TestClient(_app(lim))
        for _ in range(50):
            assert client.get("/api/health").status_code == 200

    def test_turning_it_off_turns_it_off(self):
        """The suite runs with it off, so this is the configuration everything else depends on."""
        s = get_settings()
        before = s.ratelimit.enabled
        s.ratelimit.enabled = False
        try:
            client = TestClient(_app(MemoryLimiter()))
            for _ in range(10):
                assert client.get("/api/frames/abc/image").status_code == 200
        finally:
            s.ratelimit.enabled = before


class TestTheBudgetsThemselves:
    def test_presign_is_tighter_than_ordinary_media(self, limiting):
        """Each presign hands out a credential that outlives the request, so it is the one path where a
        burst is worth refusing."""
        from services.api.ratelimit import MEDIA, PRESIGN

        assert PRESIGN.rate_per_s < MEDIA.rate_per_s
        assert PRESIGN.burst < MEDIA.burst

        lim = MemoryLimiter()
        client = TestClient(_app(lim))
        codes = [client.post("/api/upload/presign-put").status_code for _ in range(int(PRESIGN.burst) + 5)]
        assert 429 in codes, "the credential-minting path had no effective ceiling"


class TestItRunsAfterAuth:
    """The bug this class exists for took the whole application down on one page load.

    Middleware added last is outermost, so registering the limiter last put it *before* AuthMiddleware.
    `request.state.principal_id` did not exist yet, every request fell back to keying on the client address,
    and behind the Next proxy that is one address for every user of the app. One person opening a frame
    exhausted the budget for everybody, and the editor issues about seventy requests to open a single frame.
    """

    def test_the_limiter_is_registered_before_auth_so_it_sits_inside_it(self):
        from services.api.main import app
        from services.api.main import AuthMiddleware

        names = [m.cls.__name__ for m in app.user_middleware]
        # user_middleware is outermost-first, so the limiter must appear AFTER auth in this list.
        auth_at = next(i for i, n in enumerate(names) if n == AuthMiddleware.__name__)
        limiter_at = next(
            i for i, m in enumerate(app.user_middleware)
            if "RateLimit" in repr(getattr(m, "kwargs", {}).get("dispatch", "")) or "RateLimit" in repr(m))
        assert limiter_at > auth_at, (
            "the limiter runs before auth, so it cannot see who the caller is and keys every user of the "
            "app onto one proxy address")

    def test_an_authenticated_request_is_keyed_by_the_person_not_the_address(self, limiting):
        """Behind a proxy, keying by address is keying every user together."""
        from starlette.middleware.base import BaseHTTPMiddleware

        app = FastAPI()

        class FakeAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.principal_id = "user-123"
                return await call_next(request)

        # Added first, so it is innermost and runs after FakeAuth, which is the real ordering.
        app.middleware("http")(RateLimitMiddleware(limiter=MemoryLimiter()))
        app.add_middleware(FakeAuth)

        @app.get("/api/frames/{fid}/image")
        async def image(fid: str):
            return {"fid": fid}

        r = TestClient(app).get("/api/frames/abc/image")
        assert r.headers["X-RateLimit-Keyed-By"] == "principal"

    def test_the_media_budget_covers_opening_a_frame(self, limiting):
        """Roughly seventy requests before anybody has drawn anything: frame, objects, lanes, drivable,
        relationships, adverse, cuboids, segmentation, dynamics, ontology, users, eleven thumbnails and a
        crop per object."""
        from services.api.ratelimit import MEDIA

        assert MEDIA.burst >= 70, "one page load would exhaust the media budget"
