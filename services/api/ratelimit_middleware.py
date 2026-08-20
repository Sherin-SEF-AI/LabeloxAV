"""The rate limiter, wired into the request path.

Separate from `ratelimit.py` so the policy is testable without a request: the bucket arithmetic, the route
classification and the eviction rule are pure functions with their own tests, and this file is the wiring
that decides who a caller is and what to do when they are over.

Two decisions worth stating.

**Identity comes from `request.state.principal_id`**, published by `AuthMiddleware`. Resolving it here would
mean a third independent identity resolution per request, and three implementations of "who is this" is how
they drift. Where there is no principal, the key is the client address, and the header says which was used,
because a limit keyed on an address behaves very differently behind a proxy and an operator should not have
to guess which one they are seeing.

**A refusal always says when to come back.** A 429 without `Retry-After` is an invitation to retry
immediately, which turns a limit into a busy loop.
"""

from __future__ import annotations

import inspect

from fastapi.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger
from services.api.ratelimit import (
    ANONYMOUS,
    MemoryLimiter,
    RedisLimiter,
    classify,
    retry_after_seconds,
)

log = get_logger("api.ratelimit")

# Paths that must never be throttled. A health check that gets a 429 takes an instance out of a load
# balancer, and a metrics scrape that does hides the very traffic that explains why.
EXEMPT_PREFIXES = ("/api/health", "/api/readyz", "/metrics", "/api/openapi.json", "/docs", "/openapi.json")


class RateLimitMiddleware:
    """A limiter keyed by caller and route class, shared across workers when Redis is reachable.

    It was memory-only despite the module docstring promising otherwise, which meant N workers enforced N
    times the advertised rate - and the header said nothing about it, so the limit looked like it held.
    The backend is now reported in X-RateLimit-Backend, so a deployment that has fallen back to
    per-process buckets is visible rather than assumed.
    """

    def __init__(self, limiter=None) -> None:
        # Built lazily on first use: the limiter is constructed at import time, before settings or the
        # event loop exist, and a Redis client made here would bind to the wrong loop under uvicorn.
        self._explicit = limiter
        self._limiter = limiter
        self._tried_redis = False

    def _get_limiter(self):
        if self._limiter is not None:
            return self._limiter
        if not self._tried_redis:
            self._tried_redis = True
            try:
                import redis.asyncio as aioredis

                from core.config import get_settings as _s
                cfg = _s().redis
                self._limiter = RedisLimiter(
                    aioredis.Redis(host=cfg.host, port=cfg.port, db=getattr(cfg, "db", 0),
                                   socket_timeout=0.25, socket_connect_timeout=0.25))
                log.info("api.ratelimit.backend", backend="redis")
            except Exception as exc:  # noqa: BLE001
                log.warning("api.ratelimit.redis_unavailable", error=str(exc), backend="memory")
                self._limiter = MemoryLimiter()
        return self._limiter or MemoryLimiter()

    async def __call__(self, request, call_next):
        settings = get_settings()
        if not settings.ratelimit.enabled:
            return await call_next(request)

        path = request.url.path
        if path.startswith(EXEMPT_PREFIXES):
            return await call_next(request)

        principal = getattr(request.state, "principal_id", None)
        if principal:
            key, keyed_by = f"user:{principal}", "principal"
        else:
            client = request.client.host if request.client else "unknown"
            key, keyed_by = f"addr:{client}", "address"

        klass, budget, cost = classify(path, request.method)
        # An anonymous caller gets the smaller of the route's budget and the anonymous one, so an
        # unauthenticated loop cannot spend an authenticated allowance.
        if not principal and budget.rate_per_s > ANONYMOUS.rate_per_s:
            budget = ANONYMOUS

        limiter = self._get_limiter()
        result = limiter.check(key, klass, budget, cost)
        allowed, wait = await result if inspect.isawaitable(result) else result
        backend = "memory" if isinstance(limiter, MemoryLimiter) else (
            "memory-degraded" if getattr(limiter, "degraded", False) else "redis")
        if not allowed:
            retry = retry_after_seconds(wait)
            log.warning("api.rate_limited", path=path, klass=klass, keyed_by=keyed_by, retry_after=retry)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry), "X-RateLimit-Class": klass,
                         "X-RateLimit-Keyed-By": keyed_by, "X-RateLimit-Backend": backend},
                content={"detail": f"rate limit exceeded for {klass} requests; retry in {retry}s",
                         "klass": klass, "retry_after": retry},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Class"] = klass
        response.headers["X-RateLimit-Keyed-By"] = keyed_by
        response.headers["X-RateLimit-Backend"] = backend
        return response
