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

import time

from fastapi.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger
from services.api.ratelimit import ANONYMOUS, MemoryLimiter, classify, retry_after_seconds

log = get_logger("api.ratelimit")

# Paths that must never be throttled. A health check that gets a 429 takes an instance out of a load
# balancer, and a metrics scrape that does hides the very traffic that explains why.
EXEMPT_PREFIXES = ("/api/health", "/api/readyz", "/metrics", "/api/openapi.json", "/docs", "/openapi.json")


class RateLimitMiddleware:
    """One shared limiter per process, keyed by caller and route class."""

    def __init__(self, limiter: MemoryLimiter | None = None) -> None:
        self.limiter = limiter or MemoryLimiter()

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

        allowed, wait = self.limiter.check(key, klass, budget, cost, now=time.monotonic())
        if not allowed:
            retry = retry_after_seconds(wait)
            log.warning("api.rate_limited", path=path, klass=klass, keyed_by=keyed_by, retry_after=retry)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry), "X-RateLimit-Class": klass,
                         "X-RateLimit-Keyed-By": keyed_by},
                content={"detail": f"rate limit exceeded for {klass} requests; retry in {retry}s",
                         "klass": klass, "retry_after": retry},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Class"] = klass
        response.headers["X-RateLimit-Keyed-By"] = keyed_by
        return response
