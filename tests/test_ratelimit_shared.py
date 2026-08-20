"""A rate limit that N workers each enforce separately is a rate limit of N times the advertised rate.

services/api/ratelimit.py's docstring said "Redis when it is there, memory when it is not" and there was no
Redis limiter in the codebase: the middleware constructed a MemoryLimiter unconditionally. Two API workers
therefore allowed twice the budget, four allowed four times, and X-RateLimit-Keyed-By reported nothing
about it, so the limit looked like it held. That matters most on the media routes the limiter exists for -
frames of Indian roads under DPDPA, some of which take no user dependency at all.

The property under test is not "it limits" (test_ratelimit.py covers the bucket arithmetic). It is that
two independently constructed limiters, standing in for two workers, draw down one budget.
"""

from __future__ import annotations

import uuid

import pytest

from services.api.ratelimit import Budget, MemoryLimiter, RedisLimiter

pytestmark = pytest.mark.infra


def _redis():
    aioredis = pytest.importorskip("redis.asyncio", reason="redis client not installed")
    from core.config import get_settings

    cfg = get_settings().redis
    return aioredis.Redis(host=cfg.host, port=cfg.port, db=getattr(cfg, "db", 0),
                          socket_timeout=1.0, socket_connect_timeout=1.0)


async def _skip_unless_redis(client):
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"redis not reachable: {exc}")


class TestOneBudgetAcrossWorkers:
    async def test_two_limiters_share_one_bucket(self):
        client = _redis()
        await _skip_unless_redis(client)
        # Two limiter objects, no shared Python state between them: this is what two uvicorn workers are.
        worker_a, worker_b = RedisLimiter(client), RedisLimiter(client)
        key = f"test:{uuid.uuid4()}"
        budget = Budget(rate_per_s=1.0, burst=4.0)

        spent = 0
        for _ in range(4):
            allowed, _ = await worker_a.check(key, "media", budget, 1.0)
            spent += int(allowed)
        assert spent == 4, "the first worker should be able to spend the whole burst"

        # The budget is now empty. A second worker must see that, which is the entire point.
        allowed, retry = await worker_b.check(key, "media", budget, 1.0)
        assert allowed is False, (
            "a second worker got its own full bucket; the limit is per process and N workers enforce N "
            "times the advertised rate")
        assert retry > 0

    async def test_a_memory_limiter_does_not_share_which_is_why_this_matters(self):
        # The contrast, stated so the test above cannot be misread as trivially true.
        a, b = MemoryLimiter(), MemoryLimiter()
        key = f"test:{uuid.uuid4()}"
        budget = Budget(rate_per_s=1.0, burst=2.0)
        for _ in range(2):
            a.check(key, "media", budget, 1.0)
        assert a.check(key, "media", budget, 1.0)[0] is False
        assert b.check(key, "media", budget, 1.0)[0] is True, (
            "two MemoryLimiters are independent - this is the behaviour RedisLimiter replaces")

    async def test_the_refill_is_continuous(self):
        client = _redis()
        await _skip_unless_redis(client)
        limiter = RedisLimiter(client)
        key = f"test:{uuid.uuid4()}"
        budget = Budget(rate_per_s=10.0, burst=2.0)
        # Drain, then hand the bucket a timestamp a second later rather than sleeping: the arithmetic is
        # what is under test, and a test that sleeps to prove a refill is a slow test that proves a clock.
        now = 1_000_000.0
        for _ in range(2):
            await limiter.check(key, "media", budget, 1.0, now=now)
        assert (await limiter.check(key, "media", budget, 1.0, now=now))[0] is False
        assert (await limiter.check(key, "media", budget, 1.0, now=now + 1.0))[0] is True


class TestItFailsOpen:
    async def test_an_unreachable_redis_serves_the_request(self):
        """A limiter is a guard rail, not the service.

        If Redis is down the right behaviour is to serve and say so, not to take the API down defending it
        against load that may not be arriving. The fallback is per-process, which is the old behaviour, and
        `degraded` is what the middleware reports in X-RateLimit-Backend so this is visible.
        """
        aioredis = pytest.importorskip("redis.asyncio")
        dead = aioredis.Redis(host="127.0.0.1", port=1, socket_timeout=0.05,
                              socket_connect_timeout=0.05)
        limiter = RedisLimiter(dead)
        allowed, _ = await limiter.check(f"test:{uuid.uuid4()}", "media", Budget(1.0, 4.0), 1.0)
        assert allowed is True
        assert limiter.degraded is True


class TestChallengeStateSurvivesADifferentWorker:
    """MFA handles and OIDC PKCE state were module-level dicts.

    The second-factor request, and the OIDC callback in particular - which is a *browser redirect* and can
    land anywhere - reached whichever worker the load balancer chose. Under N workers an (N-1)/N share of
    sign-ins failed with "this sign-in has expired", which is indistinguishable from a genuinely expired
    handle and so read as flakiness rather than as a deployment that cannot run two workers.
    """

    async def test_one_store_can_read_what_another_wrote(self):
        from services.api.ephemeral import EphemeralStore, _redis

        if _redis() is None:
            pytest.skip("redis not reachable")
        # Two store objects with no shared Python state: this is what two workers are.
        worker_a = EphemeralStore("test-mfa", 300)
        worker_b = EphemeralStore("test-mfa", 300)
        handle = f"h-{uuid.uuid4()}"

        await worker_a.put(handle, {"user_id": "u-1"})
        got = await worker_b.take(handle)
        assert got == {"user_id": "u-1"}, (
            "the second worker could not see the challenge the first issued; sign-in is per-process again")

    async def test_a_handle_is_still_single_use_across_workers(self):
        from services.api.ephemeral import EphemeralStore, _redis

        if _redis() is None:
            pytest.skip("redis not reachable")
        a, b = EphemeralStore("test-mfa", 300), EphemeralStore("test-mfa", 300)
        handle = f"h-{uuid.uuid4()}"
        await a.put(handle, {"user_id": "u-1"})
        assert await b.take(handle) is not None
        # Read and delete are one atomic step, so two workers racing cannot both consume it.
        assert await a.take(handle) is None

    async def test_it_still_works_with_no_redis_at_all(self):
        # A single-worker or laptop deployment keeps the previous behaviour exactly.
        from services.api.ephemeral import EphemeralStore

        store = EphemeralStore("test-local", 300)
        handle = f"h-{uuid.uuid4()}"
        await store.put(handle, {"user_id": "u-2"})
        assert await store.take(handle) == {"user_id": "u-2"}
        assert await store.take(handle) is None
