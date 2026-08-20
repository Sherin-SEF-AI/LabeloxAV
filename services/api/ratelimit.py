"""A budget per caller, because there was none at all.

The API had exactly one 429 in it, a per-account login lockout, and nothing anywhere that bounded how fast a
caller could pull. That matters most on the media routes: `/api/frames/{id}/image`, the crops and the
segmentation rasters are frames of Indian roads under DPDPA, two of them take no user dependency at all, and
a loop against them is limited only by how fast the disk can read. Presigned-URL minting is the same shape
with a sharper edge, since each call hands out a credential that outlives the request.

**The token bucket is deliberately not a fixed window.** A fixed window lets a caller spend the whole budget
in the last second of one window and the whole of the next in the first second of the next, which is twice
the intended rate at exactly the moment a scraper is going fastest. A bucket refills continuously, so the
rate it enforces is the rate it advertises.

**Costs differ by route.** One frame read is one unit; minting a presigned URL is many, because the result is
a key to the object store that survives the connection. A single limit over both would either throttle
ordinary review or leave the credential path effectively open.

**Redis when it is there, memory when it is not.** Redis has been running and doing nothing but answering
liveness probes; a shared bucket is what makes a limit hold across more than one worker. Without it the
limiter still works per process, which is honest for a single-process deployment and is stated in the
response header rather than assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# Requests per second, sustained, and how much burst is tolerated above it. The burst is what makes an
# editor opening a frame with forty crops feel instant while a loop pulling the same route flat out does not.
@dataclass(frozen=True)
class Budget:
    rate_per_s: float
    burst: float

    def __post_init__(self) -> None:
        if self.rate_per_s <= 0 or self.burst <= 0:
            raise ValueError("a budget must allow something through")


# These are sized against what the application actually does, not against what a limiter looks tidy at.
# Opening ONE frame in the editor issues the frame, its objects, lanes, drivable, relationships, adverse,
# cuboids, segmentation, dynamics, the ontology, the users list, eleven filmstrip thumbnails and a crop per
# object: roughly seventy requests before anybody has drawn anything. A budget that reads as generous on
# paper is a broken editor in practice, and the first version of this file proved it.
#
# The threat being bounded is bulk exfiltration of frames of Indian roads, not a person working quickly. A
# reviewer peaks at a frame or two a second; an unbounded loop goes as fast as the disk. Sustained rates
# here sit far above the first and far below the second.

# The default budget for an authenticated caller doing ordinary work.
DEFAULT = Budget(rate_per_s=50.0, burst=200.0)

# Media reads: an editor opens a frame and pulls its crops in a burst, then goes quiet while somebody looks
# at it. The burst has to cover a whole page load with room for a reviewer paging quickly through frames.
MEDIA = Budget(rate_per_s=30.0, burst=300.0)

# Minting a presigned URL hands out a credential that outlives the request, so this stays the tightest
# budget. The burst covers a multipart upload's parts, which legitimately need one presign each.
PRESIGN = Budget(rate_per_s=5.0, burst=60.0)

# An unauthenticated caller gets enough to load the login page and sign in, and not enough to walk the
# media routes.
ANONYMOUS = Budget(rate_per_s=5.0, burst=40.0)


@dataclass
class Bucket:
    """One caller's allowance for one class of route."""

    tokens: float
    updated_at: float
    budget: Budget

    def take(self, cost: float, now: float) -> tuple[bool, float]:
        """Spend `cost` if it is there. Returns (allowed, seconds until it would be)."""
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.budget.burst, self.tokens + elapsed * self.budget.rate_per_s)
        self.updated_at = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        # How long until the shortfall has refilled, so the answer can say when to come back rather than
        # leaving a client to guess and hammer.
        return False, (cost - self.tokens) / self.budget.rate_per_s


@dataclass
class MemoryLimiter:
    """Per-process buckets. Correct for one worker and honest about being no more than that."""

    buckets: dict[tuple[str, str], Bucket] = field(default_factory=dict)
    # A cap, so a limiter meant to bound memory use cannot become the thing that exhausts it: an unbounded
    # dict keyed by client address is a memory leak with a rate limit written on it.
    max_buckets: int = 50_000

    def check(self, key: str, klass: str, budget: Budget, cost: float,
              now: float | None = None) -> tuple[bool, float]:
        now = time.monotonic() if now is None else now
        k = (key, klass)
        bucket = self.buckets.get(k)
        if bucket is None:
            if len(self.buckets) >= self.max_buckets:
                self._evict(now)
            bucket = Bucket(tokens=budget.burst, updated_at=now, budget=budget)
            self.buckets[k] = bucket
        return bucket.take(cost, now)

    def _evict(self, now: float) -> None:
        """Drop the buckets that are full, which is to say the callers who are not using their allowance.

        Evicting a full bucket costs nothing: a caller who comes back gets a fresh full one, which is what
        they would have had anyway.
        """
        stale = [k for k, b in self.buckets.items()
                 if b.tokens + (now - b.updated_at) * b.budget.rate_per_s >= b.budget.burst]
        for k in stale:
            del self.buckets[k]
        if not stale:
            # Nothing was idle, so drop the oldest half rather than refusing to serve anyone new.
            for k in sorted(self.buckets, key=lambda k: self.buckets[k].updated_at)[:len(self.buckets) // 2]:
                del self.buckets[k]


# The same token bucket, in one round trip and atomically.
#
# Read-modify-write from Python would be a race: two workers reading 1 token both see 1 and both allow, so
# the limit leaks by exactly the number of workers under the load it is meant to stop. Lua runs server-side
# on one thread, so the read, the refill and the take are indivisible.
#
# Keys expire after the time it takes an empty bucket to refill completely. A bucket at full is
# indistinguishable from one that never existed, so letting it lapse costs nothing and bounds the keyspace
# without an eviction pass.
_TOKEN_BUCKET_LUA = """
local key   = KEYS[1]
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local cost  = tonumber(ARGV[3])
local now   = tonumber(ARGV[4])

local b = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(b[1])
local ts     = tonumber(b[2])
if tokens == nil then tokens = burst; ts = now end

tokens = math.min(burst, tokens + (now - ts) * rate)

local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry = (cost - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(burst / rate) + 1)
return {allowed, tostring(retry)}
"""


class RedisLimiter:
    """Buckets shared across every worker, which is the only way a limit means what it advertises.

    Fails OPEN. A limiter is a guard rail, not the service: if Redis is unreachable the right behaviour is
    to serve the request and say so in the logs, not to take the API down defending it against a load that
    may not even be arriving. The middleware reports which backend answered, so a deployment that has
    quietly fallen back to per-process buckets is visible rather than assumed.
    """

    def __init__(self, client, prefix: str = "lbx:rl:") -> None:
        self._redis = client
        self._prefix = prefix
        self._sha: str | None = None
        self._fallback = MemoryLimiter()
        self.degraded = False

    async def check(self, key: str, klass: str, budget: Budget, cost: float,
                    now: float | None = None) -> tuple[bool, float]:
        at = time.time() if now is None else now
        try:
            if self._sha is None:
                self._sha = await self._redis.script_load(_TOKEN_BUCKET_LUA)
            allowed, retry = await self._redis.evalsha(
                self._sha, 1, f"{self._prefix}{klass}:{key}",
                budget.rate_per_s, budget.burst, cost, at)
            self.degraded = False
            return bool(int(allowed)), float(retry)
        except Exception:  # noqa: BLE001
            # NOSCRIPT after a Redis restart lands here too; the next call reloads the script.
            self._sha = None
            self.degraded = True
            return self._fallback.check(key, klass, budget, cost)


def classify(path: str, method: str) -> tuple[str, Budget, float]:
    """Which budget a request draws on, and what it costs. Returns (class, budget, cost).

    Keyed on the path rather than the resolved route, because the limiter runs before routing: a request
    for a path no route matches must still be counted, since that is what a scanner sends.
    """
    from services.api.media import is_media_read

    if path.endswith("/presign-put") or path.endswith("/presign") or "/presign" in path:
        return "presign", PRESIGN, 1.0
    if is_media_read(path):
        return "media", MEDIA, 1.0
    # Everything else, including writes. A write is not cheaper than a read here because the expensive part
    # of a write in this system is the work it queues, not the request.
    return "api", DEFAULT, 1.0


def retry_after_seconds(wait: float) -> int:
    """A whole number of seconds, never zero: Retry-After: 0 invites an immediate retry."""
    return max(1, int(wait + 0.999))
