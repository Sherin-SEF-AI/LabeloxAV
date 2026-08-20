"""Short-lived, single-use challenge state that survives being read by a different worker.

MFA handles and OIDC PKCE state were plain module-level dicts. Under more than one uvicorn worker the
follow-up request lands wherever the load balancer sends it, so an (N-1)/N share of sign-ins failed with
"this sign-in has expired; start again" - and the failure is indistinguishable from a genuinely expired
handle, so it reads as flakiness rather than as a deployment that cannot run two workers.

services/api/identity.py:15 already records that account lockout was put in the database *because* this
deployment can run more than one API replica. That reasoning simply was not carried through to the two
stores that gate the sign-in itself.

Redis rather than Postgres: these entries live for a couple of minutes, are written once and read once, and
want a TTL rather than a sweep. Falling back to a process-local dict when Redis is absent keeps a
single-worker or laptop deployment working exactly as before, which is also what the test suite runs on.

Single-use is enforced here rather than by the caller. `take` deletes and returns in one step, so a
replayed handle finds nothing whether the backing store is shared or local - the property both call sites
were relying on their `dict.pop` for.
"""

from __future__ import annotations

import json
import time
from typing import Any

from core.logging import get_logger

log = get_logger("api.ephemeral")

# Keyed by the running event loop, not a single global. An asyncio Redis client binds its connection pool
# to the loop it is first used on, so one cached client outlives the loop that made it and every later call
# fails with "Event loop is closed" - which this store would then silently absorb into its process-local
# fallback, quietly undoing the whole point. One client per loop, and a loop that goes away takes its
# client with it.
_clients: dict[Any, Any] = {}


def _redis():
    """The client for the running loop, or None if Redis is unavailable here."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    if loop in _clients:
        return _clients[loop]
    try:
        import redis.asyncio as aioredis

        from core.config import get_settings

        cfg = get_settings().redis
        client = aioredis.Redis(host=cfg.host, port=cfg.port, db=getattr(cfg, "db", 0),
                                socket_timeout=0.25, socket_connect_timeout=0.25)
    except Exception as exc:  # noqa: BLE001
        log.warning("ephemeral.redis_unavailable", error=str(exc),
                    note="challenge state is per-process; do not run more than one API worker")
        client = None
    # Bounded: a long-lived process has one loop, and a test process makes one per test. Clearing at a
    # low ceiling keeps this from being a slow leak in the suite without needing loop close hooks.
    if len(_clients) > 64:
        _clients.clear()
    _clients[loop] = client
    return client


class EphemeralStore:
    """put/take with a TTL. Shared across workers when Redis answers, process-local when it does not."""

    def __init__(self, namespace: str, ttl_s: float) -> None:
        self._ns = namespace
        self._ttl = ttl_s
        self._local: dict[str, tuple[str, float]] = {}

    def _key(self, handle: str) -> str:
        return f"lbx:eph:{self._ns}:{handle}"

    def _sweep_local(self, now: float) -> None:
        for k, (_, at) in list(self._local.items()):
            if now - at > self._ttl:
                self._local.pop(k, None)

    async def put(self, handle: str, value: dict) -> None:
        payload = json.dumps(value)
        client = _redis()
        if client is not None:
            try:
                await client.set(self._key(handle), payload, ex=int(self._ttl) + 1)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("ephemeral.put_failed", ns=self._ns, error=str(exc))
        now = time.time()
        self._sweep_local(now)
        self._local[handle] = (payload, now)

    async def take(self, handle: str | None) -> dict | None:
        """Read and delete in one step. A second attempt with the same handle finds nothing."""
        if not handle:
            return None
        client = _redis()
        if client is not None:
            try:
                # GETDEL is atomic, so two workers racing the same handle cannot both consume it.
                raw = await client.getdel(self._key(handle))
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                log.warning("ephemeral.take_failed", ns=self._ns, error=str(exc))
        now = time.time()
        self._sweep_local(now)
        entry = self._local.pop(handle, None)
        if entry is None:
            return None
        payload, at = entry
        if now - at > self._ttl:
            return None
        return json.loads(payload)
