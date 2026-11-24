"""Who is on a frame right now, and which one of them may edit it.

Two people could open the same frame and neither was told. The only protection was the optimistic version
on each object, which speaks once, after the work is done: the second save answers 409 and the editor says
"reload to continue". Nothing warned either of them beforehand.

**Presence** is a heartbeat with a short TTL. A tab says "I am here" every few seconds; a tab that stops
saying so is gone a few seconds later, with no sweep and no "leave" message that a closed laptop would
never send.

**The edit lock** is one key per frame, taken with SET NX PX, so two tabs racing for a free frame cannot
both win. It is refreshed by the same heartbeat and expires by itself if the holder disappears.

**The lock belongs to a tab, not a person.** Release and refresh compare a per-tab token. Keyed on the user
alone, a second tab of the same person would silently share the lock and the two would overwrite each
other exactly as two people do. The token never leaves the server again: other tabs are told who holds
the frame, never the secret that would let them release it.

**It fails open.** Presence is a courtesy on top of the version check, not the thing that keeps data
correct. If Redis is unreachable every call answers "nobody here, nobody holds it, degraded", and the
editor carries on with the protection it had before this existed.

The logic sits behind a small storage interface so it can be exercised without Redis or a clock:
`MemoryPresenceStore` is that interface over two dicts, `RedisPresenceStore` is the real one.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from core.logging import get_logger

log = get_logger("collab.presence")

# A tab is present for this long after its last heartbeat, and the client beats three times inside it, so
# one dropped request does not make somebody vanish and come back.
PRESENCE_TTL_S = 15.0
HEARTBEAT_S = 5.0
# The lock outlives its holder by the same margin: long enough to ride out a slow request, short enough
# that a closed laptop frees the frame before anybody has to go and find a reviewer.
LOCK_TTL_S = 15.0

# What a tab token may look like. It is spliced into a Redis value with a separator, so the alphabet is
# closed here rather than escaped there.
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


@dataclass(frozen=True)
class Identity:
    """The acting user, as resolved from their credential. Never built from anything a client typed."""

    user_id: str
    name: str
    role: str


@dataclass(frozen=True)
class LockHolder:
    token: str
    user_id: str
    name: str
    role: str
    since_ms: int

    def public(self, viewer: Identity | None = None) -> dict:
        """The holder as another tab may see it: who and since when, never the token."""
        return {"user_id": self.user_id, "name": self.name, "role": self.role, "since_ms": self.since_ms,
                "same_user": viewer is not None and viewer.user_id == self.user_id}


class TakeOverDenied(Exception):
    """The caller may not take this frame from whoever holds it."""

    def __init__(self, holder: LockHolder) -> None:
        super().__init__(f"only a reviewer or an admin can take a frame over from {holder.name}")
        self.holder = holder


class FrameLockedError(Exception):
    """Somebody else holds the frame's edit lock."""

    def __init__(self, holder: LockHolder) -> None:
        super().__init__(f"this frame is being edited by {holder.name}")
        self.holder = holder


# ---- storage ---------------------------------------------------------------------------------------------


class PresenceStore(Protocol):
    """The few operations the logic needs, each one atomic. Values are opaque strings to the store, except
    that a lock value begins with its holder's token and a `|`, which is what release and refresh compare."""

    async def touch(self, frame_id: str, member: str, now_ms: int, ttl_ms: int) -> list[str]:
        """Record `member` as seen at `now_ms`, forget anyone not seen for `ttl_ms`, return who is left."""
        ...

    async def leave(self, frame_id: str, member: str) -> None: ...

    async def get_lock(self, frame_id: str) -> str | None: ...

    async def set_lock_if_free(self, frame_id: str, value: str, ttl_ms: int) -> bool: ...

    async def refresh_lock(self, frame_id: str, token: str, ttl_ms: int) -> bool:
        """Extend the lock, only if `token` holds it."""
        ...

    async def release_lock(self, frame_id: str, token: str) -> bool:
        """Delete the lock, only if `token` holds it."""
        ...

    async def swap_lock(self, frame_id: str, value: str, ttl_ms: int,
                        only_if_token: str | None = None) -> tuple[bool, str | None]:
        """Replace the lock whoever holds it, or only while `only_if_token` still does. Returns whether it
        was replaced, and the value that was there."""
        ...


def _held_by(value: str | None, token: str) -> bool:
    return value is not None and value.startswith(token + "|")


class MemoryPresenceStore:
    """The interface over two dicts and an injected clock. For tests, and for nothing else: it is one
    process's view, which is the very thing presence exists to see past."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._seen: dict[str, dict[str, int]] = {}
        self._locks: dict[str, tuple[str, float]] = {}      # frame -> (value, expires_at seconds)

    def _live(self, frame_id: str) -> str | None:
        entry = self._locks.get(frame_id)
        if entry is None:
            return None
        if self._clock() >= entry[1]:
            del self._locks[frame_id]
            return None
        return entry[0]

    async def touch(self, frame_id: str, member: str, now_ms: int, ttl_ms: int) -> list[str]:
        seen = self._seen.setdefault(frame_id, {})
        seen[member] = now_ms
        for m, at in list(seen.items()):
            if at <= now_ms - ttl_ms:
                del seen[m]
        return sorted(seen)

    async def leave(self, frame_id: str, member: str) -> None:
        self._seen.get(frame_id, {}).pop(member, None)

    async def get_lock(self, frame_id: str) -> str | None:
        return self._live(frame_id)

    async def set_lock_if_free(self, frame_id: str, value: str, ttl_ms: int) -> bool:
        if self._live(frame_id) is not None:
            return False
        self._locks[frame_id] = (value, self._clock() + ttl_ms / 1000)
        return True

    async def refresh_lock(self, frame_id: str, token: str, ttl_ms: int) -> bool:
        value = self._live(frame_id)
        if not _held_by(value, token):
            return False
        self._locks[frame_id] = (value or "", self._clock() + ttl_ms / 1000)
        return True

    async def release_lock(self, frame_id: str, token: str) -> bool:
        if not _held_by(self._live(frame_id), token):
            return False
        del self._locks[frame_id]
        return True

    async def swap_lock(self, frame_id: str, value: str, ttl_ms: int,
                        only_if_token: str | None = None) -> tuple[bool, str | None]:
        current = self._live(frame_id)
        if only_if_token is not None and not _held_by(current, only_if_token):
            return False, current
        self._locks[frame_id] = (value, self._clock() + ttl_ms / 1000)
        return True, current


# Compare-then-act has to happen in one step on the server. Read the lock from Python, decide, then write,
# and a lock that expired and was re-taken between the read and the write is released or extended by the
# tab that no longer holds it. Lua runs on Redis's one thread, so the comparison and the write are
# indivisible. ARGV[1] is always the token being compared.
_REFRESH_LUA = """
local v = redis.call('GET', KEYS[1])
if v and string.sub(v, 1, string.len(ARGV[1]) + 1) == ARGV[1] .. '|' then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  return 1
end
return 0
"""

_RELEASE_LUA = """
local v = redis.call('GET', KEYS[1])
if v and string.sub(v, 1, string.len(ARGV[1]) + 1) == ARGV[1] .. '|' then
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
"""

# ARGV: expected token ('' for "whoever holds it"), the new value, the TTL in ms.
_SWAP_LUA = """
local v = redis.call('GET', KEYS[1])
if ARGV[1] ~= '' then
  if not v or string.sub(v, 1, string.len(ARGV[1]) + 1) ~= ARGV[1] .. '|' then
    return {0, v}
  end
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
return {1, v}
"""


def _text(raw: Any) -> str | None:
    if raw is None:
        return None
    return raw.decode() if isinstance(raw, bytes | bytearray) else str(raw)


class RedisPresenceStore:
    """Presence as a sorted set scored by last-seen time, the lock as one string key with a TTL."""

    def __init__(self, client: Any, prefix: str = "lbx:collab:") -> None:
        self._r = client
        self._prefix = prefix

    def _presence(self, frame_id: str) -> str:
        return f"{self._prefix}presence:{frame_id}"

    def _lock(self, frame_id: str) -> str:
        return f"{self._prefix}lock:{frame_id}"

    async def touch(self, frame_id: str, member: str, now_ms: int, ttl_ms: int) -> list[str]:
        key = self._presence(frame_id)
        pipe = self._r.pipeline(transaction=False)
        pipe.zadd(key, {member: now_ms})
        pipe.zremrangebyscore(key, "-inf", now_ms - ttl_ms)
        pipe.zrange(key, 0, -1)
        # The whole set lapses once nobody has beaten for a TTL, so an abandoned frame leaves no key.
        pipe.pexpire(key, ttl_ms)
        _, _, members, _ = await pipe.execute()
        return [m for m in (_text(x) for x in members) if m is not None]

    async def leave(self, frame_id: str, member: str) -> None:
        await self._r.zrem(self._presence(frame_id), member)

    async def get_lock(self, frame_id: str) -> str | None:
        return _text(await self._r.get(self._lock(frame_id)))

    async def set_lock_if_free(self, frame_id: str, value: str, ttl_ms: int) -> bool:
        return bool(await self._r.set(self._lock(frame_id), value, nx=True, px=ttl_ms))

    async def refresh_lock(self, frame_id: str, token: str, ttl_ms: int) -> bool:
        return bool(int(await self._r.eval(_REFRESH_LUA, 1, self._lock(frame_id), token, ttl_ms)))

    async def release_lock(self, frame_id: str, token: str) -> bool:
        return bool(int(await self._r.eval(_RELEASE_LUA, 1, self._lock(frame_id), token)))

    async def swap_lock(self, frame_id: str, value: str, ttl_ms: int,
                        only_if_token: str | None = None) -> tuple[bool, str | None]:
        done, previous = await self._r.eval(_SWAP_LUA, 1, self._lock(frame_id),
                                            only_if_token or "", value, ttl_ms)
        return bool(int(done)), _text(previous)


# ---- the logic -------------------------------------------------------------------------------------------


def _tab_id(token: str) -> str:
    """A tab's public handle. Presence is listed to everyone on the frame, so it must not carry the token."""
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _member(who: Identity, token: str) -> str:
    return json.dumps({"u": who.user_id, "n": who.name, "r": who.role, "t": _tab_id(token)},
                      sort_keys=True, separators=(",", ":"))


def _lock_value(who: Identity, token: str, now_ms: int) -> str:
    body = json.dumps({"u": who.user_id, "n": who.name, "r": who.role, "at": now_ms},
                      sort_keys=True, separators=(",", ":"))
    return f"{token}|{body}"


def parse_lock(value: str | None) -> LockHolder | None:
    """The holder a stored lock value names, or None for no lock or a value nobody can read. An unreadable
    lock is treated as no lock: it will expire, and refusing every edit until it does would be the lock
    failing closed over a corrupt string."""
    if not value or "|" not in value:
        return None
    token, body = value.split("|", 1)
    try:
        d = json.loads(body)
        return LockHolder(token=token, user_id=str(d["u"]), name=str(d["n"]), role=str(d["r"]),
                          since_ms=int(d["at"]))
    except (ValueError, KeyError, TypeError):
        return None


def degraded_state(frame_id: str, reason: str = "presence store unavailable") -> dict:
    """What every endpoint answers when the store cannot be reached: nobody here, nobody holds it, and say
    so. The client reads `degraded` as "edit as you could before"."""
    return {"frame_id": frame_id, "degraded": True, "reason": reason, "others": [],
            "lock": {"mine": False, "holder": None}, "can_take_over": False,
            "ttl_s": PRESENCE_TTL_S, "heartbeat_s": HEARTBEAT_S}


class PresenceService:
    def __init__(self, store: PresenceStore | Callable[[], PresenceStore], *,
                 clock: Callable[[], float] = time.time,
                 presence_ttl_s: float = PRESENCE_TTL_S, lock_ttl_s: float = LOCK_TTL_S) -> None:
        self._source = store
        self._clock = clock
        self._presence_ttl_ms = int(presence_ttl_s * 1000)
        self._lock_ttl_ms = int(lock_ttl_s * 1000)

    @property
    def _store(self) -> PresenceStore:
        # A factory is resolved per call and inside the caller's try block, so a Redis client that cannot
        # even be constructed is a failed operation (answered as `degraded`), not a startup error.
        return self._source() if callable(self._source) else self._source

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _state(self, frame_id: str, who: Identity, token: str, members: list[str],
               holder: LockHolder | None, privileged: bool) -> dict:
        mine = holder is not None and holder.token == token
        people: dict[str, dict] = {}
        for raw in members:
            try:
                m = json.loads(raw)
                uid = str(m["u"])
                if uid == who.user_id:
                    continue        # this tab, or another of the same person's: not somebody else
                people[uid] = {"user_id": uid, "name": str(m["n"]), "role": str(m["r"]),
                               "editing": holder is not None and holder.user_id == uid}
            except (ValueError, KeyError, TypeError):
                continue
        return {
            "frame_id": frame_id, "degraded": False,
            "others": sorted(people.values(), key=lambda p: (p["name"].lower(), p["user_id"])),
            "lock": {"mine": mine, "holder": holder.public(who) if holder is not None else None},
            # Shown so the client can offer the button; the take-over itself checks again.
            "can_take_over": (holder is not None and not mine
                              and (privileged or holder.user_id == who.user_id)),
            "ttl_s": self._presence_ttl_ms / 1000, "heartbeat_s": HEARTBEAT_S,
        }

    async def heartbeat(self, frame_id: str, who: Identity, token: str, *, acquire: bool = False,
                        privileged: bool = False) -> dict:
        """Say "this tab is here", keep the lock alive if this tab holds it, and report the frame.

        With `acquire`, also take the lock if nobody holds it. That is how an editor whose lock lapsed
        while its tab was in the background gets it back in the same round trip, provided nobody took it.
        """
        try:
            now_ms = self._now_ms()
            members = await self._store.touch(frame_id, _member(who, token), now_ms, self._presence_ttl_ms)
            held = await self._store.refresh_lock(frame_id, token, self._lock_ttl_ms)
            if not held and acquire:
                await self._store.set_lock_if_free(frame_id, _lock_value(who, token, now_ms),
                                                   self._lock_ttl_ms)
            # Read back rather than trusting the two answers above: whoever the store says holds it now is
            # the truth, including when the acquire lost a race.
            holder = parse_lock(await self._store.get_lock(frame_id))
            return self._state(frame_id, who, token, members, holder, privileged)
        except Exception as exc:  # noqa: BLE001
            log.warning("collab.presence.degraded", op="heartbeat", error=str(exc))
            return degraded_state(frame_id)

    async def acquire(self, frame_id: str, who: Identity, token: str, *, privileged: bool = False) -> dict:
        return await self.heartbeat(frame_id, who, token, acquire=True, privileged=privileged)

    async def release(self, frame_id: str, who: Identity, token: str, *, leave: bool = True) -> dict:
        """Give the lock back, if this tab holds it, and by default stop being present. A tab that does
        not hold the lock releases nothing, whoever it belongs to."""
        try:
            released = await self._store.release_lock(frame_id, token)
            if leave:
                await self._store.leave(frame_id, _member(who, token))
            return {"frame_id": frame_id, "released": released, "degraded": False}
        except Exception as exc:  # noqa: BLE001
            log.warning("collab.presence.degraded", op="release", error=str(exc))
            return {"frame_id": frame_id, "released": False, "degraded": True}

    async def take_over(self, frame_id: str, who: Identity, token: str, *,
                        privileged: bool) -> tuple[dict, LockHolder | None]:
        """Take the lock from whoever holds it. Returns the frame's state and the holder who was displaced
        (None when nobody was, which is then no different from acquiring).

        `privileged` is the caller's role cleared by the route. Without it a person may still take the
        frame from their own other tab, and from nobody else.
        """
        try:
            now_ms = self._now_ms()
            value = _lock_value(who, token, now_ms)
            current = parse_lock(await self._store.get_lock(frame_id))
            displaced: LockHolder | None = None
            if current is None:
                await self._store.set_lock_if_free(frame_id, value, self._lock_ttl_ms)
            elif current.token == token:
                await self._store.refresh_lock(frame_id, token, self._lock_ttl_ms)
            elif privileged:
                # Whoever holds it by now, which may no longer be `current`: the displaced holder is the
                # one the swap reports, so the record names the person actually interrupted.
                _, previous = await self._store.swap_lock(frame_id, value, self._lock_ttl_ms)
                displaced = parse_lock(previous)
            elif current.user_id == who.user_id:
                # Only while that same tab still holds it. Unconditional here would let an annotator take
                # a frame from a stranger who acquired it between the read above and this write.
                swapped, _ = await self._store.swap_lock(frame_id, value, self._lock_ttl_ms,
                                                         only_if_token=current.token)
                displaced = current if swapped else None
            else:
                raise TakeOverDenied(current)
            if displaced is not None and displaced.token == token:
                displaced = None
            members = await self._store.touch(frame_id, _member(who, token), now_ms, self._presence_ttl_ms)
            holder = parse_lock(await self._store.get_lock(frame_id))
            return self._state(frame_id, who, token, members, holder, privileged), displaced
        except TakeOverDenied:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("collab.presence.degraded", op="take_over", error=str(exc))
            return degraded_state(frame_id), None

    async def blocking_holder(self, frame_id: str, user_id: str, token: str | None = None) -> LockHolder | None:
        """Who stops this caller editing the frame, or None when nothing does.

        Nothing does when the frame is unlocked, when this tab holds the lock, or when the store cannot be
        reached (fail open). A caller that sends no tab token is compared by user instead: weaker, since it
        cannot tell one person's two tabs apart, but it still stops everybody else.
        """
        try:
            holder = parse_lock(await self._store.get_lock(frame_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("collab.presence.degraded", op="blocking_holder", error=str(exc))
            return None
        if holder is None:
            return None
        if token is not None:
            return None if holder.token == token else holder
        return None if holder.user_id == str(user_id) else holder


# ---- the running application's instance --------------------------------------------------------------------

# Keyed by the running event loop, for the reason services/api/ephemeral.py gives: an asyncio Redis client
# binds its pool to the loop it is first used on, and a single cached client outlives that loop.
_clients: dict[Any, Any] = {}


def _redis() -> Any:
    import asyncio

    loop = asyncio.get_running_loop()
    if loop in _clients:
        return _clients[loop]
    import redis.asyncio as aioredis

    from core.config import get_settings

    cfg = get_settings().redis
    # Short timeouts: a heartbeat arrives every few seconds from every open editor, and one that waits on a
    # dead Redis would hold an API worker for as long as the default allows.
    client = aioredis.Redis(host=cfg.host, port=cfg.port, db=getattr(cfg, "db", 0),
                            socket_timeout=0.25, socket_connect_timeout=0.25)
    if len(_clients) > 64:
        _clients.clear()
    _clients[loop] = client
    return client


_service: PresenceService | None = None


def get_presence() -> PresenceService:
    global _service
    if _service is None:
        _service = PresenceService(lambda: RedisPresenceStore(_redis()))
    return _service


def identity_of(user: Any) -> Identity:
    """The Identity for an authenticated User row."""
    return Identity(user_id=str(user.user_id), name=str(user.name), role=str(user.role or ""))


async def assert_can_edit(frame_id: Any, user: Any, tab_token: str | None = None) -> None:
    """Refuse an edit to a frame somebody else holds. The call a save path makes before it writes.

    Raises HTTP 423 naming the holder. Passes when the frame is unlocked, when the caller's tab holds the
    lock, when no user is resolved (auth disabled) and when Redis is down: this narrows the window the
    version check already covers, and must never be the reason a save fails by itself.

    `tab_token` is the editor's per-tab token (the `X-Lbx-Tab` request header). Without it the holder is
    compared by user, which lets one person's second tab through.
    """
    if user is None:
        return
    token = tab_token if tab_token and TOKEN_RE.match(tab_token) else None
    holder = await get_presence().blocking_holder(str(frame_id), str(user.user_id), token)
    if holder is not None:
        from fastapi import HTTPException

        raise HTTPException(status_code=423, detail=str(FrameLockedError(holder)),
                            headers={"X-Lbx-Lock-Holder": holder.user_id})


async def assert_editor_may_save(frame_ids, user: Any, tab_token: str | None) -> None:
    """The save-path check: refuse a write from the frame editor to a frame another editor holds.

    Only a request that says it comes from the editor (it carries the editor's tab token, the `X-Lbx-Tab`
    header) is checked. That scoping is deliberate. The lock means "somebody has this frame open to edit";
    it is what stops two editors overwriting each other, and the editor is the only client that takes it.
    The review queue, a bulk accept across five hundred frames, the agents and the SDK write objects too,
    without ever opening a frame, and they are already protected by the per-object version check. Applying
    the lock to them would let one forgotten editor tab block a reviewer's whole batch.

    Raises HTTP 423 naming the holder on the first frame held by someone else.
    """
    if user is None or not tab_token or not TOKEN_RE.match(tab_token):
        return
    for frame_id in dict.fromkeys(str(f) for f in frame_ids if f is not None):
        await assert_can_edit(frame_id, user, tab_token)

