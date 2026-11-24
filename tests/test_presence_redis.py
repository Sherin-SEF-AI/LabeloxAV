"""The Redis half of presence: that the real store keeps the contract the in-memory one is tested against.

test_presence_logic.py proves the logic over a fake. What it cannot prove is the three Lua scripts, which
are the entire reason release, refresh and take-over are safe: a compare done in Python and a write done
after it is a lock that the wrong tab can release. So the same operations are run here against a real
Redis, under a prefix nobody else uses and with TTLs short enough that nothing outlives the test.

Skips when Redis is not reachable, like the shared rate-limit test this is shaped after.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from services.collab.presence import (
    Identity,
    PresenceService,
    RedisPresenceStore,
    TakeOverDenied,
)

pytestmark = pytest.mark.infra

ASHA = Identity("u-asha", "Asha", "annotator")
BINU = Identity("u-binu", "Binu", "annotator")
RAVI = Identity("u-ravi", "Ravi", "reviewer")
TAB_A, TAB_A2, TAB_B, TAB_R = ("tab-asha-aaaaaaaaaaaa", "tab-asha-second-aaaa", "tab-binu-bbbbbbbbbbbb",
                               "tab-ravi-rrrrrrrrrrrr")


async def _store() -> RedisPresenceStore:
    aioredis = pytest.importorskip("redis.asyncio", reason="redis client not installed")
    from core.config import get_settings

    cfg = get_settings().redis
    client = aioredis.Redis(host=cfg.host, port=cfg.port, db=getattr(cfg, "db", 0),
                            socket_timeout=1.0, socket_connect_timeout=1.0)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"redis not reachable: {exc}")
    return RedisPresenceStore(client, prefix=f"lbx:collab:test:{uuid.uuid4().hex}:")


def _frame() -> str:
    return str(uuid.uuid4())


async def test_set_if_free_is_exclusive_and_expires():
    store, frame = await _store(), _frame()
    assert await store.set_lock_if_free(frame, f"{TAB_A}|a", 150) is True
    assert await store.set_lock_if_free(frame, f"{TAB_B}|b", 150) is False
    assert await store.get_lock(frame) == f"{TAB_A}|a"
    await asyncio.sleep(0.3)
    assert await store.get_lock(frame) is None
    assert await store.set_lock_if_free(frame, f"{TAB_B}|b", 150) is True


async def test_a_concurrent_race_has_one_winner():
    store, frame = await _store(), _frame()
    wins = await asyncio.gather(*(store.set_lock_if_free(frame, f"tab-racer-{i:012d}|x", 2000)
                                  for i in range(20)))
    assert sum(wins) == 1
    await store.swap_lock(frame, "done|x", 1)


async def test_refresh_and_release_compare_the_token():
    store, frame = await _store(), _frame()
    await store.set_lock_if_free(frame, f"{TAB_A}|a", 2000)
    # Not the holder, and not a prefix of the holder either.
    for stranger in (TAB_B, TAB_A[:-1]):
        assert await store.refresh_lock(frame, stranger, 2000) is False
        assert await store.release_lock(frame, stranger) is False
    assert await store.get_lock(frame) == f"{TAB_A}|a"
    assert await store.refresh_lock(frame, TAB_A, 2000) is True
    assert await store.release_lock(frame, TAB_A) is True
    assert await store.get_lock(frame) is None
    # Nothing to refresh or release once it is gone.
    assert await store.refresh_lock(frame, TAB_A, 2000) is False
    assert await store.release_lock(frame, TAB_A) is False


async def test_refresh_extends_the_ttl():
    store, frame = await _store(), _frame()
    await store.set_lock_if_free(frame, f"{TAB_A}|a", 200)
    await asyncio.sleep(0.12)
    assert await store.refresh_lock(frame, TAB_A, 400) is True
    await asyncio.sleep(0.2)                       # past the original expiry
    assert await store.get_lock(frame) == f"{TAB_A}|a"
    await store.release_lock(frame, TAB_A)


async def test_swap_reports_who_was_there_and_honours_the_condition():
    store, frame = await _store(), _frame()
    assert await store.swap_lock(frame, f"{TAB_A}|a", 2000) == (True, None)
    assert await store.swap_lock(frame, f"{TAB_B}|b", 2000, only_if_token=TAB_R) == (False, f"{TAB_A}|a")
    assert await store.get_lock(frame) == f"{TAB_A}|a"
    assert await store.swap_lock(frame, f"{TAB_B}|b", 2000, only_if_token=TAB_A) == (True, f"{TAB_A}|a")
    assert await store.swap_lock(frame, f"{TAB_R}|r", 2000) == (True, f"{TAB_B}|b")
    await store.release_lock(frame, TAB_R)
    # Conditional on a holder, and there is none.
    assert await store.swap_lock(frame, f"{TAB_A}|a", 2000, only_if_token=TAB_A) == (False, None)


async def test_presence_members_lapse_by_score_and_leave_at_once():
    store, frame = await _store(), _frame()
    assert await store.touch(frame, "asha", 1_000, 500) == ["asha"]
    assert sorted(await store.touch(frame, "binu", 1_200, 500)) == ["asha", "binu"]
    assert await store.touch(frame, "binu", 1_600, 500) == ["binu"]      # asha last seen 600 ms ago
    await store.leave(frame, "binu")
    assert await store.touch(frame, "ravi", 1_700, 500) == ["ravi"]
    await store.leave(frame, "ravi")


async def test_the_service_end_to_end_over_redis():
    store, frame = await _store(), _frame()
    svc = PresenceService(store, clock=time.time, presence_ttl_s=2.0, lock_ttl_s=2.0)
    try:
        assert (await svc.acquire(frame, ASHA, TAB_A))["lock"]["mine"] is True
        seen = await svc.acquire(frame, BINU, TAB_B)
        assert seen["lock"]["mine"] is False and seen["lock"]["holder"]["name"] == "Asha"
        assert [p["name"] for p in seen["others"]] == ["Asha"]
        assert (await svc.acquire(frame, ASHA, TAB_A2))["lock"]["holder"]["same_user"] is True
        with pytest.raises(TakeOverDenied):
            await svc.take_over(frame, BINU, TAB_B, privileged=False)
        state, displaced = await svc.take_over(frame, RAVI, TAB_R, privileged=True)
        assert state["lock"]["mine"] is True and displaced is not None and displaced.name == "Asha"
        assert (await svc.heartbeat(frame, ASHA, TAB_A, acquire=True))["lock"]["mine"] is False
        assert await svc.blocking_holder(frame, "u-asha", TAB_A) is not None
        assert await svc.blocking_holder(frame, "u-ravi", TAB_R) is None
        assert (await svc.release(frame, ASHA, TAB_A))["released"] is False
        assert (await svc.release(frame, RAVI, TAB_R))["released"] is True
    finally:
        for who, tab in ((ASHA, TAB_A), (ASHA, TAB_A2), (BINU, TAB_B), (RAVI, TAB_R)):
            await svc.release(frame, who, tab)
