"""Presence and the frame edit lock, as logic: no Redis, no database, no wall clock.

Everything here runs against the in-memory store with an injected clock, so "the holder disappeared and the
lock expired" is a line that advances a number rather than a sleep. The HTTP surface (auth, the project
guard, the activity record a take-over leaves) is in test_presence_api.py, which needs the database.

The Redis store's three Lua scripts are not exercised here; they are the same compare-then-act the
in-memory store does in Python, and the contract both must meet is what these tests pin down.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from services.collab.presence import (
    HEARTBEAT_S,
    LOCK_TTL_S,
    PRESENCE_TTL_S,
    TOKEN_RE,
    Identity,
    MemoryPresenceStore,
    PresenceService,
    TakeOverDenied,
    assert_can_edit,
    parse_lock,
)

FRAME = "11111111-1111-1111-1111-111111111111"
OTHER_FRAME = "22222222-2222-2222-2222-222222222222"

ASHA = Identity("u-asha", "Asha", "annotator")
BINU = Identity("u-binu", "Binu", "annotator")
RAVI = Identity("u-ravi", "Ravi", "reviewer")

TAB_A = "tab-asha-aaaaaaaaaaaa"
TAB_A2 = "tab-asha-second-aaaa"
TAB_B = "tab-binu-bbbbbbbbbbbb"
TAB_R = "tab-ravi-rrrrrrrrrrrr"


class Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def svc(clock: Clock) -> PresenceService:
    return PresenceService(MemoryPresenceStore(clock), clock=clock)


def names(state: dict) -> list[str]:
    return [p["name"] for p in state["others"]]


# ---- presence ----------------------------------------------------------------------------------------------


async def test_a_heartbeat_lists_everyone_else_on_the_frame(svc):
    await svc.heartbeat(FRAME, ASHA, TAB_A)
    await svc.heartbeat(FRAME, RAVI, TAB_R)
    seen = await svc.heartbeat(FRAME, BINU, TAB_B)
    assert names(seen) == ["Asha", "Ravi"]
    assert seen["degraded"] is False
    assert seen["heartbeat_s"] == HEARTBEAT_S and seen["ttl_s"] == PRESENCE_TTL_S


async def test_you_are_not_one_of_the_others_even_with_two_tabs(svc):
    await svc.heartbeat(FRAME, ASHA, TAB_A)
    seen = await svc.heartbeat(FRAME, ASHA, TAB_A2)
    assert seen["others"] == []
    # And the other person sees one Asha, not one per tab.
    assert names(await svc.heartbeat(FRAME, BINU, TAB_B)) == ["Asha"]


async def test_presence_is_per_frame(svc):
    await svc.heartbeat(FRAME, ASHA, TAB_A)
    assert names(await svc.heartbeat(OTHER_FRAME, BINU, TAB_B)) == []


async def test_someone_who_stops_beating_is_gone_after_the_ttl(svc, clock):
    await svc.heartbeat(FRAME, ASHA, TAB_A)
    clock.advance(PRESENCE_TTL_S - 1)
    assert names(await svc.heartbeat(FRAME, BINU, TAB_B)) == ["Asha"]
    clock.advance(1.5)
    assert names(await svc.heartbeat(FRAME, BINU, TAB_B)) == []


async def test_beating_keeps_you_present_indefinitely(svc, clock):
    for _ in range(10):
        await svc.heartbeat(FRAME, ASHA, TAB_A)
        clock.advance(HEARTBEAT_S)
    assert names(await svc.heartbeat(FRAME, BINU, TAB_B)) == ["Asha"]


async def test_leaving_removes_you_at_once(svc):
    await svc.heartbeat(FRAME, ASHA, TAB_A)
    await svc.release(FRAME, ASHA, TAB_A)
    assert names(await svc.heartbeat(FRAME, BINU, TAB_B)) == []


async def test_the_tab_token_is_never_shown_to_anyone(svc):
    mine = await svc.acquire(FRAME, ASHA, TAB_A)
    theirs = await svc.heartbeat(FRAME, BINU, TAB_B)
    for state in (mine, theirs):
        assert TAB_A not in json.dumps(state)
        assert TAB_B not in json.dumps(state)


# ---- the lock ----------------------------------------------------------------------------------------------


async def test_the_first_to_ask_holds_the_lock_and_the_second_is_told_who(svc):
    a = await svc.acquire(FRAME, ASHA, TAB_A)
    assert a["lock"]["mine"] is True
    assert a["lock"]["holder"]["name"] == "Asha"

    b = await svc.acquire(FRAME, BINU, TAB_B)
    assert b["lock"]["mine"] is False
    assert b["lock"]["holder"] == {"user_id": "u-asha", "name": "Asha", "role": "annotator",
                                   "since_ms": a["lock"]["holder"]["since_ms"], "same_user": False}
    assert b["others"] == [{"user_id": "u-asha", "name": "Asha", "role": "annotator", "editing": True}]


async def test_a_plain_heartbeat_does_not_take_a_free_lock(svc):
    state = await svc.heartbeat(FRAME, ASHA, TAB_A)
    assert state["lock"] == {"mine": False, "holder": None}
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A, acquire=True))["lock"]["mine"] is True


async def test_a_race_for_a_free_frame_has_exactly_one_winner(svc):
    people = [(Identity(f"u-{i}", f"P{i}", "annotator"), f"tab-racer-{i:012d}") for i in range(12)]
    states = await asyncio.gather(*(svc.acquire(FRAME, who, tab) for who, tab in people))
    assert sum(1 for s in states if s["lock"]["mine"]) == 1


async def test_two_tabs_of_one_person_do_not_share_the_lock(svc):
    assert (await svc.acquire(FRAME, ASHA, TAB_A))["lock"]["mine"] is True
    second = await svc.acquire(FRAME, ASHA, TAB_A2)
    assert second["lock"]["mine"] is False
    assert second["lock"]["holder"]["same_user"] is True
    # The second tab cannot keep the first tab's lock alive, nor give it away.
    assert (await svc.release(FRAME, ASHA, TAB_A2, leave=False))["released"] is False
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A))["lock"]["mine"] is True


async def test_only_the_holder_can_release(svc):
    await svc.acquire(FRAME, ASHA, TAB_A)
    assert (await svc.release(FRAME, BINU, TAB_B))["released"] is False
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A))["lock"]["mine"] is True
    assert (await svc.release(FRAME, ASHA, TAB_A))["released"] is True
    assert (await svc.acquire(FRAME, BINU, TAB_B))["lock"]["mine"] is True


async def test_the_heartbeat_keeps_the_lock_alive(svc, clock):
    await svc.acquire(FRAME, ASHA, TAB_A)
    for _ in range(8):
        clock.advance(HEARTBEAT_S)
        assert (await svc.heartbeat(FRAME, ASHA, TAB_A))["lock"]["mine"] is True
    assert (await svc.acquire(FRAME, BINU, TAB_B))["lock"]["mine"] is False


async def test_somebody_elses_heartbeat_does_not_keep_it_alive(svc, clock):
    await svc.acquire(FRAME, ASHA, TAB_A)
    for _ in range(4):
        clock.advance(HEARTBEAT_S)
        await svc.heartbeat(FRAME, BINU, TAB_B)
    # Asha has not beaten for 20 s, however often Binu has.
    assert (await svc.heartbeat(FRAME, BINU, TAB_B))["lock"]["holder"] is None


async def test_the_lock_expires_by_itself_when_the_holder_disappears(svc, clock):
    await svc.acquire(FRAME, ASHA, TAB_A)
    clock.advance(LOCK_TTL_S - 0.5)
    assert (await svc.acquire(FRAME, BINU, TAB_B))["lock"]["mine"] is False
    clock.advance(1.0)
    taken = await svc.acquire(FRAME, BINU, TAB_B)
    assert taken["lock"]["mine"] is True and taken["lock"]["holder"]["name"] == "Binu"


async def test_an_editor_back_from_the_background_gets_a_lapsed_lock_back_if_nobody_took_it(svc, clock):
    first = await svc.acquire(FRAME, ASHA, TAB_A)
    clock.advance(LOCK_TTL_S * 4)
    back = await svc.heartbeat(FRAME, ASHA, TAB_A, acquire=True)
    assert back["lock"]["mine"] is True
    assert back["lock"]["holder"]["since_ms"] > first["lock"]["holder"]["since_ms"]


async def test_an_editor_back_from_the_background_finds_it_taken(svc, clock):
    await svc.acquire(FRAME, ASHA, TAB_A)
    clock.advance(LOCK_TTL_S * 2)
    await svc.acquire(FRAME, BINU, TAB_B)
    back = await svc.heartbeat(FRAME, ASHA, TAB_A, acquire=True)
    assert back["lock"]["mine"] is False and back["lock"]["holder"]["name"] == "Binu"


async def test_refreshing_does_not_move_when_the_lock_was_taken(svc, clock):
    since = (await svc.acquire(FRAME, ASHA, TAB_A))["lock"]["holder"]["since_ms"]
    clock.advance(HEARTBEAT_S)
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A))["lock"]["holder"]["since_ms"] == since


# ---- taking over ---------------------------------------------------------------------------------------------


async def test_a_reviewer_takes_the_frame_and_the_displaced_holder_is_named(svc):
    await svc.acquire(FRAME, ASHA, TAB_A)
    state, displaced = await svc.take_over(FRAME, RAVI, TAB_R, privileged=True)
    assert state["lock"]["mine"] is True and state["lock"]["holder"]["name"] == "Ravi"
    assert displaced is not None and (displaced.user_id, displaced.name) == ("u-asha", "Asha")

    # The editor who was displaced finds out on their next heartbeat, and cannot take it back by beating.
    lost = await svc.heartbeat(FRAME, ASHA, TAB_A, acquire=True)
    assert lost["lock"]["mine"] is False and lost["lock"]["holder"]["name"] == "Ravi"
    assert lost["can_take_over"] is False


async def test_an_annotator_cannot_take_a_frame_from_someone_else(svc):
    await svc.acquire(FRAME, ASHA, TAB_A)
    with pytest.raises(TakeOverDenied) as denied:
        await svc.take_over(FRAME, BINU, TAB_B, privileged=False)
    assert denied.value.holder.name == "Asha"
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A))["lock"]["mine"] is True


async def test_anyone_can_take_the_frame_from_their_own_other_tab(svc):
    await svc.acquire(FRAME, ASHA, TAB_A)
    state, displaced = await svc.take_over(FRAME, ASHA, TAB_A2, privileged=False)
    assert state["lock"]["mine"] is True
    assert displaced is not None and displaced.token == TAB_A
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A))["lock"]["mine"] is False


async def test_taking_over_a_free_frame_displaces_nobody(svc):
    state, displaced = await svc.take_over(FRAME, BINU, TAB_B, privileged=False)
    assert state["lock"]["mine"] is True and displaced is None


async def test_taking_over_what_you_already_hold_displaces_nobody(svc):
    await svc.acquire(FRAME, RAVI, TAB_R)
    state, displaced = await svc.take_over(FRAME, RAVI, TAB_R, privileged=True)
    assert state["lock"]["mine"] is True and displaced is None


async def test_the_take_over_button_is_offered_to_those_who_may_use_it(svc):
    await svc.acquire(FRAME, ASHA, TAB_A)
    assert (await svc.heartbeat(FRAME, BINU, TAB_B, privileged=False))["can_take_over"] is False
    assert (await svc.heartbeat(FRAME, RAVI, TAB_R, privileged=True))["can_take_over"] is True
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A2, privileged=False))["can_take_over"] is True
    # Never to the holder, and never on a free frame.
    assert (await svc.heartbeat(FRAME, ASHA, TAB_A, privileged=True))["can_take_over"] is False
    assert (await svc.heartbeat(OTHER_FRAME, RAVI, TAB_R, privileged=True))["can_take_over"] is False


# ---- who may save --------------------------------------------------------------------------------------------


async def test_the_holder_may_edit_and_nobody_else(svc):
    await svc.acquire(FRAME, ASHA, TAB_A)
    assert await svc.blocking_holder(FRAME, "u-asha", TAB_A) is None
    blocked = await svc.blocking_holder(FRAME, "u-binu", TAB_B)
    assert blocked is not None and blocked.name == "Asha"
    # The same person's second tab is somebody else as far as the lock is concerned.
    assert await svc.blocking_holder(FRAME, "u-asha", TAB_A2) is not None


async def test_without_a_tab_token_the_holder_is_compared_by_user(svc):
    await svc.acquire(FRAME, ASHA, TAB_A)
    assert await svc.blocking_holder(FRAME, "u-asha") is None
    assert await svc.blocking_holder(FRAME, "u-binu") is not None


async def test_an_unlocked_frame_blocks_nobody(svc, clock):
    assert await svc.blocking_holder(FRAME, "u-binu", TAB_B) is None
    await svc.acquire(FRAME, ASHA, TAB_A)
    clock.advance(LOCK_TTL_S + 1)
    assert await svc.blocking_holder(FRAME, "u-binu", TAB_B) is None


async def test_assert_can_edit_refuses_with_423_and_names_the_holder(svc, monkeypatch):
    from fastapi import HTTPException

    import services.collab.presence as mod

    monkeypatch.setattr(mod, "_service", svc)

    class User:
        def __init__(self, who: Identity) -> None:
            self.user_id, self.name, self.role = who.user_id, who.name, who.role

    await svc.acquire(FRAME, ASHA, TAB_A)
    await assert_can_edit(FRAME, User(ASHA), TAB_A)
    await assert_can_edit(FRAME, User(ASHA))                 # no token: same user passes
    await assert_can_edit(FRAME, None, None)                 # auth off, nobody to compare
    with pytest.raises(HTTPException) as refused:
        await assert_can_edit(FRAME, User(BINU), TAB_B)
    assert refused.value.status_code == 423
    assert "Asha" in refused.value.detail
    with pytest.raises(HTTPException):
        await assert_can_edit(FRAME, User(ASHA), TAB_A2)     # the same person's other tab


# ---- failing open --------------------------------------------------------------------------------------------


class BrokenStore:
    """Every operation raises, which is what a Redis that is down looks like from here."""

    def __getattr__(self, name: str):
        async def boom(*_a, **_k):
            raise ConnectionError("redis is down")

        return boom


async def test_a_dead_store_degrades_instead_of_raising():
    svc = PresenceService(BrokenStore())
    for state in (await svc.heartbeat(FRAME, ASHA, TAB_A, acquire=True), await svc.acquire(FRAME, ASHA, TAB_A)):
        assert state["degraded"] is True
        assert state["others"] == []
        assert state["lock"] == {"mine": False, "holder": None}
    assert await svc.release(FRAME, ASHA, TAB_A) == {"frame_id": FRAME, "released": False, "degraded": True}
    state, displaced = await svc.take_over(FRAME, RAVI, TAB_R, privileged=True)
    assert state["degraded"] is True and displaced is None
    # And a save is not refused because presence is down.
    assert await svc.blocking_holder(FRAME, "u-binu", TAB_B) is None


async def test_a_store_that_cannot_even_be_built_degrades_too():
    def no_client():
        raise RuntimeError("no redis module, no settings, no loop")

    state = await PresenceService(no_client).heartbeat(FRAME, ASHA, TAB_A)
    assert state["degraded"] is True


# ---- the stored value ----------------------------------------------------------------------------------------


def test_an_unreadable_lock_value_is_no_lock():
    assert parse_lock(None) is None
    assert parse_lock("") is None
    assert parse_lock("no-separator") is None
    assert parse_lock("tok|not json") is None
    assert parse_lock('tok|{"u": "x"}') is None


def test_tab_tokens_have_a_closed_alphabet():
    assert TOKEN_RE.match(TAB_A)
    for bad in ("short", "has|pipe-aaaaaaaaaaaaaaaa", "has space aaaaaaaaaaaaaaa", "x" * 65, ""):
        assert not TOKEN_RE.match(bad), bad


async def test_the_memory_store_swap_is_conditional_on_the_expected_holder(clock):
    store = MemoryPresenceStore(clock)
    assert await store.set_lock_if_free(FRAME, f"{TAB_A}|a", 1000)
    assert await store.swap_lock(FRAME, f"{TAB_B}|b", 1000, only_if_token=TAB_R) == (False, f"{TAB_A}|a")
    assert await store.swap_lock(FRAME, f"{TAB_B}|b", 1000, only_if_token=TAB_A) == (True, f"{TAB_A}|a")
    assert await store.get_lock(FRAME) == f"{TAB_B}|b"
    # A token that is a prefix of the holder's is not the holder.
    assert await store.release_lock(FRAME, TAB_B[:-1]) is False
    assert await store.refresh_lock(FRAME, TAB_B[:-1], 1000) is False
