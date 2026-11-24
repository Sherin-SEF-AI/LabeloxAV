"""Presence and the frame edit lock over HTTP, with auth on.

What only the mounted app can show: that identity comes from the token and from nothing the client sends,
that taking a frame over is role-gated on the server and leaves an activity record naming who was
displaced, that a frame in a members-only project answers 404 here as it does everywhere else, and that a
dead presence store is a 200 with `degraded`, never a 500.

The store is swapped for the in-memory one, so these do not depend on Redis being up and two runs cannot
see each other's locks. The Redis store itself is covered in test_presence_redis.py and the logic in
test_presence_logic.py; neither needs the database.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from core.config import get_settings
from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db

TAB_A = "tab-first-aaaaaaaaaaaa"
TAB_B = "tab-second-bbbbbbbbbbb"
TAB_C = "tab-third-cccccccccccc"


def _client() -> TestClient:
    from services.api.main import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def auth_on():
    s = get_settings()
    prev = s.auth.enabled
    s.auth.enabled = True
    yield
    s.auth.enabled = prev


@pytest.fixture(autouse=True)
def memory_presence(monkeypatch):
    """One in-memory presence service per test, in place of the Redis-backed one."""
    import services.collab.presence as mod

    svc = mod.PresenceService(mod.MemoryPresenceStore())
    monkeypatch.setattr(mod, "_service", svc)
    return svc


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _user(role: str) -> dict:
    """A seeded user of `role`: its id, its name as the database has it, and a Bearer header."""
    from _authutil import headers_for, seed_user

    uid = seed_user(role)

    async def _name() -> str:
        from db.models import User
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            return (await db.get(User, uuid.UUID(uid))).name

    return {"id": uid, "name": _run(_name()), "headers": headers_for(uid)}


def _url(frame_id: str, what: str) -> str:
    return f"/api/presence/frames/{frame_id}/{what}"


def _frame() -> str:
    # Presence does not require the frame to exist: an id nobody can see answers 404 through the project
    # guard, and an id that names nothing holds nothing.
    return str(uuid.uuid4())


def test_every_route_needs_a_token():
    fid = _frame()
    with _client() as c:
        for what in ("heartbeat", "lock/acquire", "lock/release", "lock/takeover"):
            assert c.post(_url(fid, what), json={"tab": TAB_A}).status_code == 401, what


def test_a_malformed_frame_id_or_tab_token_is_rejected():
    ann = _user("annotator")
    with _client() as c:
        assert c.post(_url("not-a-uuid", "heartbeat"), json={"tab": TAB_A},
                      headers=ann["headers"]).status_code == 422
        for bad in ("short", "has|pipe-aaaaaaaaaaaaaaaa", "x" * 65):
            assert c.post(_url(_frame(), "heartbeat"), json={"tab": bad},
                          headers=ann["headers"]).status_code == 422, bad
        assert c.post(_url(_frame(), "heartbeat"), json={}, headers=ann["headers"]).status_code == 422


def test_the_first_editor_holds_the_lock_and_the_second_is_told_who():
    a, b = _user("annotator"), _user("annotator")
    fid = _frame()
    with _client() as c:
        r = c.post(_url(fid, "lock/acquire"), json={"tab": TAB_A}, headers=a["headers"])
        assert r.status_code == 200, r.text
        mine = r.json()
        assert mine["degraded"] is False
        assert mine["lock"]["mine"] is True
        assert mine["lock"]["holder"]["user_id"] == a["id"]
        assert mine["others"] == []

        r = c.post(_url(fid, "lock/acquire"), json={"tab": TAB_B}, headers=b["headers"])
        assert r.status_code == 200, r.text
        theirs = r.json()
        assert theirs["lock"]["mine"] is False
        assert theirs["lock"]["holder"]["name"] == a["name"]
        assert theirs["lock"]["holder"]["same_user"] is False
        assert theirs["can_take_over"] is False
        assert theirs["others"] == [{"user_id": a["id"], "name": a["name"], "role": "annotator",
                                     "editing": True}]

        # The holder's next heartbeat sees the newcomer, and still holds the lock.
        beat = c.post(_url(fid, "heartbeat"), json={"tab": TAB_A}, headers=a["headers"]).json()
        assert beat["lock"]["mine"] is True
        assert [p["user_id"] for p in beat["others"]] == [b["id"]]
        # No tab token ever comes back.
        assert TAB_A not in r.text and TAB_B not in r.text


def test_identity_comes_from_the_token_not_from_the_body():
    a, b = _user("annotator"), _user("annotator")
    fid = _frame()
    with _client() as c:
        # Everything a client might try to assert about itself, alongside the one field that is read.
        forged = {"tab": TAB_A, "name": "Somebody Else", "user_id": b["id"], "role": "admin",
                  "user": {"name": "Somebody Else", "role": "admin"}}
        r = c.post(_url(fid, "lock/acquire"), json=forged, headers=a["headers"])
        assert r.status_code == 200, r.text
        holder = r.json()["lock"]["holder"]
        assert (holder["user_id"], holder["name"], holder["role"]) == (a["id"], a["name"], "annotator")
        # And a forged role in the body does not buy a take-over.
        forged_b = {"tab": TAB_B, "role": "admin", "name": "Admin"}
        assert c.post(_url(fid, "lock/takeover"), json=forged_b, headers=b["headers"]).status_code == 403


def test_only_the_holding_tab_can_release():
    a, b = _user("annotator"), _user("annotator")
    fid = _frame()
    with _client() as c:
        c.post(_url(fid, "lock/acquire"), json={"tab": TAB_A}, headers=a["headers"])
        # Another person, and the same person from another tab: neither releases it.
        assert c.post(_url(fid, "lock/release"), json={"tab": TAB_B},
                      headers=b["headers"]).json()["released"] is False
        assert c.post(_url(fid, "lock/release"), json={"tab": TAB_C},
                      headers=a["headers"]).json()["released"] is False
        # The holder's own heartbeat still finds the lock where it left it.
        assert c.post(_url(fid, "heartbeat"), json={"tab": TAB_A},
                      headers=a["headers"]).json()["lock"]["mine"] is True

        assert c.post(_url(fid, "lock/release"), json={"tab": TAB_A},
                      headers=a["headers"]).json()["released"] is True
        taken = c.post(_url(fid, "lock/acquire"), json={"tab": TAB_B}, headers=b["headers"]).json()
        assert taken["lock"]["mine"] is True


def test_an_annotator_cannot_take_over_and_the_lock_is_untouched():
    a, b = _user("annotator"), _user("annotator")
    fid = _frame()
    with _client() as c:
        c.post(_url(fid, "lock/acquire"), json={"tab": TAB_A}, headers=a["headers"])
        r = c.post(_url(fid, "lock/takeover"), json={"tab": TAB_B}, headers=b["headers"])
        assert r.status_code == 403, r.text
        assert c.post(_url(fid, "heartbeat"), json={"tab": TAB_A},
                      headers=a["headers"]).json()["lock"]["mine"] is True


def _take_over_events(frame_id: str) -> list[dict]:
    async def _read():
        from sqlalchemy import select

        from db.models import ActivityEvent
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            rows = (await db.execute(select(ActivityEvent).where(
                ActivityEvent.verb == "took_over_frame", ActivityEvent.subject_id == frame_id))).scalars().all()
            return [{"user_id": str(r.user_id), "user_name": r.user_name, "subject_type": r.subject_type,
                     "summary": r.summary, "meta": r.meta} for r in rows]

    return _run(_read())


@pytest.mark.parametrize("role", ["reviewer", "admin"])
def test_a_reviewer_or_admin_takes_over_and_it_is_recorded(role):
    a, boss = _user("annotator"), _user(role)
    fid = _frame()
    with _client() as c:
        c.post(_url(fid, "lock/acquire"), json={"tab": TAB_A}, headers=a["headers"])
        seen = c.post(_url(fid, "heartbeat"), json={"tab": TAB_B}, headers=boss["headers"]).json()
        assert seen["can_take_over"] is True

        r = c.post(_url(fid, "lock/takeover"), json={"tab": TAB_B}, headers=boss["headers"])
        assert r.status_code == 200, r.text
        assert r.json()["lock"]["mine"] is True
        assert r.json()["lock"]["holder"]["user_id"] == boss["id"]

        # The displaced editor learns it on the next heartbeat, even one that asks for the lock back.
        lost = c.post(_url(fid, "heartbeat"), json={"tab": TAB_A, "acquire": True},
                      headers=a["headers"]).json()
        assert lost["lock"]["mine"] is False
        assert lost["lock"]["holder"]["user_id"] == boss["id"]

    events = _take_over_events(fid)
    assert len(events) == 1, events
    assert events[0]["user_id"] == boss["id"]
    assert events[0]["subject_type"] == "frame"
    assert a["name"] in events[0]["summary"]
    assert events[0]["meta"]["previous_holder_id"] == a["id"]


def test_taking_over_a_free_frame_or_your_own_tab_records_nothing():
    rev = _user("reviewer")
    fid = _frame()
    with _client() as c:
        assert c.post(_url(fid, "lock/takeover"), json={"tab": TAB_A},
                      headers=rev["headers"]).json()["lock"]["mine"] is True
        # The same person, from a second tab.
        assert c.post(_url(fid, "lock/takeover"), json={"tab": TAB_B},
                      headers=rev["headers"]).json()["lock"]["mine"] is True
    assert _take_over_events(fid) == []


def test_an_annotator_can_take_over_their_own_other_tab():
    a = _user("annotator")
    fid = _frame()
    with _client() as c:
        c.post(_url(fid, "lock/acquire"), json={"tab": TAB_A}, headers=a["headers"])
        second = c.post(_url(fid, "lock/acquire"), json={"tab": TAB_B}, headers=a["headers"]).json()
        assert second["lock"]["mine"] is False
        assert second["lock"]["holder"]["same_user"] is True and second["can_take_over"] is True
        r = c.post(_url(fid, "lock/takeover"), json={"tab": TAB_B}, headers=a["headers"])
        assert r.status_code == 200 and r.json()["lock"]["mine"] is True


def test_a_dead_presence_store_is_a_200_that_says_degraded(monkeypatch):
    import services.collab.presence as mod

    def no_redis():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(mod, "_service", mod.PresenceService(no_redis))
    rev = _user("reviewer")
    fid = _frame()
    with _client() as c:
        for what in ("heartbeat", "lock/acquire", "lock/takeover"):
            r = c.post(_url(fid, what), json={"tab": TAB_A}, headers=rev["headers"])
            assert r.status_code == 200, (what, r.text)
            body = r.json()
            assert body["degraded"] is True and body["others"] == [], what
            assert body["lock"] == {"mine": False, "holder": None}, what
        r = c.post(_url(fid, "lock/release"), json={"tab": TAB_A}, headers=rev["headers"])
        assert r.status_code == 200 and r.json() == {"frame_id": fid, "released": False, "degraded": True}
    assert _take_over_events(fid) == []


# ---- the project guard ---------------------------------------------------------------------------------------


def _seed_session() -> tuple[str, str]:
    """A session with one frame, in the default project. Returns (session_id, frame_id)."""

    async def _seed():
        from db.models import Frame
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.api.tenancy import default_project_id
        from services.autolabel.ontology import get_ontology

        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id=f"PRS-{sid.hex[:6]}", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(2), city="BLR", sensors={},
                             ontology_version=get_ontology().version,
                             project_id=await default_project_id(db)))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/0.jpg",
                         width=1920, height=1080, quality=0.9, scene={}))
            await db.commit()
        return str(sid), str(fid)

    return _run(_seed())


def test_presence_on_a_members_only_frame_is_404_to_an_outsider(memory_presence):
    sid, fid = _seed_session()
    admin, member, outsider = _user("admin"), _user("annotator"), _user("annotator")
    with _client() as c:
        r = c.post("/api/projects", json={"name": f"presence-{uuid.uuid4().hex[:8]}", "visibility": "members"},
                   headers=admin["headers"])
        assert r.status_code == 200, r.text
        pid = r.json()["project_id"]
        r = c.post(f"/api/projects/{pid}/sessions", json={"session_ids": [sid]}, headers=admin["headers"])
        assert r.status_code == 200 and r.json()["moved"] == 1, r.text
        assert c.put(f"/api/projects/{pid}/members/{member['id']}", headers=admin["headers"]).status_code == 200

        # The member is on the frame and holds it.
        r = c.post(_url(fid, "lock/acquire"), json={"tab": TAB_A}, headers=member["headers"])
        assert r.status_code == 200 and r.json()["lock"]["mine"] is True, r.text

        # The outsider can learn nothing about it and change nothing on it, on any route.
        for what in ("heartbeat", "lock/acquire", "lock/release", "lock/takeover"):
            r = c.post(_url(fid, what), json={"tab": TAB_B}, headers=outsider["headers"])
            assert r.status_code == 404, (what, r.text)
            assert member["name"] not in r.text

        # Nothing the outsider sent registered: they are not present, and the member still holds the lock.
        beat = c.post(_url(fid, "heartbeat"), json={"tab": TAB_A}, headers=member["headers"]).json()
        assert beat["lock"]["mine"] is True and beat["others"] == []

        # An administrator sees every project, and so sees who is here.
        seen = c.post(_url(fid, "heartbeat"), json={"tab": TAB_C}, headers=admin["headers"]).json()
        assert [p["user_id"] for p in seen["others"]] == [member["id"]]
        assert seen["lock"]["holder"]["user_id"] == member["id"]
