"""Members-only projects: who sees what, end to end over HTTP.

An administrator makes a project visible to its members only, moves a session into it and adds one annotator.
That annotator sees the session; another annotator does not, neither in the listings nor by id (404, so the
response does not confirm the item exists); the administrator sees everything. A project left at the default
"team" visibility changes nothing for anyone.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


def _client():
    from services.api.main import app

    return TestClient(app)


def _seed_session() -> tuple[str, str]:
    """A session with one frame, in the default project when there is one. Returns (session_id, frame_id)."""
    from _authutil import _clear_db_cache

    async def _seed():
        from db.models import Frame
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.api.tenancy import default_project_id
        from services.autolabel.ontology import get_ontology

        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id=f"PRJ-{sid.hex[:6]}", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(2), city="BLR", sensors={}, ontology_version=get_ontology().version,
                             project_id=await default_project_id(db)))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/0.jpg",
                         width=1920, height=1080, quality=0.9, scene={}))
            await db.commit()
        return str(sid), str(fid)

    _clear_db_cache()
    try:
        return asyncio.run(_seed())
    finally:
        _clear_db_cache()


@pytest.fixture
def restricted():
    """A members-only project holding one session, one member annotator and one outsider annotator."""
    from _authutil import headers_for, seed_user

    sid, fid = _seed_session()
    admin = headers_for(seed_user("admin"))
    member_id = seed_user("annotator")
    member, outsider = headers_for(member_id), headers_for(seed_user("annotator"))
    with _client() as c:
        r = c.post("/api/projects", json={"name": f"restricted-{uuid.uuid4().hex[:8]}", "visibility": "members"},
                   headers=admin)
        assert r.status_code == 200, r.text
        pid = r.json()["project_id"]
        r = c.post(f"/api/projects/{pid}/sessions", json={"session_ids": [sid]}, headers=admin)
        assert r.status_code == 200 and r.json()["moved"] == 1, r.text
        assert c.put(f"/api/projects/{pid}/members/{member_id}", headers=admin).status_code == 200
    return {"pid": pid, "sid": sid, "fid": fid, "admin": admin, "member": member, "outsider": outsider,
            "member_id": member_id}


def _listed(c: TestClient, headers: dict, sid: str) -> bool:
    r = c.get("/api/sessions/page", params={"limit": 500}, headers=headers)
    assert r.status_code == 200, r.text
    page = {s["session_id"] for s in r.json()["sessions"]}
    r = c.get("/api/sessions", params={"limit": 1000}, headers=headers)
    assert r.status_code == 200, r.text
    flat = {s["session_id"] for s in r.json()}
    assert (sid in page) == (sid in flat), "the two session listings disagree"
    return sid in page


def test_an_outsider_cannot_reach_a_members_only_session(restricted):
    with _client() as c:
        h = restricted["outsider"]
        assert not _listed(c, h, restricted["sid"])
        assert c.get(f"/api/sessions/{restricted['sid']}/stats", headers=h).status_code == 404
        assert c.get(f"/api/frames/{restricted['fid']}", headers=h).status_code == 404
        # A query parameter names the item just as a path segment does.
        assert c.get("/api/analytics/overview", params={"session_id": restricted["sid"]},
                     headers=h).status_code == 404
        # The project itself is not listed either.
        names = {p["project_id"] for p in c.get("/api/projects", headers=h).json()["projects"]}
        assert restricted["pid"] not in names


def test_a_member_and_an_administrator_can(restricted):
    with _client() as c:
        for who in ("member", "admin"):
            h = restricted[who]
            assert _listed(c, h, restricted["sid"]), who
            assert c.get(f"/api/sessions/{restricted['sid']}/stats", headers=h).status_code == 200, who
            assert c.get(f"/api/frames/{restricted['fid']}", headers=h).status_code != 404, who
        mine = {p["project_id"]: p for p in c.get("/api/projects", headers=restricted["member"]).json()["projects"]}
        assert mine[restricted["pid"]]["is_member"] and mine[restricted["pid"]]["sessions"] == 1


def test_find_and_home_leave_it_out(restricted):
    with _client() as c:
        q = f"PRJ-{uuid.UUID(restricted['sid']).hex[:6]}"
        hits = lambda h: {x["id"] for x in c.get("/api/find", params={"q": q}, headers=h).json()["results"]}  # noqa: E731
        assert restricted["sid"] in hits(restricted["member"])
        assert restricted["sid"] not in hits(restricted["outsider"])

        seen_by_member = c.get("/api/home", headers=restricted["member"]).json()["corpus"]["sessions"]
        seen_by_outsider = c.get("/api/home", headers=restricted["outsider"]).json()["corpus"]["sessions"]
        assert seen_by_member > seen_by_outsider


def test_removing_a_member_takes_the_access_away(restricted):
    with _client() as c:
        r = c.delete(f"/api/projects/{restricted['pid']}/members/{restricted['member_id']}",
                     headers=restricted["admin"])
        assert r.status_code == 200 and r.json()["member"] is False
        assert c.get(f"/api/sessions/{restricted['sid']}/stats", headers=restricted["member"]).status_code == 404


def test_opening_the_project_to_the_team_gives_everyone_access(restricted):
    with _client() as c:
        r = c.patch(f"/api/projects/{restricted['pid']}", json={"visibility": "team"}, headers=restricted["admin"])
        assert r.status_code == 200 and r.json()["visibility"] == "team"
        assert c.get(f"/api/sessions/{restricted['sid']}/stats", headers=restricted["outsider"]).status_code == 200
        assert _listed(c, restricted["outsider"], restricted["sid"])


def test_only_an_administrator_manages_projects(restricted):
    with _client() as c:
        h = restricted["member"]
        assert c.post("/api/projects", json={"name": f"x-{uuid.uuid4().hex[:6]}"}, headers=h).status_code == 403
        assert c.patch(f"/api/projects/{restricted['pid']}", json={"visibility": "team"},
                       headers=h).status_code == 403
        assert c.get(f"/api/projects/{restricted['pid']}/members", headers=h).status_code == 403
        assert c.post(f"/api/projects/{restricted['pid']}/sessions", json={"session_ids": [restricted["sid"]]},
                      headers=h).status_code == 403


def test_bad_input_is_refused(restricted):
    with _client() as c:
        h = restricted["admin"]
        assert c.post("/api/projects", json={"name": "y", "visibility": "secret"}, headers=h).status_code == 400
        name = c.get("/api/projects", headers=h).json()["projects"][0]["name"]
        assert c.post("/api/projects", json={"name": name}, headers=h).status_code == 409
        assert c.patch("/api/projects/not-a-uuid", json={"visibility": "team"}, headers=h).status_code == 400
        assert c.patch(f"/api/projects/{uuid.uuid4()}", json={"visibility": "team"}, headers=h).status_code == 404


def test_a_team_project_hides_nothing():
    """The default: with every project visible to the team, an annotator's session listing is the admin's."""
    from _authutil import headers_for, seed_user

    sid, _ = _seed_session()
    with _client() as c:
        assert _listed(c, headers_for(seed_user("annotator")), sid)


def test_a_listing_that_does_not_scope_itself_still_leaves_hidden_rows_out(restricted):
    """The net under every JSON response (services/api/project_filter.py), on a route that knows nothing of
    projects: rows naming a hidden session or frame are dropped, wherever they sit, and the rest pass."""
    from fastapi import APIRouter, FastAPI

    from services.api.project_filter import filter_hidden_rows
    from services.api.project_guard import PROJECT_GUARD

    visible_sid, visible_fid = _seed_session()
    router = APIRouter()

    @router.get("/rows")
    async def rows():
        return {"frames": [{"frame_id": restricted["fid"]}, {"frame_id": visible_fid}],
                "groups": [{"items": [{"session_id": restricted["sid"], "n": 1}, {"session_id": visible_sid, "n": 2}]}],
                "unrelated": [{"frame_id": "not-a-uuid"}, 3]}

    app = FastAPI()
    app.include_router(router, prefix="/api", dependencies=PROJECT_GUARD)
    app.middleware("http")(filter_hidden_rows)
    with TestClient(app) as c:
        out = c.get("/api/rows", headers=restricted["outsider"]).json()
        assert out["frames"] == [{"frame_id": visible_fid}]
        assert out["groups"][0]["items"] == [{"session_id": visible_sid, "n": 2}]
        assert out["unrelated"] == [{"frame_id": "not-a-uuid"}, 3]

        whole = c.get("/api/rows", headers=restricted["member"]).json()
        assert len(whole["frames"]) == 2 and len(whole["groups"][0]["items"]) == 2


def test_analytics_count_only_what_the_caller_may_see(restricted):
    """Aggregates name no item, but before this they counted members-only projects for everyone, so the
    numbers still said what was in them."""
    with _client() as c:
        out = c.get("/api/analytics/overview", headers=restricted["outsider"]).json()
        member = c.get("/api/analytics/overview", headers=restricted["member"]).json()
        admin = c.get("/api/analytics/overview", headers=restricted["admin"]).json()
        # The member and the outsider differ by exactly this project's one session and frame. An
        # administrator sees at least as much (earlier tests leave other members-only projects behind).
        assert member["sessions"] == out["sessions"] + 1 and member["frames"] == out["frames"] + 1
        assert admin["sessions"] >= member["sessions"]
        pii_out = c.get("/api/analytics/pii", headers=restricted["outsider"]).json()
        pii_member = c.get("/api/analytics/pii", headers=restricted["member"]).json()
        assert pii_member["total_frames"] == pii_out["total_frames"] + 1
