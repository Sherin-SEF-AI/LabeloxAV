"""An annotator can fix their own work, and only their own.

Every route under /api/objects sat behind a reviewer floor, so an annotator could draw a box and then
could not move, relabel or delete it: a box drawn in the wrong place stayed wrong until a reviewer got to
it. The floor is lifted for the routes that edit one object, and services/object_access.py decides per
object. These pin both halves over real requests with real tokens: what an annotator now can do, and
what they still cannot (another person's box, anything a reviewer has ruled on, anything accepted).
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from core.config import get_settings
from core.timebase import now_ns
from services.object_access import FINAL, NOT_YOURS, REVIEWED, is_restricted, refusal

# The second half makes real requests against the database.
pytestmark = pytest.mark.db

# ---------------------------------------------------------------------------------- the rule, on its own


def _user(role="annotator", uid=None, name="asha"):
    return SimpleNamespace(role=role, user_id=uid or uuid.uuid4(), name=name)


def _obj(state="review", annotator_id=None):
    return SimpleNamespace(state=state, annotator_id=annotator_id, object_id=uuid.uuid4())


def test_reviewers_admins_and_installs_without_roles_are_not_restricted():
    assert not is_restricted(_user("reviewer")) and not is_restricted(_user("admin")) and not is_restricted(None)
    assert is_restricted(_user("annotator"))
    assert refusal(_obj("accepted", uuid.uuid4()), _user("reviewer"), True) is None
    assert refusal(_obj("accepted", uuid.uuid4()), None, True) is None


def test_an_annotator_may_change_their_own_box_and_an_unreviewed_machine_label():
    me = _user()
    assert refusal(_obj("review", me.user_id), me, False) is None
    assert refusal(_obj("submitted", me.user_id), me, False) is None      # their own submission, not yet ruled on
    assert refusal(_obj("review", None), me, False) is None               # machine-made, nobody has touched it
    assert refusal(_obj("auto_accept", None), me, False) is None          # the gate's opinion is not a review


def test_an_annotator_may_not_change_someone_elses_box():
    assert refusal(_obj("review", uuid.uuid4()), _user(), False) == NOT_YOURS


def test_an_annotator_may_not_change_what_someone_else_has_ruled_on():
    me = _user()
    assert refusal(_obj("review", None), me, True) == REVIEWED
    assert refusal(_obj("rejected", me.user_id), me, True) == REVIEWED    # even their own box, once reviewed


@pytest.mark.parametrize("state", ["accepted", "settled"])
def test_an_annotator_may_not_change_a_label_that_already_counts(state):
    me = _user()
    assert refusal(_obj(state, me.user_id), me, False) == FINAL


def test_the_role_floor_lets_an_annotator_reach_exactly_the_object_edit_routes():
    from services.api.main import _required_role

    oid = "0b6f9c1e-3d2a-4c5b-8e7f-123456789abc"
    for method, path in (("POST", f"/api/objects/{oid}/review"), ("POST", "/api/objects/bulk-review"),
                         ("PUT", f"/api/objects/{oid}/mask"), ("DELETE", f"/api/objects/{oid}"),
                         ("GET", f"/api/objects/{oid}"), ("GET", f"/api/objects/{oid}/explain"),
                         ("GET", f"/api/objects/{oid}/crop")):
        assert _required_role(path, method) == "annotator", (method, path)
    # Everything else under the prefix is still a reviewer's: a route added later is not opened by default.
    for method, path in (("POST", f"/api/objects/{oid}/propagate"), ("POST", f"/api/objects/{oid}/sam_propagate"),
                         ("POST", "/api/objects/crops"), ("POST", f"/api/objects/{oid}/relate"),
                         ("DELETE", f"/api/objects/{oid}/mask"), ("PUT", f"/api/objects/{oid}"),
                         ("POST", f"/api/objects/{oid}/review/extra"), ("GET", "/api/objects/not-a-uuid")):
        assert _required_role(path, method) == "reviewer", (method, path)


# ----------------------------------------------------------------------------------------- over real requests



def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@pytest.fixture
def auth_on():
    s = get_settings()
    prev = s.auth.enabled
    s.auth.enabled = True
    yield
    s.auth.enabled = prev


def _run(coro):
    from db.session import get_engine, get_sessionmaker

    async def wrapped():
        try:
            return await coro
        finally:
            await get_engine().dispose()

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    try:
        return asyncio.run(wrapped())
    finally:
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()


async def _seed() -> dict:
    """Two annotators and a reviewer, and one frame holding: a box each annotator drew, a machine label
    nobody has touched, a machine label the reviewer has ruled on, and an accepted one."""
    from db.models import Frame, Object, Review, User
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = onto.by_name("sedan").id
    tag = uuid.uuid4().hex[:8]
    asha, bala, rev = (User(name=f"asha-{tag}", role="annotator"), User(name=f"bala-{tag}", role="annotator"),
                       User(name=f"rev-{tag}", role="reviewer"))
    async with get_sessionmaker()() as db:
        db.add_all([asha, bala, rev])
        await db.flush()
        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        db.add(DbSession(session_id=sid, vehicle_id=f"ACC-{tag}", start_ts_ns=ts, end_ts_ns=ts + 1, city="BLR",
                         sensors={}, ontology_version=onto.version))
        await db.flush()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=f"s3://x/{tag}.jpg",
                     width=640, height=480, quality=0.9))
        await db.flush()

        def obj(**kw):
            o = Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid, bbox=[10, 10, 60, 50], conf=0.8,
                       attrs={}, **kw)
            db.add(o)
            return o

        mine = obj(source="human", state="review", annotator_id=asha.user_id)
        theirs = obj(source="human", state="review", annotator_id=bala.user_id)
        machine = obj(source="fused", state="review")
        reviewed = obj(source="human", state="rejected")
        accepted = obj(source="human", state="accepted")
        await db.flush()
        db.add(Review(object_id=reviewed.object_id, reviewer=rev.name, user_id=rev.user_id, action="reject",
                      before={}, after={}, ts_ns=ts))
        await db.commit()
        return {"asha": str(asha.user_id), "bala": str(bala.user_id), "rev": str(rev.user_id),
                "frame": str(fid), "mine": str(mine.object_id), "theirs": str(theirs.object_id),
                "machine": str(machine.object_id), "reviewed": str(reviewed.object_id),
                "accepted": str(accepted.object_id)}


async def _state(object_id: str):
    from db.models import Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        o = await db.get(Object, uuid.UUID(object_id))
        return None if o is None else (o.state, [round(v) for v in o.bbox])


def _bearer(user_id: str) -> dict:
    from services.api.auth_token import mint_token

    return {"Authorization": f"Bearer {mint_token(user_id, get_settings().auth.signing_key)}"}


def _client():
    from fastapi.testclient import TestClient

    from db.session import get_engine, get_sessionmaker
    from services.api.main import app

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    return TestClient(app)


@requires_infra
def test_an_annotator_moves_and_deletes_their_own_box(auth_on):
    s = _run(_seed())
    asha = _bearer(s["asha"])
    with _client() as c:
        r = c.post(f"/api/objects/{s['mine']}/review", headers=asha,
                   json={"action": "adjust", "bbox": [20, 20, 70, 60], "state": "accepted"})
        assert r.status_code == 200, r.text
        # Moved, and the approval was a submission: the reviewer step is intact.
        assert r.json()["bbox"] == [20, 20, 70, 60] and r.json()["state"] == "submitted"
        assert c.get(f"/api/objects/{s['mine']}", headers=asha).status_code == 200
        assert c.delete(f"/api/objects/{s['mine']}", headers=asha).status_code == 200
    assert _run(_state(s["mine"])) is None


@requires_infra
def test_an_annotator_corrects_a_machine_label_nobody_has_reviewed(auth_on):
    s = _run(_seed())
    with _client() as c:
        r = c.post(f"/api/objects/{s['machine']}/review", headers=_bearer(s["asha"]),
                   json={"action": "adjust", "bbox": [5, 5, 40, 40]})
        assert r.status_code == 200, r.text
        # Once one annotator has corrected it, it is theirs: a second annotator does not rework it.
        r2 = c.post(f"/api/objects/{s['machine']}/review", headers=_bearer(s["bala"]),
                    json={"action": "adjust", "bbox": [1, 1, 2, 2]})
        assert r2.status_code == 403 and "reviewed by someone else" in r2.json()["detail"]
    assert _run(_state(s["machine"]))[1] == [5, 5, 40, 40]


@requires_infra
@pytest.mark.parametrize("which,why", [("theirs", "drawn by someone else"),
                                       ("reviewed", "reviewed by someone else"),
                                       ("accepted", "already accepted")])
def test_an_annotator_cannot_change_or_delete_what_is_not_theirs(auth_on, which, why):
    s = _run(_seed())
    asha = _bearer(s["asha"])
    before = _run(_state(s[which]))
    with _client() as c:
        r = c.post(f"/api/objects/{s[which]}/review", headers=asha,
                   json={"action": "adjust", "bbox": [1, 1, 2, 2]})
        assert r.status_code == 403 and why in r.json()["detail"], r.text
        d = c.delete(f"/api/objects/{s[which]}", headers=asha)
        assert d.status_code == 403 and why in d.json()["detail"], d.text
    assert _run(_state(s[which])) == before, "a refused change must change nothing"


@requires_infra
def test_a_bulk_action_that_includes_someone_elses_label_changes_nothing(auth_on):
    s = _run(_seed())
    with _client() as c:
        r = c.post("/api/objects/bulk-review", headers=_bearer(s["asha"]),
                   json={"object_ids": [s["mine"], s["machine"], s["theirs"]], "action": "reject"})
        assert r.status_code == 403 and "Nothing was changed" in r.json()["detail"], r.text
        ok = c.post("/api/objects/bulk-review", headers=_bearer(s["asha"]),
                    json={"object_ids": [s["mine"], s["machine"]], "action": "reject"})
        assert ok.status_code == 200 and ok.json()["updated"] == 2, ok.text
    assert _run(_state(s["mine"]))[0] == "rejected" and _run(_state(s["theirs"]))[0] == "review"


@requires_infra
def test_a_reviewer_is_not_restricted_and_other_object_routes_stay_reviewer_only(auth_on):
    s = _run(_seed())
    with _client() as c:
        for which in ("theirs", "reviewed", "accepted"):
            r = c.post(f"/api/objects/{s[which]}/review", headers=_bearer(s["rev"]),
                       json={"action": "adjust", "bbox": [3, 3, 30, 30]})
            assert r.status_code == 200, (which, r.text)
        # Propagating a label across frames is still a reviewer's action.
        p = c.post(f"/api/objects/{s['mine']}/propagate", headers=_bearer(s["asha"]))
        assert p.status_code == 403
