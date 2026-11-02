"""Which classes a project labels: templates, the stored set, and where it applies.

The ontology stays one list with global ids; a project narrows it. "india" is the whole ontology (NULL, so a
class added later is in it), "generic" drops the Indian long tail, and a custom list is checked against the
ontology. Ignore regions and the team's own classes are in every set.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.db


def _client():
    from services.api.main import app

    return TestClient(app)


def test_the_generic_template_leaves_out_the_indian_long_tail_and_keeps_what_every_project_needs():
    from services.autolabel.ontology import get_ontology
    from services.project_classes import GENERIC, INDIA, template_ids

    onto = get_ontology()
    generic = set(template_ids(GENERIC) or [])
    assert template_ids(INDIA) is None
    assert onto.by_name("sedan").id in generic and onto.by_name("pedestrian").id in generic
    assert onto.by_name("autorickshaw").id not in generic
    for c in onto.classes:
        if c.l0 == "ignore" or c.l1 == "custom":
            assert c.id in generic, c.name
        elif c.india:
            assert c.id not in generic, c.name


def test_a_custom_set_is_checked_against_the_ontology():
    from services.autolabel.ontology import get_ontology
    from services.project_classes import validate

    onto = get_ontology()
    ids = validate([onto.by_name("sedan").id])
    assert onto.by_name("sedan").id in ids
    assert all(i in ids for i in (c.id for c in onto.classes if c.l0 == "ignore"))
    with pytest.raises(ValueError, match="not in the ontology"):
        validate([999_999])
    # Only surfaces chosen: the set still holds the team's own classes, and is refused only if none of those is
    # an object either.
    stuff = [c.id for c in onto.classes if not onto.is_thing(c.id) and c.l0 != "ignore"][:3]
    if any(onto.is_thing(c.id) for c in onto.classes if c.l1 == "custom"):
        assert set(stuff) <= set(validate(stuff))
    else:
        with pytest.raises(ValueError, match="at least one object class"):
            validate(stuff)


@dataclass
class _Obj:
    class_id: int


@dataclass
class _Fused:
    obj: _Obj


def test_auto_labelling_writes_only_the_projects_classes():
    from services.project_classes import keep_in_set

    fused = [_Fused(_Obj(11)), _Fused(_Obj(6)), _Fused(_Obj(11))]
    assert keep_in_set(fused, None) == (fused, 0)
    kept, dropped = keep_in_set(fused, frozenset({11}))
    assert [g.obj.class_id for g in kept] == [11, 11] and dropped == 1


def _project_with_frame(c: TestClient, admin: dict) -> tuple[str, str]:
    """A project holding one session with one frame. Returns (project_id, frame_id)."""
    import asyncio

    from _authutil import _clear_db_cache

    from core.timebase import now_ns, seconds_to_ns

    pid = c.post("/api/projects", json={"name": f"classes-{uuid.uuid4().hex[:8]}"}, headers=admin).json()["project_id"]

    async def _seed():
        from db.models import Frame
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology

        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="CLS-1", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                             city="BLR", sensors={}, ontology_version=get_ontology().version,
                             project_id=uuid.UUID(pid)))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/0.jpg",
                         width=1920, height=1080, quality=0.9, scene={}))
            await db.commit()
        return str(fid)

    _clear_db_cache()
    try:
        return pid, asyncio.run(_seed())
    finally:
        _clear_db_cache()


def test_a_project_set_reaches_the_editor_through_the_ontology():
    from _authutil import auth_headers

    from services.autolabel.ontology import get_ontology

    admin = auth_headers("admin")
    onto = get_ontology()
    with _client() as c:
        pid, fid = _project_with_frame(c, admin)
        before = c.get("/api/ontology", params={"frame_id": fid}, headers=admin).json()
        assert before["active_class_ids"] is None and before["class_set"] == "india"

        r = c.put(f"/api/projects/{pid}/classes", json={"template": "generic"}, headers=admin)
        assert r.status_code == 200, r.text
        assert r.json()["class_set"] == "generic"

        after = c.get("/api/ontology", params={"frame_id": fid}, headers=admin).json()
        assert after["class_set"] == "generic"
        assert onto.by_name("autorickshaw").id not in after["active_class_ids"]
        assert len(after["classes"]) == len(onto.classes), "every class stays nameable"

        listed = {p["project_id"]: p for p in c.get("/api/projects", headers=admin).json()["projects"]}
        assert listed[pid]["class_set"] == "generic" and listed[pid]["class_count"] < listed[pid]["of"]

        sedan = onto.by_name("sedan").id
        r = c.put(f"/api/projects/{pid}/classes", json={"class_ids": [sedan]}, headers=admin)
        assert r.status_code == 200 and r.json()["class_set"] == "custom"
        mine = c.get("/api/ontology", params={"project_id": pid}, headers=admin).json()["active_class_ids"]
        assert sedan in mine and onto.by_name("pedestrian").id not in mine

        assert c.put(f"/api/projects/{pid}/classes", json={"template": "india"}, headers=admin).json()["class_set"] == "india"
        assert c.get("/api/ontology", params={"frame_id": fid}, headers=admin).json()["active_class_ids"] is None


def test_setting_classes_is_refused_when_malformed_or_not_an_administrator():
    from _authutil import auth_headers

    admin = auth_headers("admin")
    with _client() as c:
        pid, _ = _project_with_frame(c, admin)
        url = f"/api/projects/{pid}/classes"
        assert c.put(url, json={}, headers=admin).status_code == 400
        assert c.put(url, json={"template": "generic", "class_ids": [11]}, headers=admin).status_code == 400
        assert c.put(url, json={"template": "martian"}, headers=admin).status_code == 400
        assert c.put(url, json={"class_ids": [999_999]}, headers=admin).status_code == 400
        assert c.put(url, json={"template": "generic"}, headers=auth_headers("reviewer")).status_code == 403
        t = c.get("/api/ontology/templates", headers=auth_headers("annotator")).json()["templates"]
        assert {x["name"] for x in t} == {"india", "generic"}
