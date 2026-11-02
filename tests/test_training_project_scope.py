"""Training on one project: its sessions only, and the classes it labels unless told otherwise.

A team that keeps a client's drives in their own project wants a detector trained on those drives, for the
classes that project labels. Training started by somebody who cannot see the project is refused like any
other request for something hidden from them.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _seed_project(name: str, classes: list[int], class_set: list[int] | None, visibility: str = "team") -> tuple[str, str]:
    """A project with one session holding one object per class. Returns (project_id, session_id)."""
    from db.models import Frame, LabelProject, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    async def _go():
        pid, sid, fid, ts = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(LabelProject(project_id=pid, name=f"{name}-{pid.hex[:6]}", class_ids=class_set, visibility=visibility))
            await db.flush()
            db.add(DbSession(session_id=sid, project_id=pid, vehicle_id="TRAIN-P", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={}, ontology_version=onto.version))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/t.jpg",
                         width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            for i, cid in enumerate(classes):
                x = 10.0 + 80 * i
                db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid, bbox=[x, 10.0, x + 60, 70.0],
                              conf=0.9, source="fused", state="review", attrs={}, provenance={}, version=1))
            await db.commit()
        return str(pid), str(sid)

    return _run(_go())


def _things():
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    sedan, pedestrian, auto = onto.by_name("sedan").id, onto.by_name("pedestrian").id, onto.by_name("autorickshaw").id
    return sedan, pedestrian, auto


def test_a_project_scope_trains_on_its_sessions_and_its_classes():
    from services.training.dataset_builder import BuildSpec, _select

    sedan, pedestrian, auto = _things()
    mine, my_sid = _seed_project("mine", [sedan, pedestrian, auto], class_set=[sedan, pedestrian])
    _, other_sid = _seed_project("other", [sedan, auto], class_set=None)

    rows = _run(_select(BuildSpec(project_id=mine, conf_floor=0.0)))
    assert {r["session_id"] for r in rows} == {my_sid}
    assert {r["class_id"] for r in rows} == {sedan, pedestrian}, "the project's class set, autorickshaw left out"

    # An explicit class list wins over the project's set.
    rows = _run(_select(BuildSpec(project_id=mine, include_classes=["autorickshaw"], conf_floor=0.0)))
    assert {r["class_id"] for r in rows} == {auto}
    assert other_sid not in {r["session_id"] for r in _run(_select(BuildSpec(project_id=mine, conf_floor=0.0)))}


def test_training_on_a_project_hidden_from_the_caller_is_refused():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    sedan, _, _ = _things()
    hidden, _ = _seed_project("hidden", [sedan], class_set=None, visibility="members")
    with TestClient(app) as c:
        body = {"task_type": "detection", "dataset_spec": {"project_id": hidden}, "hparams": {"epochs": 1}}
        r = c.post("/api/training/start", json=body, headers=auth_headers("reviewer"))
        assert r.status_code == 404, r.text
        bad = {**body, "dataset_spec": {"project_id": "not-a-uuid"}}
        assert c.post("/api/training/start", json=bad, headers=auth_headers("reviewer")).status_code == 400
