"""The home page's numbers are counts from the tables, and its first-run steps are facts about them."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns, seconds_to_ns
from db.models import AutolabelJob, Frame, Object, OntologyClass
from db.models import Session as DbSession
from db.session import get_sessionmaker

pytestmark = pytest.mark.db


async def _seed(states: list[str]) -> uuid.UUID:
    sid = uuid.uuid4()
    start = now_ns() + seconds_to_ns(10_000)  # newest session in the test database, so it leads the recent list
    async with get_sessionmaker()() as db:
        cid = (await db.execute(select(OntologyClass.id).limit(1))).scalar_one()
        db.add(DbSession(session_id=sid, vehicle_id="HOME-01", start_ts_ns=start, end_ts_ns=start + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri="s3://labeloxav/x.jpg",
                     width=640, height=480, quality=0.9))
        await db.flush()
        for st in states:
            db.add(Object(frame_id=fid, class_id=cid, bbox=[1, 1, 10, 10], conf=0.5, state=st))
        await db.commit()
    return sid


async def test_recent_session_shows_its_own_progress():
    from services.api.routers.home import home

    sid = await _seed(["review", "review", "accepted"])
    async with get_sessionmaker()() as db:
        body = await home(db)
    mine = next(s for s in body["recent_sessions"] if s["session_id"] == str(sid))
    assert mine == {**mine, "frames": 1, "objects": 3, "waiting": 2, "vehicle_id": "HOME-01"}
    assert body["recent_sessions"][0]["session_id"] == str(sid), "the newest session is first"


async def test_corpus_counts_add_up():
    from services.api.routers.home import home

    await _seed(["review", "accepted", "rejected"])
    async with get_sessionmaker()() as db:
        body = await home(db)
    c = body["corpus"]
    assert c["objects"] == sum(c["by_state"].values())
    assert c["waiting"] == c["by_state"].get("review", 0)
    assert c["settled"] == sum(c["by_state"].get(s, 0) for s in ("accepted", "auto_accept", "rejected"))


async def test_steps_follow_the_data():
    from services.api.routers.home import home

    await _seed(["accepted"])
    async with get_sessionmaker()() as db:
        db.add(AutolabelJob(job_id=uuid.uuid4(), session_id=uuid.uuid4(), status="done"))
        await db.commit()
        body = await home(db)
    assert body["steps"]["data"] and body["steps"]["autolabel"] and body["steps"]["review"]


async def test_an_export_by_script_ticks_the_export_step():
    """The SDK and CLI export through /api/export, which leaves a dataset commit and no ExportJob."""
    from db.models import DatasetCommit
    from services.api.routers.home import home
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        assert (await home(db))["steps"]["export"] is False
        db.add(DatasetCommit(commit_id=f"test-{uuid.uuid4().hex[:12]}", slice_spec={"name": "by-script"},
                             object_count=1, ontology_version=get_ontology().version))
        await db.commit()
        assert (await home(db))["steps"]["export"] is True


async def test_a_failed_job_is_listed_with_its_reason():
    from services.api.routers.home import home

    jid = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(AutolabelJob(job_id=jid, session_id=uuid.uuid4(), status="error", error="no frames for session"))
        await db.commit()
        body = await home(db)
    failed = {f["job_id"]: f for f in body["jobs"]["failed"]}
    assert str(jid) in failed and failed[str(jid)]["error"] == "no frames for session"


def test_any_signed_in_person_can_read_it():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    with TestClient(app) as c:
        assert c.get("/api/home").status_code == 401
        assert c.get("/api/home", headers=auth_headers("annotator")).status_code == 200


async def test_an_old_failure_is_history_not_attention():
    from datetime import UTC, datetime, timedelta

    from services.api.routers.home import ATTENTION_DAYS, home

    jid = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(AutolabelJob(job_id=jid, session_id=uuid.uuid4(), status="error", error="long ago"))
        await db.commit()
        await db.execute(AutolabelJob.__table__.update().where(AutolabelJob.job_id == jid)
                         .values(updated_at=datetime.now(UTC) - timedelta(days=ATTENTION_DAYS + 1)))
        await db.commit()
        body = await home(db)
    assert str(jid) not in {f["job_id"] for f in body["jobs"]["failed"]}
