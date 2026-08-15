"""A frame that cannot be read was retried forever.

The corpus relabel walks forward by excluding frames that already have a committed child `relabel` run. A
frame that raises never gets one, so the next run selects it again, fails again, and marks it done again,
inside a progress record that is discarded when the run ends.

Eleven frames whose images are absent from object storage sat at the head of that queue. Successive batches
read 0 frames in 5 seconds and reported `committed`, and "frames remaining" stayed at 11 no matter how many
times the job ran. The pass could never reach zero.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from core.timebase import now_ns
from db.models import AgentRun, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.agent.relabel_agent import run_relabel_all

pytestmark = pytest.mark.db


async def _seed_unreadable(db) -> tuple[uuid.UUID, uuid.UUID]:
    """A frame whose URI is well formed but whose object is not in the store, which is what the eleven are.

    Returns the session too, because every walk below is scoped to it. Unscoped, `run_relabel_all` reads the
    whole database: this suite shares one, and once enough dead fixtures accumulate the walk trips its own
    consecutive-failure guard on somebody else's frames and stops before it ever reaches this one. The test
    then fails for a reason that has nothing to do with what it is asserting, which is what it did.
    """
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-SKIP", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                 img_uri=f"s3://labeloxav/missing/{uuid.uuid4().hex}.jpg", width=1920, height=1080))
    await db.flush()
    db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=1, bbox=[10, 10, 200, 200],
                  conf=0.5, source="auto_accept", state="review"))
    await db.commit()
    return fid, sess.session_id


async def _child_runs(db, frame_id) -> int:
    return int((await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.kind == "relabel", AgentRun.scope["frame_id"].astext == str(frame_id)))).scalar() or 0)


async def test_an_unreadable_frame_is_recorded_so_the_walk_moves_past_it():
    """Without this the frame has no child run, so every later pass selects it again."""
    async with get_sessionmaker()() as db:
        fid, sid = await _seed_unreadable(db)
        rid = uuid.uuid4()
        db.add(AgentRun(run_id=rid, kind="relabel_all", status="running", scope={}, created_by="t"))
        await db.commit()

    await run_relabel_all(rid, max_frames=500, created_by="t", session_id=str(sid))

    async with get_sessionmaker()() as db:
        assert await _child_runs(db, fid) >= 1


async def test_the_skip_is_marked_as_a_skip_not_as_work_done():
    """It must not read as "this frame was relabelled". It was read and could not be used."""
    async with get_sessionmaker()() as db:
        fid, sid = await _seed_unreadable(db)
        rid = uuid.uuid4()
        db.add(AgentRun(run_id=rid, kind="relabel_all", status="running", scope={}, created_by="t"))
        await db.commit()

    await run_relabel_all(rid, max_frames=500, created_by="t", session_id=str(sid))

    async with get_sessionmaker()() as db:
        row = (await db.execute(
            select(AgentRun).where(AgentRun.kind == "relabel",
                                   AgentRun.scope["frame_id"].astext == str(fid)))).scalars().first()
    assert row is not None
    assert row.status == "skipped"
    assert row.changes == {}
    assert row.critic.get("error"), "the reason is kept so the skip is auditable rather than silent"


async def test_a_second_pass_does_not_select_the_same_dead_frame_again():
    """The behaviour that was missing: the corpus pass has to be able to reach zero remaining."""
    async with get_sessionmaker()() as db:
        fid, sid = await _seed_unreadable(db)
        rid = uuid.uuid4()
        db.add(AgentRun(run_id=rid, kind="relabel_all", status="running", scope={}, created_by="t"))
        await db.commit()
    await run_relabel_all(rid, max_frames=500, created_by="t", session_id=str(sid))

    async with get_sessionmaker()() as db:
        before = await _child_runs(db, fid)
        rid2 = uuid.uuid4()
        db.add(AgentRun(run_id=rid2, kind="relabel_all", status="running", scope={}, created_by="t"))
        await db.commit()
    await run_relabel_all(rid2, max_frames=500, created_by="t", session_id=str(sid))

    async with get_sessionmaker()() as db:
        after = await _child_runs(db, fid)
    assert after == before, "the second pass must not have touched the frame at all"
