""""Relabel all frames" has to be able to reach all the frames.

The frame query was `select distinct(frame_id) where source <> 'human' limit N`, with no ordering and no
exclusion. Postgres answers that identically every time, so a default max_frames=200 run covered 200 of the
34,121 eligible frames and covered the same 200 on every subsequent run. Pressing the button a hundred times
re-read those 200 and never touched the other 33,921.

Nothing about that looked broken from outside. The runs committed, the counts were real, and because the
model happened to agree on those particular frames the totals were zero, so it read as a job that found
nothing rather than a job that could not see anything. That is why the test is about coverage and not about
relabel counts: the counts were never the symptom.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Text, distinct, select

from core.timebase import now_ns
from db.models import AgentRun, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker

pytestmark = pytest.mark.db


def _eligible_query(session_id=None):
    """The selection `run_relabel_all` uses, as a query the test can run directly.

    Duplicated rather than imported because the production copy is welded into a background coroutine that
    needs a GPU, an object store and real imagery to reach. What matters here is the set it selects, and
    that is expressible on its own.
    """
    seen = (select(AgentRun.scope["frame_id"].astext)
            .where(AgentRun.kind == "relabel", AgentRun.scope["frame_id"].astext.isnot(None)))
    q = (select(distinct(Object.frame_id))
         .where(Object.source != "human", Object.frame_id.cast(Text).notin_(seen)))
    if session_id is not None:
        # Scoped in the tests because the suite shares one database: ordering across every frame in it would
        # interleave other tests' rows and make a coverage assertion about this test's frames unprovable.
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    return q.order_by(Object.frame_id)


async def _seed(db, n_frames: int) -> tuple[uuid.UUID, list[uuid.UUID]]:
    onto_version = "test"
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-RELABEL", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=onto_version)
    db.add(sess)
    ids = []
    for _ in range(n_frames):
        f = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/t.jpg", width=100, height=100)
        db.add(f)
        db.add(Object(object_id=uuid.uuid4(), frame_id=f.frame_id, class_id=1,
                      bbox=[1.0, 1.0, 20.0, 20.0], conf=0.5, source="fused", state="review"))
        ids.append(f.frame_id)
    await db.commit()
    return sess.session_id, ids


def _mark_read(db, frame_id: uuid.UUID) -> None:
    """What `commit_relabel` leaves behind for every frame it reads, whether or not it changed anything."""
    db.add(AgentRun(run_id=uuid.uuid4(), kind="relabel", status="committed",
                    scope={"frame_id": str(frame_id)}, policy={}, counts={"total": 1}, changes={}, critic={}))


async def test_a_second_run_does_not_re_read_the_first_runs_frames():
    """The defect. Two runs in a row used to return the identical frame list."""
    async with get_sessionmaker()() as db:
        sid, _ = await _seed(db, 6)
        first = list((await db.execute(_eligible_query(sid).limit(3))).scalars().all())
        assert len(first) == 3
        for fid in first:
            _mark_read(db, fid)
        await db.commit()

        second = list((await db.execute(_eligible_query(sid).limit(3))).scalars().all())
        assert set(second).isdisjoint(set(first)), "a second run must move on to frames not yet read"


async def test_repeated_runs_eventually_cover_everything():
    """The property the name claims. Without it the job is bounded at its first batch forever."""
    async with get_sessionmaker()() as db:
        sid, ids = await _seed(db, 9)
        seeded = set(ids)
        covered: set[uuid.UUID] = set()
        for _ in range(8):
            batch = list((await db.execute(_eligible_query(sid).limit(2))).scalars().all())
            if not batch:
                break
            for fid in batch:
                _mark_read(db, fid)
            await db.commit()
            covered |= set(batch)
        assert seeded <= covered, "every seeded frame should have been reached within a few runs"


async def test_a_frame_read_but_unchanged_still_counts_as_read():
    """The subtle half. `commit_relabel` writes a child run for every frame it reads, changed or not, and
    that is deliberately the marker: keying on "did it change something" would make a frame the model agrees
    about get re-read forever, which is exactly the population the old query was stuck on."""
    async with get_sessionmaker()() as db:
        sid, frames = await _seed(db, 2)
        _mark_read(db, frames[0])   # changes={} : read, nothing altered
        await db.commit()
        remaining = list((await db.execute(_eligible_query(sid))).scalars().all())
        assert frames[0] not in remaining
        assert frames[1] in remaining


async def test_the_order_is_deterministic():
    """Two runs of the same query must agree, so a run is reproducible and two runs can be told apart by
    what they covered."""
    async with get_sessionmaker()() as db:
        sid, _ = await _seed(db, 5)
        a = list((await db.execute(_eligible_query(sid).limit(4))).scalars().all())
        b = list((await db.execute(_eligible_query(sid).limit(4))).scalars().all())
        assert a == b


async def test_human_labelled_objects_are_never_selected():
    """The agent does not overwrite people, and the selection is the first place that is enforced."""
    async with get_sessionmaker()() as db:
        sid, frames = await _seed(db, 2)
        obj = (await db.execute(
            select(Object).where(Object.frame_id == frames[0]))).scalars().first()
        obj.source = "human"
        await db.commit()
        remaining = list((await db.execute(_eligible_query(sid))).scalars().all())
        assert frames[0] not in remaining
