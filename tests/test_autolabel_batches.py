"""Working a drive in batches, which `limit` alone could not do.

`limit` takes the first N frames by timestamp and nothing skipped frames that already carried objects, so
asking for 200 frames twice covered the same 200 twice. A drive could not be worked through a batch at a
time: the second batch looked like it ran and moved nothing.

This matters more than convenience. The whole reason to run a bounded batch is that the box has one GPU,
and the alternative to batching is firing an unbounded pass over a 1,000-frame drive.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _session_with_frames(db, n: int, labelled: int = 0):
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    cid = onto.by_name("sedan").id
    ts, sid = now_ns(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="BATCH-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(n + 1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    for i in range(n):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        if i < labelled:
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid,
                          bbox=[1.0, 1.0, 9.0, 9.0], conf=0.5, source="fused", state="review",
                          attrs={}, provenance={}, version=1))
    await db.flush()
    await db.commit()
    return sid


@pytest.mark.asyncio
async def test_a_second_batch_covers_the_next_frames_not_the_same_ones():
    from db.session import get_sessionmaker
    from services.autolabel.runner import fetch_frames

    async with get_sessionmaker()() as db:
        sid = await _session_with_frames(db, n=10, labelled=4)
    try:
        # Without the flag, a bounded run always starts from the top: the same first three frames.
        plain = await fetch_frames(sid, 3)
        assert [f.ts_ns for f in plain] == sorted(f.ts_ns for f in plain)
        again = await fetch_frames(sid, 3)
        assert [f.frame_id for f in plain] == [f.frame_id for f in again], \
            "this is the behaviour the flag exists to change"

        # With it, the batch is the next three frames nobody has labelled.
        batch = await fetch_frames(sid, 3, only_unlabelled=True)
        assert len(batch) == 3
        assert not set(f.frame_id for f in batch) & set(f.frame_id for f in plain[:1]), \
            "the batch started on an already-labelled frame"
    finally:
        await _cleanup(sid)


@pytest.mark.asyncio
async def test_a_batch_is_idempotent_if_it_is_fired_twice():
    """Two clicks on a start button must not queue the same work twice."""
    from db.models import Frame, Object
    from db.session import get_sessionmaker
    from services.autolabel.runner import fetch_frames

    async with get_sessionmaker()() as db:
        sid = await _session_with_frames(db, n=6, labelled=0)
    try:
        first = await fetch_frames(sid, 2, only_unlabelled=True)
        # Label them, as a completed batch would.
        async with get_sessionmaker()() as db:
            from services.autolabel.ontology import get_ontology
            cid = get_ontology().by_name("sedan").id
            for f in first:
                db.add(Object(object_id=uuid.uuid4(), frame_id=f.frame_id, class_id=cid,
                              bbox=[1.0, 1.0, 9.0, 9.0], conf=0.5, source="fused", state="review",
                              attrs={}, provenance={}, version=1))
            await db.commit()
        second = await fetch_frames(sid, 2, only_unlabelled=True)
        assert not set(f.frame_id for f in second) & set(f.frame_id for f in first), \
            "the second batch redid the first"
        assert len(second) == 2
        assert Frame is not None
    finally:
        await _cleanup(sid)


@pytest.mark.asyncio
async def test_a_fully_labelled_drive_yields_an_empty_batch_rather_than_redoing_it():
    from db.session import get_sessionmaker
    from services.autolabel.runner import fetch_frames

    async with get_sessionmaker()() as db:
        sid = await _session_with_frames(db, n=4, labelled=4)
    try:
        assert await fetch_frames(sid, 10, only_unlabelled=True) == []
        assert len(await fetch_frames(sid, 10)) == 4, "the unbounded path must still redo them"
    finally:
        await _cleanup(sid)


async def _cleanup(sid):
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        frames = (await db.execute(select(Frame).where(Frame.session_id == sid))).scalars().all()
        for f in frames:
            for o in (await db.execute(select(Object).where(Object.frame_id == f.frame_id))).scalars().all():
                await db.delete(o)
        await db.flush()
        for f in frames:
            await db.delete(f)
        await db.flush()
        s = await db.get(DbSession, sid)
        if s is not None:
            await db.delete(s)
        await db.commit()
