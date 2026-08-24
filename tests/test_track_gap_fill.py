"""An object that blinks out for a frame and comes back.

Stepping to the next frame and finding an object gone, then back a frame later, is what 9,460 of 11,287
tracks do: 137,960 missing frames, in holes averaging 1.6 frames and rarely longer than 5. The feature
built to fill them, services/intelligence/propagate.py interpolate_track, had never run once, because it
wrote a `source` the ck_object_source constraint rejects.

Two things make filling them safe rather than reckless: a hole longer than a flicker is refused instead of
having a straight line drawn through it, and the whole sweep is one run that can be taken back.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _track_with_hole(db, onto, *, total: int, present: list[int], class_name: str = "sedan"):
    """A session of `total` frames where the track only has a box on the `present` indices."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.models import Track

    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    cid = onto.by_name(class_name).id
    ts, sid, tid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="GAP-2", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(total + 1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts,
                 last_ts_ns=ts + seconds_to_ns(total), trajectory={}, id_switch_flags={},
                 tracker_version="test", intents={}))
    await db.flush()

    for i in range(total):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        if i in present:
            # A box that moves ten pixels a frame, so an interpolated one has a right answer to be near.
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, track_id=tid, class_id=cid,
                          bbox=[10.0 * i, 10.0, 10.0 * i + 50, 90.0], conf=0.9, source="fused",
                          state="review", attrs={}, provenance={}, version=1))
    await db.flush()
    return sid, tid


async def _cleanup(db, sid, tid, run_id=None):
    from sqlalchemy import select

    from db.models import AgentRun, Frame, Object
    from db.models import Session as DbSession
    from db.models import Track

    for o in (await db.execute(select(Object).where(Object.track_id == tid))).scalars().all():
        await db.delete(o)
    await db.flush()
    tr = await db.get(Track, tid)
    if tr is not None:
        await db.delete(tr)
    await db.flush()
    for fr in (await db.execute(select(Frame).where(Frame.session_id == sid))).scalars().all():
        await db.delete(fr)
    await db.flush()
    sess = await db.get(DbSession, sid)
    if sess is not None:
        await db.delete(sess)
    if run_id is not None:
        run = await db.get(AgentRun, run_id)
        if run is not None:
            await db.delete(run)
    await db.commit()


@pytest.mark.asyncio
async def test_a_short_hole_is_filled_with_a_box_between_its_neighbours():
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.intelligence.propagate import interpolate_track

    async with get_sessionmaker()() as db:
        sid, tid = await _track_with_hole(db, get_ontology(), total=6, present=[0, 1, 4, 5])
        await db.commit()
    try:
        res = await interpolate_track(tid)
        assert res["created"] == 2, f"expected the two missing frames, got {res}"
        async with get_sessionmaker()() as db:
            made = (await db.execute(select(Object).where(
                Object.track_id == tid, Object.source == "interpolated"))).scalars().all()
            xs = sorted(o.bbox[0] for o in made)
            # The real boxes sit at x = 10 * frame index, so the filled ones belong at 20 and 30.
            assert xs == pytest.approx([20.0, 30.0]), f"boxes are not between their neighbours: {xs}"
            assert all(o.state == "annotate" and o.conf == 0.5 for o in made), \
                "a linear guess must not land as a confident label"
    finally:
        async with get_sessionmaker()() as db:
            await _cleanup(db, sid, tid)


@pytest.mark.asyncio
async def test_a_long_hole_is_refused_rather_than_drawn_through():
    """A straight line across twenty frames is a line through whatever the object actually did. On a
    turning vehicle it leaves the road, and every box on it would be wrong while looking like data."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.intelligence.propagate import interpolate_track

    async with get_sessionmaker()() as db:
        sid, tid = await _track_with_hole(db, get_ontology(), total=24, present=[0, 1, 22, 23])
        await db.commit()
    try:
        res = await interpolate_track(tid, max_gap=12)
        assert res["created"] == 0, "a twenty-frame hole must not be filled"
        assert res["skipped_long_gaps"] == 20, f"the skip must be counted, got {res}"
        # And it is a refusal, not an inability: without the bound the same track fills.
        res2 = await interpolate_track(tid)
        assert res2["created"] == 20
    finally:
        async with get_sessionmaker()() as db:
            await _cleanup(db, sid, tid)


@pytest.mark.asyncio
async def test_the_sweep_is_one_run_that_deletes_what_it_created():
    """interpolate_track wrote objects with no record at all, which is fine behind a button for one track
    and not something to run across nine thousand of them."""
    from sqlalchemy import select

    from db.models import AgentRun, Object
    from db.session import get_sessionmaker
    from services.agent.runs import revert_run
    from services.autolabel.ontology import get_ontology
    from services.quality.track_gap_backfill import run_gap_fill, start_gap_fill

    async with get_sessionmaker()() as db:
        sid, tid = await _track_with_hole(db, get_ontology(), total=6, present=[0, 1, 4, 5])
        res = await start_gap_fill(db, created_by="test")
        await db.commit()
    run_id = uuid.UUID(res["run_id"])
    try:
        await run_gap_fill(run_id)
        async with get_sessionmaker()() as db:
            made = (await db.execute(select(Object).where(
                Object.track_id == tid, Object.source == "interpolated"))).scalars().all()
            assert made, "the sweep created nothing"
            run = await db.get(AgentRun, run_id)
            assert run.status == "committed"
            mine = {k for k in (run.changes or {}) if uuid.UUID(k) in {o.object_id for o in made}}
            assert len(mine) == len(made), "every created box must be recorded for the undo"
            assert all((run.changes or {})[k].get("created") for k in mine)

        async with get_sessionmaker()() as db:
            await revert_run(db, run_id)
        async with get_sessionmaker()() as db:
            left = (await db.execute(select(Object).where(
                Object.track_id == tid, Object.source == "interpolated"))).scalars().all()
            assert not left, f"revert left {len(left)} invented boxes behind"
            real = (await db.execute(select(Object).where(
                Object.track_id == tid, Object.source == "fused"))).scalars().all()
            assert len(real) == 4, "revert must not touch the real detections"
    finally:
        async with get_sessionmaker()() as db:
            await _cleanup(db, sid, tid, run_id)
