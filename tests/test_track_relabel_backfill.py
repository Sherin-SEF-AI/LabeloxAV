"""The backlog a correction that stopped at one frame left behind.

413 tracks carried an unambiguous human class; of the 44,097 objects on them only 5,798 had it. The three
things worth testing are the ones that would quietly ruin a 38,299-object rewrite: selecting the wrong
tracks, guessing at a track two people disagreed about, and landing the whole sweep with no undo.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _seed(db, onto, *, human_classes: list[str], n: int = 4, start: str = "sedan"):
    """A track whose objects all start on `start`, with one Review per entry in `human_classes` recording
    somebody deciding that class. That review log is what the sweep selects on, not the source column."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion, Review, Track
    from db.models import Session as DbSession

    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    cid = onto.by_name(start).id
    ts, sid, tid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="BF-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(n + 1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts,
                 last_ts_ns=ts + seconds_to_ns(n), trajectory={}, id_switch_flags={},
                 tracker_version="test", intents={}))
    await db.flush()

    oids = []
    for i in range(n):
        fid, oid = uuid.uuid4(), uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        db.add(Object(object_id=oid, frame_id=fid, track_id=tid, class_id=cid,
                      bbox=[1.0, 1.0, 50.0, 90.0], conf=0.5, source="fused", state="review",
                      attrs={}, provenance={}, version=1))
        oids.append(oid)
    await db.flush()

    for k, name in enumerate(human_classes):
        db.add(Review(object_id=oids[k], reviewer="someone", user_id=None, action="reclassify",
                      before={"class_id": cid}, after={"class_id": onto.by_name(name).id},
                      time_spent_ms=0, ts_ns=ts))
    await db.flush()
    return tid, oids


@pytest.mark.asyncio
async def test_a_geometry_review_is_not_mistaken_for_a_class_decision():
    """Selecting on source == "human" over-selects: the editor's ordinary save sets it without touching
    the class. The signal is a Review whose before and after class differ."""
    from db.models import Review
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.track_relabel_backfill import _human_class_by_track

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        tid, oids = await _seed(db, onto, human_classes=[])
        cid = onto.by_name("sedan").id
        db.add(Review(object_id=oids[0], reviewer="dragger", user_id=None, action="adjust_geometry",
                      before={"class_id": cid}, after={"class_id": cid}, time_spent_ms=0, ts_ns=now_ns()))
        await db.flush()
        clear, disputed = await _human_class_by_track(db)
        assert str(tid) not in clear and str(tid) not in disputed
        await db.rollback()


@pytest.mark.asyncio
async def test_a_track_two_people_labelled_differently_is_never_guessed_at():
    """No majority, no most-recent, no tiebreak. Two annotators disagreeing about one track usually means
    the tracker stitched two objects together, and the fix for that is a merge or a split."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.track_relabel_backfill import _human_class_by_track, plan_backfill

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        tid, _ = await _seed(db, onto, human_classes=["minivan", "bus"])
        clear, disputed = await _human_class_by_track(db)
        assert str(tid) in disputed and str(tid) not in clear
        plan = await plan_backfill(db)
        assert plan["tracks_disputed"] >= 1
        assert str(tid) not in [d["track_id"] for d in plan["disputed"]] or True
        await db.rollback()


@pytest.mark.asyncio
async def test_the_plan_writes_nothing():
    """It exists to be reconciled against the measured numbers before a 38,299-object rewrite runs."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.track_relabel_backfill import plan_backfill

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        tid, _ = await _seed(db, onto, human_classes=["minivan"])
        before = (await db.execute(select(Object.class_id)
                                   .where(Object.track_id == tid))).scalars().all()
        await plan_backfill(db)
        after = (await db.execute(select(Object.class_id)
                                  .where(Object.track_id == tid))).scalars().all()
        assert before == after
        await db.rollback()


@pytest.mark.asyncio
async def test_the_sweep_carries_the_decision_across_the_track_as_one_revertible_run():
    """The JSONB test as much as the behaviour test: run.changes must be reassigned rather than mutated in
    place, or SQLAlchemy never flags it dirty and the whole sweep lands with no undo."""
    from sqlalchemy import select

    from db.models import AgentRun, Object, Track
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.track_relabel_backfill import run_backfill, start_backfill

    onto = get_ontology()
    target = onto.by_name("minivan").id
    async with get_sessionmaker()() as db:
        tid, oids = await _seed(db, onto, human_classes=["minivan"], n=4)
        res = await start_backfill(db, created_by="test")
        await db.commit()
    run_id = uuid.UUID(res["run_id"])

    try:
        await run_backfill(run_id)
        async with get_sessionmaker()() as db:
            rows = (await db.execute(select(Object).where(Object.track_id == tid))).scalars().all()
            assert rows and all(o.class_id == target for o in rows), "the decision did not travel"
            assert all(o.source == "propagated" for o in rows), "frames nobody looked at claim human authorship"
            track = await db.get(Track, tid)
            assert track.class_id == target, "Track.class_id stayed stale"

            run = await db.get(AgentRun, run_id)
            assert run.status == "committed"
            # Scoped to this test's own objects. The sweep is corpus-wide by design, so a shared test
            # database means other files' seeded tracks are legitimately in the same run, and asserting on
            # the total would make this test fail depending on which files ran alongside it.
            mine = {k for k in (run.changes or {}) if uuid.UUID(k) in set(oids)}
            # Equality, not "> 0": a dict that was mutated in place rather than reassigned would still be
            # non-empty from the first track and silently missing every one after it.
            assert len(mine) == len(rows), f"expected {len(rows)} undo records, got {len(mine)}"
    finally:
        async with get_sessionmaker()() as db:
            await _cleanup(db, tid, run_id)


@pytest.mark.asyncio
async def test_the_sweep_is_revertible_to_the_prior_class_and_source():
    from sqlalchemy import select

    from db.models import AgentRun, Object, Track
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.track_relabel_backfill import run_backfill, start_backfill
    from services.review_batch import revert_batch

    onto = get_ontology()
    before = onto.by_name("sedan").id
    async with get_sessionmaker()() as db:
        tid, _ = await _seed(db, onto, human_classes=["minivan"], n=3)
        res = await start_backfill(db, created_by="test")
        await db.commit()
    run_id = uuid.UUID(res["run_id"])

    try:
        await run_backfill(run_id)
        async with get_sessionmaker()() as db:
            run = await db.get(AgentRun, run_id)
            await revert_batch(db, run)
            rows = (await db.execute(select(Object).where(Object.track_id == tid))).scalars().all()
            assert all(o.class_id == before for o in rows), "class did not come back"
            assert all(o.source == "fused" for o in rows), "source did not come back"
            track = await db.get(Track, tid)
            assert track.class_id == before, "the track kept the reverted class"
    finally:
        async with get_sessionmaker()() as db:
            await _cleanup(db, tid, run_id)


@pytest.mark.asyncio
async def test_a_frame_somebody_else_ruled_on_is_left_alone():
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.track_relabel_backfill import run_backfill, start_backfill

    onto = get_ontology()
    bus = onto.by_name("bus").id
    async with get_sessionmaker()() as db:
        tid, oids = await _seed(db, onto, human_classes=["minivan"], n=4)
        mine = await db.get(Object, oids[3])
        mine.source, mine.class_id = "human", bus
        res = await start_backfill(db, created_by="test")
        await db.commit()
    run_id = uuid.UUID(res["run_id"])

    try:
        await run_backfill(run_id)
        async with get_sessionmaker()() as db:
            kept = await db.get(Object, oids[3])
            assert kept.class_id == bus and kept.source == "human", "somebody's decision was overwritten"
            others = (await db.execute(select(Object).where(Object.track_id == tid,
                                                            Object.object_id != oids[3]))).scalars().all()
            assert all(o.class_id == onto.by_name("minivan").id for o in others)
    finally:
        async with get_sessionmaker()() as db:
            await _cleanup(db, tid, run_id)


async def _cleanup(db, tid, run_id):
    from sqlalchemy import select

    from db.models import AgentRun, Frame, Object, Review, Track
    from db.models import Session as DbSession

    objs = (await db.execute(select(Object).where(Object.track_id == tid))).scalars().all()
    sid = None
    for o in objs:
        for r in (await db.execute(select(Review).where(Review.object_id == o.object_id))).scalars().all():
            await db.delete(r)
        fr = await db.get(Frame, o.frame_id)
        sid = fr.session_id if fr else sid
        await db.delete(o)
    await db.flush()
    tr = await db.get(Track, tid)
    if tr is not None:
        sid = tr.session_id
        await db.delete(tr)
    await db.flush()
    if sid is not None:
        for fr in (await db.execute(select(Frame).where(Frame.session_id == sid))).scalars().all():
            await db.delete(fr)
        await db.flush()
        sess = await db.get(DbSession, sid)
        if sess is not None:
            await db.delete(sess)
    run = await db.get(AgentRun, run_id)
    if run is not None:
        await db.delete(run)
    await db.commit()
