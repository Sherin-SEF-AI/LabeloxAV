"""Track.class_id, the denormalised copy nothing kept in sync, and the button it broke.

`Track` carries its own `class_id` alongside every object's. Only track creation ever wrote it, so a track
relabel moved 93 objects and left the track claiming the old class. Measured: 2,019 tracks had drifted from
their own objects.

That is not cosmetic, because `services/intelligence/propagate.py` fills interpolated gaps from it. Relabel
a track and then interpolate, two clicks apart on the same page, and the new boxes come back with the class
you just corrected away from.

The same line had never run at all: it wrote `source="interp"`, which is not one of the nine values
`ck_object_source` admits, so every call raised a check violation. Zero objects in the corpus carry it.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _user(db, role: str = "reviewer"):
    from db.models import User

    u = User(user_id=uuid.uuid4(), name=f"sync-{uuid.uuid4().hex[:6]}", role=role)
    db.add(u)
    await db.flush()
    return u


async def _track_with_gaps(db, onto, *, class_name: str = "sedan"):
    """Two keyframes with empty frames between them, which is what interpolate exists to fill."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion, Track
    from db.models import Session as DbSession

    cid = next(c.id for c in onto.classes if c.name == class_name)
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    ts, sid, tid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="GAP-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(10), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts,
                 last_ts_ns=ts + seconds_to_ns(4), trajectory={}, id_switch_flags={},
                 tracker_version="test", intents={}))
    await db.flush()

    oids = []
    for i in range(5):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        # Only the first and last frame carry a box; frames 1-3 are the gap.
        if i in (0, 4):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, track_id=tid, class_id=cid,
                          bbox=[float(i), 1.0, float(i) + 50, 90.0], conf=0.5, source="fused",
                          state="review", attrs={}, provenance={}, version=1))
            oids.append(oid)
    await db.flush()
    return sid, tid, oids


@pytest.mark.asyncio
async def test_a_relabel_moves_the_tracks_own_class_too():
    from db.models import Track
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        _, tid, _ = await _track_with_gaps(db, onto)
        await relabel_track(tid, RelabelTrackIn(class_name="minivan"), db, await _user(db))
        track = await db.get(Track, tid)
        assert track.class_id == onto.by_name("minivan").id, "Track.class_id stayed on the old class"
        await db.rollback()


@pytest.mark.asyncio
async def test_reverting_a_relabel_puts_the_tracks_class_back():
    """The track is not an object, so it has no entry in the run's changes. Without an explicit restore a
    revert leaves the track claiming the class the operator just took back."""
    from db.models import AgentRun, Track
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology
    from services.review_batch import revert_batch

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        before = onto.by_name("sedan").id
        _, tid, _ = await _track_with_gaps(db, onto, class_name="sedan")
        out = await relabel_track(tid, RelabelTrackIn(class_name="minivan"), db, await _user(db))
        run = await db.get(AgentRun, uuid.UUID(out["run_id"]))
        await revert_batch(db, run)
        track = await db.get(Track, tid)
        assert track.class_id == before, "the track kept the reverted class"
        await db.rollback()


@pytest.mark.asyncio
async def test_interpolating_a_relabelled_track_does_not_reinject_the_old_class():
    """The two-click bug: relabel on the track page, then interpolate gaps on the same page, and the new
    boxes arrive with the class you corrected away from. This also exercises the source value, which was
    "interp" and is rejected by ck_object_source, so this path could never run at all."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology
    from services.intelligence.propagate import interpolate_track

    onto = get_ontology()
    target = onto.by_name("minivan").id
    async with get_sessionmaker()() as db:
        _, tid, _ = await _track_with_gaps(db, onto, class_name="sedan")
        await relabel_track(tid, RelabelTrackIn(class_name="minivan"), db, await _user(db))
        await db.commit()

    try:
        out = await interpolate_track(tid)
        assert out["created"] > 0, f"interpolate created nothing: {out}"
        async with get_sessionmaker()() as db:
            rows = (await db.execute(
                select(Object).where(Object.track_id == tid,
                                     Object.provenance["method"].astext == "interpolate"))).scalars().all()
            assert rows, "no interpolated boxes were written"
            assert all(o.class_id == target for o in rows), \
                "gap boxes came back on the pre-relabel class"
            assert all(o.source == "interpolated" for o in rows), \
                "source must be one ck_object_source admits"
    finally:
        async with get_sessionmaker()() as db:
            from db.models import Frame, Track
            from db.models import Session as DbSession
            objs = (await db.execute(select(Object).where(Object.track_id == tid))).scalars().all()
            sid = None
            for o in objs:
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
            await db.commit()
