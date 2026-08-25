"""What the editor's drive picker needs to know about a session, and the two ways it went wrong.

The picker has to separate a drive nobody has started from one somebody is part way through. The obvious
signal, the existing accepted-plus-auto_accept fraction, cannot: its median across this corpus is 0.011, so
a percentage bar reads about 1% on nearly every drive.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _session(db, *, frames: int, objects_per_frame: int = 0):
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
    db.add(DbSession(session_id=sid, vehicle_id="STATE-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(frames + 1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    oids = []
    for i in range(frames):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        for _ in range(objects_per_frame):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[1.0, 1.0, 9.0, 9.0],
                          conf=0.5, source="fused", state="review", attrs={}, provenance={}, version=1))
            oids.append(oid)
    await db.flush()
    return sid, oids


@pytest.mark.asyncio
async def test_a_session_with_no_frames_is_still_reported():
    """126 of 377 sessions here are LiDAR and 3D captures with no camera frames, and the editor 404s on
    them. Driving the query from Frame returned 252 rows and silently omitted every one of the sessions
    this endpoint exists to mark."""
    from db.session import get_sessionmaker
    from services.api.routers.meta import session_states

    async with get_sessionmaker()() as db:
        sid, _ = await _session(db, frames=0)
        rows = {r["session_id"]: r for r in await session_states(db)}
        assert str(sid) in rows, "a frameless session vanished from the report"
        assert rows[str(sid)]["frames"] == 0
        await db.rollback()


@pytest.mark.asyncio
async def test_frames_without_detections_are_distinguished_from_labelled_ones():
    from db.session import get_sessionmaker
    from services.api.routers.meta import session_states

    async with get_sessionmaker()() as db:
        bare, _ = await _session(db, frames=3, objects_per_frame=0)
        full, _ = await _session(db, frames=3, objects_per_frame=2)
        rows = {r["session_id"]: r for r in await session_states(db)}
        assert rows[str(bare)]["frames"] == 3 and rows[str(bare)]["objects"] == 0
        assert rows[str(full)]["objects"] == 6
        await db.rollback()


@pytest.mark.asyncio
async def test_only_a_persons_review_counts_as_worked_on():
    """Machine writers put rows in `review` too: 27,396 of them here are from the track-relabel backfill
    against 1,613 written by a human. Counting those would mark almost every drive as reviewed by somebody
    who never opened it."""
    from db.models import Review, User
    from db.session import get_sessionmaker
    from services.api.routers.meta import session_states

    async with get_sessionmaker()() as db:
        sid, oids = await _session(db, frames=2, objects_per_frame=2)
        u = User(user_id=uuid.uuid4(), name=f"who-{uuid.uuid4().hex[:6]}", role="reviewer")
        db.add(u)
        await db.flush()

        # A machine row, exactly as the backfill writes it: no user_id.
        db.add(Review(object_id=oids[0], reviewer="track_relabel_backfill", user_id=None,
                      action="reclassify_track_backfill", before={}, after={}, time_spent_ms=0,
                      ts_ns=now_ns()))
        await db.flush()
        rows = {r["session_id"]: r for r in await session_states(db)}
        assert rows[str(sid)]["reviewed_objects"] == 0, "a machine review counted as a person's"

        db.add(Review(object_id=oids[1], reviewer=u.name, user_id=u.user_id, action="confirm",
                      before={}, after={}, time_spent_ms=0, ts_ns=now_ns()))
        await db.flush()
        rows = {r["session_id"]: r for r in await session_states(db)}
        assert rows[str(sid)]["reviewed_objects"] == 1, "a person's review was not counted"
        await db.rollback()
