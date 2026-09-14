"""Confirming a whole track without asserting a class, and without claiming to have drawn it.

Tube review is built on one keystroke covering a whole track, and the numbers say why: 7,512 tracks carry
ten or more objects and hold 549,038 between them, 95% of the corpus, while the median track is 93 frames
and the median number of frames a person has ever touched on one is 1.

Until this endpoint the only track-wide write was `relabel_track`, which requires a class name and writes
`source="propagated"` on every frame. Accepting a correctly labelled track therefore meant restating its
own class and relabelling ninety-three frames to say so. `services/api/routers/track_events.py` already
recorded that relabel "became a review path an annotator could use to accept a whole track", which is a
workaround, not an endpoint.

The tests that matter here are the ones about what accept does NOT do. It must not write `source="human"`,
because that is this repo's marker for "an agent must not touch this" and ninety-two of the ninety-three
frames were never drawn by anyone; and it must not write `source="propagated"` either, because nothing was
propagated. So `apply_review_batch` grew a `source=None`, meaning leave the object's own source alone, and
the Review row is what records who approved it.
"""

import uuid

import pytest
import pytest_asyncio

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _user(db, role: str, name: str = "tester"):
    """A real row: Review.user_id carries a foreign key into app_user, so a stand-in with a fresh uuid
    passes every assertion in this file and then fails at the flush."""
    from db.models import User

    u = User(user_id=uuid.uuid4(), name=f"{name}-{uuid.uuid4().hex[:6]}", role=role)
    db.add(u)
    await db.flush()
    return u


async def _track(db, onto, n: int, *, class_name: str = "sedan", sources: list[str] | None = None,
                 states: list[str] | None = None):
    """One track across n frames, one object per frame, which is what the corpus looks like: exactly one
    object per frame in 11,286 of 11,287 tracks."""
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
    db.add(DbSession(session_id=sid, vehicle_id="TRK-A", start_ts_ns=ts,
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
                      bbox=[1.0, 1.0, 50.0, 90.0], conf=0.5,
                      source=(sources[i] if sources else "fused"),
                      state=(states[i] if states else "review"),
                      attrs={}, provenance={}, version=1))
        oids.append(oid)
    await db.flush()
    _MADE.append(sid)
    return tid, oids


# Session ids created by _track, drained after each test. `accept_track` is the API endpoint and commits,
# which is correct for an endpoint and means the trailing `db.rollback()` in these tests undoes nothing it
# did. Rows left behind are not merely untidy: a frame surviving in the test database changes which OTHER
# tests run, because tests/test_unscoped_attrs_migration.py skips when the corpus holds no frame.
_MADE: list = []


@pytest_asyncio.fixture(autouse=True)
async def _clean_up():
    yield
    if not _MADE:
        return
    from sqlalchemy import text

    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        for sid in _MADE:
            await db.execute(text("""
                delete from review where object_id in (
                    select o.object_id from object o join frame f on f.frame_id = o.frame_id
                    where f.session_id = :s)"""), {"s": str(sid)})
            await db.execute(text("delete from object where frame_id in "
                                  "(select frame_id from frame where session_id = :s)"), {"s": str(sid)})
            await db.execute(text("delete from frame where session_id = :s"), {"s": str(sid)})
            await db.execute(text("delete from track where session_id = :s"), {"s": str(sid)})
            await db.execute(text("delete from session where session_id = :s"), {"s": str(sid)})
        await db.commit()
    _MADE.clear()


def _payload(**kw):
    from services.api.routers.tracks import AcceptTrackIn

    return AcceptTrackIn(**kw)


@pytest.mark.asyncio
async def test_accept_moves_state_and_leaves_source_alone():
    """The whole point. An interpolated box that a reviewer approves is still an interpolated box.

    If accept wrote source="human" here, every consumer that filters on it would read machine output as
    ground truth, and the rows would become un-self-healable, since source == "human" is this repo's flag
    for "an agent must not touch this".
    """
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.tracks import accept_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        srcs = ["fused", "interpolated", "interpolated", "auto_accept"]
        tid, oids = await _track(db, get_ontology(), 4, sources=srcs)
        out = await accept_track(tid, _payload(), db, await _user(db, "reviewer"))

        assert out["accepted"] == 4
        rows = (await db.execute(select(Object.state, Object.source)
                                 .where(Object.object_id.in_(oids)))).all()
        assert {s for s, _ in rows} == {"accepted"}, "every frame should have moved to accepted"
        assert sorted(src for _, src in rows) == sorted(srcs), (
            "accept must not rewrite source; it is a verdict about the box, not a claim to have drawn it")
        await db.rollback()


@pytest.mark.asyncio
async def test_accept_asserts_no_class():
    """Distinct from relabel: the class on every frame is left exactly as it was, flips included, because
    a reviewer accepting a track has not said anything about what the outliers are."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.tracks import accept_track
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, onto, 3)
        before = (await db.execute(select(Object.class_id).where(Object.object_id.in_(oids)))).scalars().all()
        await accept_track(tid, _payload(), db, await _user(db, "reviewer"))
        after = (await db.execute(select(Object.class_id).where(Object.object_id.in_(oids)))).scalars().all()
        assert sorted(before) == sorted(after)
        await db.rollback()


@pytest.mark.asyncio
async def test_an_annotator_accepting_a_track_is_clamped_to_submitted():
    """The same rule the other two review paths go through. An annotator's approval is a submission
    addressed to the QA queue, not ground truth, and a whole-track action must not be the way around it."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.tracks import accept_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, get_ontology(), 4)
        out = await accept_track(tid, _payload(state="accepted"), db, await _user(db, "annotator"))
        assert out["clamped"] is True and out["state"] == "submitted"
        states = (await db.execute(select(Object.state).where(Object.object_id.in_(oids)))).scalars().all()
        assert set(states) == {"submitted"}
        await db.rollback()


@pytest.mark.asyncio
async def test_a_humans_own_frame_is_not_overwritten():
    """skip_human, same as relabel. The one frame somebody actually drew keeps its own state, so a
    whole-track accept cannot quietly overrule a rejection a person made deliberately."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.tracks import accept_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, get_ontology(), 4,
                                 sources=["fused", "human", "fused", "fused"],
                                 states=["review", "rejected", "review", "review"])
        out = await accept_track(tid, _payload(), db, await _user(db, "reviewer"))
        assert out["accepted"] == 3
        assert len(out["skipped_human"]) == 1
        state = (await db.execute(select(Object.state)
                                  .where(Object.object_id == oids[1]))).scalar_one()
        assert state == "rejected", "the frame a person rejected must survive a whole-track accept"
        await db.rollback()


@pytest.mark.asyncio
async def test_the_origin_frame_is_left_for_the_editor_holding_it():
    """Same reason relabel excludes it: bumping the version under an open editor makes that editor's own
    next save 409 against itself."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.tracks import accept_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, get_ontology(), 4)
        await accept_track(tid, _payload(origin_object_id=str(oids[0])), db, await _user(db, "reviewer"))
        v = (await db.execute(select(Object.version).where(Object.object_id == oids[0]))).scalar_one()
        assert v == 1, "the frame the editor is holding must keep its version"
        await db.rollback()


@pytest.mark.asyncio
async def test_accept_is_revertible_as_one_run():
    """A fifty-frame verdict has to be takeable back in one move. The Review rows are the audit trail and
    answer what changed; they are not an undo, because undoing through them means fifty manual reversals
    with the operator remembering each prior value."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.runs import revert_run
    from services.api.routers.tracks import accept_track
    from services.autolabel.ontology import get_ontology

    maker = get_sessionmaker()
    async with maker() as db:
        tid, oids = await _track(db, get_ontology(), 4, sources=["fused"] * 4)
        await db.commit()
        out = await accept_track(tid, _payload(), db, await _user(db, "reviewer"))
        run_id = out["run_id"]
        assert run_id, "an accept that changed rows must produce a revertible run"

    async with maker() as db:
        await revert_run(db, run_id)
        await db.commit()

    async with maker() as db:
        rows = (await db.execute(select(Object.state, Object.source)
                                 .where(Object.object_id.in_(oids)))).all()
        assert {s for s, _ in rows} == {"review"}, f"revert should restore the prior state, got {rows}"


@pytest.mark.asyncio
async def test_a_track_too_large_to_be_real_is_refused():
    """The same cap relabel uses. A track holding more objects than the limit is more likely mis-linked
    than real, and a one-key verdict over a mis-linked track is exactly the failure to avoid."""
    from fastapi import HTTPException

    from db.session import get_sessionmaker
    from services.api.deps import MAX_TRACK_RELABEL_OBJECTS
    from services.api.routers.tracks import accept_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, _ = await _track(db, get_ontology(), MAX_TRACK_RELABEL_OBJECTS + 1)
        with pytest.raises(HTTPException) as exc:
            await accept_track(tid, _payload(), db, await _user(db, "reviewer"))
        assert exc.value.status_code == 409
        await db.rollback()
