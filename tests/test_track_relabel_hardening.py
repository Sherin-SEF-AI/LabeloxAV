"""A class correction that stopped at the frame it was made on, and the third review path that skipped QA.

Measured before this change: 413 tracks carried an unambiguous human class, and of the 44,097 objects on
them only 5,798 had it. The median track is 93 frames and the median number of frames a person actually
touched is 1, so 86.9% of every correction ever made was sitting on one frame while the other 92 kept the
detector's guess.

The endpoint that fixes a whole track already existed and was reachable from one page nobody visits. It was
also the one bulk write in the repo with no optimistic lock, no undo, no attribute revalidation and no role
clamp, so the first job was to stop it being a way around the QA workflow and the second was to call it
from the editor.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _user(db, role: str, name: str = "tester"):
    """A real row, because Review.user_id carries a foreign key into app_user: a SimpleNamespace with a
    fresh uuid passes every assertion in this file and fails at the flush."""
    from db.models import User

    u = User(user_id=uuid.uuid4(), name=f"{name}-{uuid.uuid4().hex[:6]}", role=role)
    db.add(u)
    await db.flush()
    return u


async def _track(db, onto, n: int, *, class_name: str = "sedan", source: str = "fused"):
    """One track across n frames, one object per frame, which is what the corpus actually looks like:
    exactly one object per frame in 11,286 of 11,287 tracks."""
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
    db.add(DbSession(session_id=sid, vehicle_id="TRK-1", start_ts_ns=ts,
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
                      bbox=[1.0, 1.0, 50.0, 90.0], conf=0.5, source=source, state="review",
                      attrs={}, provenance={}, version=1))
        oids.append(oid)
    await db.flush()
    return tid, oids


@pytest.mark.asyncio
async def test_an_annotator_cannot_accept_a_whole_track():
    """The security half. services/review_policy.py says the state rule is reached by every review path;
    this one wrote payload.state straight onto the object, so an annotator could confirm ninety frames and
    skip the QA step the whole two-stage workflow is built on."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, get_ontology(), 4)
        out = await relabel_track(tid, RelabelTrackIn(class_name="minivan", state="accepted"), db,
                                  await _user(db, "annotator"))
        assert out["clamped"] is True, "an annotator asking for accepted must be told it was clamped"
        assert out["state"] == "submitted"
        states = (await db.execute(select(Object.state).where(Object.object_id.in_(oids)))).scalars().all()
        assert set(states) == {"submitted"}, f"annotator wrote {set(states)}, expected submitted"
        await db.rollback()


@pytest.mark.asyncio
async def test_a_reviewer_still_gets_what_they_asked_for():
    """The clamp is a ceiling, not a refusal: the same request from a reviewer is honoured."""
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, _ = await _track(db, get_ontology(), 3)
        out = await relabel_track(tid, RelabelTrackIn(class_name="minivan", state="accepted"), db,
                                  await _user(db, "reviewer"))
        assert out["state"] == "accepted" and out["clamped"] is False
        await db.rollback()


@pytest.mark.asyncio
async def test_the_lock_version_advances_on_every_relabelled_frame():
    """Without it the track relabel is invisible to every other client's optimistic check, so an editor
    holding one of those frames overwrites the whole thing back on its next save."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, get_ontology(), 5)
        await relabel_track(tid, RelabelTrackIn(class_name="minivan"), db, await _user(db, "reviewer"))
        versions = (await db.execute(select(Object.version)
                                     .where(Object.object_id.in_(oids)))).scalars().all()
        assert all(v == 2 for v in versions), f"version must advance from 1, got {versions}"
        await db.rollback()


@pytest.mark.asyncio
async def test_the_frame_the_human_edited_is_left_alone():
    """The editor saves that object itself and is holding its version. Bumping it here would make the
    editor's own next save 409 against its own propagation."""
    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, get_ontology(), 4)
        origin = oids[0]
        out = await relabel_track(tid, RelabelTrackIn(class_name="minivan", origin_object_id=str(origin)),
                                  db, await _user(db, "reviewer"))
        assert out["relabeled"] == 3, "the origin frame must not be counted or touched"
        obj = await db.get(Object, origin)
        assert obj.version == 1 and obj.source == "fused", "the origin object was modified"
        await db.rollback()


@pytest.mark.asyncio
async def test_the_propagated_frames_are_not_marked_human():
    """92 of 93 frames were never looked at. Marking them human makes every consumer that filters on it
    read machine output as ground truth, and makes the rows un-self-healable, because source == "human" is
    this repo's "an agent must not touch this" flag."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, get_ontology(), 4)
        await relabel_track(tid, RelabelTrackIn(class_name="minivan"), db, await _user(db, "reviewer"))
        sources = (await db.execute(select(Object.source)
                                    .where(Object.object_id.in_(oids)))).scalars().all()
        assert set(sources) == {"propagated"}, f"expected propagated, got {set(sources)}"
        await db.rollback()


@pytest.mark.asyncio
async def test_a_frame_somebody_already_labelled_is_left_alone():
    """The same rule temporal_repair applies and the correction dialog shows as an "already" badge: bulk
    tooling that quietly overwrites a person's decision is worse than bulk tooling that does nothing."""
    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        tid, oids = await _track(db, onto, 4)
        mine = await db.get(Object, oids[1])
        mine.source = "human"
        mine.class_id = next(c.id for c in onto.classes if c.name == "bus")
        await db.flush()

        out = await relabel_track(tid, RelabelTrackIn(class_name="minivan"), db, await _user(db, "reviewer"))
        assert out["relabeled"] == 3
        assert str(oids[1]) in out["skipped_human"], "the human frame must be named, not silently skipped"
        again = await db.get(Object, oids[1])
        assert again.class_id != next(c.id for c in onto.classes if c.name == "minivan")
        await db.rollback()


@pytest.mark.asyncio
async def test_a_move_that_changes_what_kind_of_thing_it_is_is_refused():
    """Commit 38b28dd's 1,047 buses relabelled into a bus shelter. A confidence threshold cannot catch a
    confident wrong answer; the ontology can."""
    from fastapi import HTTPException

    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, _ = await _track(db, get_ontology(), 3, class_name="sedan")
        with pytest.raises(HTTPException) as err:
            await relabel_track(tid, RelabelTrackIn(class_name="road"), db, await _user(db, "reviewer"))
        assert err.value.status_code == 409
        assert "refused" in err.value.detail
        await db.rollback()


@pytest.mark.asyncio
async def test_a_reviewer_can_force_the_refused_move():
    """Without an escape the guard turns a past bad relabel into a permanently unfixable one: the 1,047
    wrong rows could never be moved back."""
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        tid, _ = await _track(db, get_ontology(), 3, class_name="sedan")
        out = await relabel_track(tid, RelabelTrackIn(class_name="road", force=True), db, await _user(db, "reviewer"))
        assert out["relabeled"] == 3
        await db.rollback()


@pytest.mark.asyncio
async def test_attributes_that_do_not_apply_to_the_new_class_are_dropped_and_named():
    """services/quality/attr_audit.py names this path by function as a corpus-corruption source. Dropped
    rather than refused, because one stale attribute on one frame must not make a 93-frame track unfixable,
    and change_record captured the prior attrs so it comes back on revert."""
    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        tid, oids = await _track(db, onto, 3, class_name="motorcycle")
        # helmet applies to a two-wheeler and not to a sedan.
        assert "helmet" in (onto.attrs_for_class(onto.by_name("motorcycle").id) or [])
        for oid in oids:
            o = await db.get(Object, oid)
            o.attrs = {"helmet": "yes"}
        await db.flush()

        out = await relabel_track(tid, RelabelTrackIn(class_name="sedan"), db, await _user(db, "reviewer"))
        if "helmet" not in (onto.attrs_for_class(onto.by_name("sedan").id) or []):
            assert out["attrs_dropped"], "an inapplicable attribute must be dropped and reported"
            kept = await db.get(Object, oids[0])
            assert "helmet" not in (kept.attrs or {})
        await db.rollback()


@pytest.mark.asyncio
async def test_a_track_relabel_is_revertible_to_its_exact_prior_state():
    """It was the one bulk write in the repo with no undo: taking back a 93-frame relabel meant 93 manual
    reversals with the operator remembering each prior value."""
    from sqlalchemy import select

    from db.models import AgentRun, Object
    from db.session import get_sessionmaker
    from services.api.deps import RelabelTrackIn
    from services.api.routers.tracks import relabel_track
    from services.autolabel.ontology import get_ontology
    from services.review_batch import revert_batch

    async with get_sessionmaker()() as db:
        onto = get_ontology()
        before_cid = onto.by_name("sedan").id
        tid, oids = await _track(db, onto, 4, class_name="sedan")
        out = await relabel_track(tid, RelabelTrackIn(class_name="minivan"), db, await _user(db, "reviewer"))
        assert out["run_id"], "a track relabel must produce a revertible run"

        run = await db.get(AgentRun, uuid.UUID(out["run_id"]))
        await revert_batch(db, run)

        rows = (await db.execute(select(Object).where(Object.object_id.in_(oids)))).scalars().all()
        assert all(o.class_id == before_cid for o in rows), "class did not come back"
        assert all(o.source == "fused" for o in rows), "source must return to the machine label"
        await db.rollback()
