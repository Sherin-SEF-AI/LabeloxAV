"""Filling an attribute nobody has ever filled, with the track as the unit where the attribute allows it.

The write path for attributes was already complete and revertible and had one caller: the modal that opens
after somebody has made a correction. What was missing was the prior question, where the attributes are
absent, and the answer measured over the corpus is "nearly everywhere": 282,061 objects are in scope for
`load_type` and 0 carry it, 139,613 for `occupant_count` and 0 carry it, so `triple_riding`, which is
derived from it, is empty too.

Two of these tests exist because of specific ways this has gone wrong before.

The JSONB one. `~Object.attrs.has_key(k)` is NULL when `attrs` itself is NULL, and a NULL predicate
excludes the row, so the obvious spelling of "missing this attribute" silently skips every object with no
attributes at all. No `object.attrs` is NULL today, so this guards a shape the schema permits rather than
repairing a live defect; it is pinned because the same predicate on a column that IS mostly NULL,
`Frame.scene`, already cost this repo a sweep that reported `remaining: 0` after scoring 1,780 of 41,752.

The other is fill-versus-overwrite. A track-wide answer touching every frame must not be able to replace a
value somebody set deliberately on one of them.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


# Session ids created by _seed, drained after each test. `attr_apply` is the API endpoint and commits,
# which is correct for an endpoint and means the trailing `db.rollback()` in these tests undoes nothing it
# did. Rows left behind are not merely untidy: a frame surviving in the test database changes which OTHER
# tests run, because tests/test_unscoped_attrs_migration.py skips when the corpus holds no frame. Twenty
# leaked rows silently un-skipped five tests that then failed for reasons unrelated to this file.
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


async def _seed_null(db, object_ids):
    """Force `attrs` to a real NULL, past the ORM default that would otherwise write `{}`.

    Raw SQL rather than `update(Object).values(attrs=None)`, which silently does nothing here: the column
    carries `default=dict`, and the ORM update path applies it, so the statement runs, reports two rows
    affected, and leaves both as `{}`. That is exactly the failure this test exists to catch, so the setup
    must not be able to fall into it.
    """
    await db.execute(text("update object set attrs = null where object_id = any(:ids)"),
                     {"ids": list(object_ids)})


async def _purge(session_id):
    """Remove everything a committing test created, in foreign-key order.

    The two revert tests below have to commit, so they cannot lean on the rollback every other test here
    uses. Deleting only the objects is not enough: the frames survive, and a frame surviving in the test
    database changes which OTHER tests run. `tests/test_unscoped_attrs_migration.py` skips when the corpus
    holds no frame, so leaving one behind silently un-skips five tests that were never meant to run in
    that state, and they fail for reasons that have nothing to do with this file.
    """
    from sqlalchemy import text

    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        await db.execute(text("""
            delete from review where object_id in (
                select o.object_id from object o join frame f on f.frame_id = o.frame_id
                where f.session_id = :s)"""), {"s": str(session_id)})
        await db.execute(text("delete from object where frame_id in "
                              "(select frame_id from frame where session_id = :s)"), {"s": str(session_id)})
        await db.execute(text("delete from frame where session_id = :s"), {"s": str(session_id)})
        await db.execute(text("delete from track where session_id = :s"), {"s": str(session_id)})
        await db.execute(text("delete from session where session_id = :s"), {"s": str(session_id)})
        await db.commit()


async def _user(db, role: str = "reviewer"):
    from db.models import User

    u = User(user_id=uuid.uuid4(), name=f"sweeper-{uuid.uuid4().hex[:6]}", role=role)
    db.add(u)
    await db.flush()
    return u


async def _seed(db, onto, *, class_name="sedan", tracks=2, per_track=3, attrs=None, areas=None):
    """`tracks` tracks of `per_track` objects each. `areas` scales each track's boxes so the
    largest-box representative is predictable."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion, Track
    from db.models import Session as DbSession

    cid = onto.by_name(class_name).id
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    ts, sid = now_ns(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="SWP-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(tracks * per_track + 1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()

    out = []
    for t in range(tracks):
        tid = uuid.uuid4()
        db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts,
                     last_ts_ns=ts + seconds_to_ns(per_track), trajectory={}, id_switch_flags={},
                     tracker_version="test", intents={}))
        await db.flush()
        scale = (areas or [10] * tracks)[t]
        oids = []
        for i in range(per_track):
            fid, oid = uuid.uuid4(), uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(t * per_track + i),
                         cam_id="cam_f", img_uri=f"s3://x/{t}-{i}.jpg", width=1920, height=1080,
                         quality=0.9, scene={}))
            await db.flush()
            # The first member of each track is the biggest, so the representative is knowable.
            side = float(scale * (per_track - i))
            db.add(Object(object_id=oid, frame_id=fid, track_id=tid, class_id=cid,
                          bbox=[0.0, 0.0, side, side], conf=0.5, source="fused", state="review",
                          attrs=(attrs or {}).get((t, i)) or {}, provenance={}, version=1))
            oids.append(oid)
        out.append((tid, oids))
    await db.flush()
    _MADE.append(sid)
    return sid, out


@pytest.mark.asyncio
async def test_an_object_whose_attrs_are_null_counts_as_missing():
    """The JSONB trap, spelled out on the shape that triggers it.

    `attrs IS NULL` and `attrs = '{}'` both mean the attribute is unanswered, but only the second is found
    by `NOT (attrs ? 'k')`: the first makes the predicate NULL, and a NULL predicate excludes the row. A
    queue built the naive way reports a small number of objects to work on and silently hides the rest.

    `object.attrs` is nullable and 0 of 578,436 rows are NULL today, because the ORM default writes `{}`.
    So this pins a guard against a shape the schema allows rather than repairing a live defect - which is
    why the row has to be forced to NULL below instead of just being left unset.
    """
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelops.attr_sweep import sweep_queue

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, tracks=2, per_track=2)
        # Two of the four get a genuinely NULL attrs; the other two keep the default empty dict.
        await _seed_null(db, [tracks[0][1][0], tracks[1][1][0]])
        q = await sweep_queue(db, onto, attr="load_type", session_id=sid, unit="object")
        assert q["remaining"] == 4, (
            f"expected all 4 objects to count as missing load_type, got {q['remaining']}; "
            "attrs IS NULL is the case the naive `NOT (attrs ? k)` drops")
        await db.rollback()


@pytest.mark.asyncio
async def test_a_track_constant_attribute_offers_one_crop_per_track():
    """One answer per object, not one per frame. `load_type` is what the truck is carrying, and asking
    about the same truck fifty times is one answer and forty-nine chances to disagree with it."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelops.attr_sweep import sweep_queue

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, tracks=3, per_track=4, areas=[10, 30, 20])
        q = await sweep_queue(db, onto, attr="load_type", session_id=sid)

        assert q["unit"] == "track", "load_type is declared track_constant, so the unit should follow"
        assert len(q["items"]) == 3, "one representative per track"
        assert {i["covers"] for i in q["items"]} == {4}, "each answer covers the whole track"
        # Largest box first, both in choosing the representative and in ordering the page: a crop you
        # cannot make out is not a question anybody can answer.
        assert [i["covers"] for i in q["items"]] == [4, 4, 4]
        reps = {i["track_id"]: i["object_id"] for i in q["items"]}
        for tid, oids in tracks:
            assert reps[str(tid)] == str(oids[0]), "the representative should be the track's largest box"
        await db.rollback()


@pytest.mark.asyncio
async def test_a_per_frame_attribute_stays_per_frame():
    """`occlusion` is how much of the object this camera can see and changes every frame, so the unit
    must not silently become the track."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelops.attr_sweep import sweep_queue

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, _ = await _seed(db, onto, tracks=2, per_track=3)
        q = await sweep_queue(db, onto, attr="occlusion", session_id=sid)
        assert q["unit"] == "object"
        assert len(q["items"]) == 6 and all(i["covers"] == 1 for i in q["items"])
        await db.rollback()


@pytest.mark.asyncio
async def test_one_answer_lands_on_every_frame_of_the_track():
    """The leverage the mode exists for."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.attrsweep import SweepApplyIn, attr_apply
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, tracks=2, per_track=5)
        tid, oids = tracks[0]
        out = await attr_apply(SweepApplyIn(attr="load_type", value="construction_material",
                                            unit="track", ids=[str(tid)]),
                               db, await _user(db))
        assert out["updated"] == 5, f"one answer should cover all 5 frames, got {out['updated']}"
        vals = (await db.execute(select(Object.attrs).where(Object.object_id.in_(oids)))).scalars().all()
        assert all(v.get("load_type") == "construction_material" for v in vals)
        # And nothing outside the track moved.
        other = (await db.execute(select(Object.attrs)
                                  .where(Object.object_id.in_(tracks[1][1])))).scalars().all()
        assert all(not (v or {}).get("load_type") for v in other)
        await db.rollback()


@pytest.mark.asyncio
async def test_a_sweep_fills_and_does_not_overwrite():
    """A track-wide answer must not replace a value somebody set deliberately on one frame. Re-running a
    sweep has to be safe, because a queue that shrinks as you work it will be re-run."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.attrsweep import SweepApplyIn, attr_apply
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, tracks=1, per_track=4,
                                  attrs={(0, 2): {"load_type": "livestock"}})
        tid, oids = tracks[0]
        out = await attr_apply(SweepApplyIn(attr="load_type", value="goods", unit="track",
                                            ids=[str(tid)]), db, await _user(db))
        assert out["updated"] == 3, "the frame that already had an answer should have been left alone"
        kept = (await db.execute(select(Object.attrs).where(Object.object_id == oids[2]))).scalar_one()
        assert kept["load_type"] == "livestock"

        # ...unless the caller says so, which is how a wrong track-wide answer gets corrected.
        out2 = await attr_apply(SweepApplyIn(attr="load_type", value="goods", unit="track",
                                             ids=[str(tid)], overwrite=True), db, await _user(db))
        assert out2["updated"] == 4
        now = (await db.execute(select(Object.attrs).where(Object.object_id == oids[2]))).scalar_one()
        assert now["load_type"] == "goods"
        await db.rollback()


@pytest.mark.asyncio
async def test_sweeping_occupant_count_derives_triple_riding():
    """`triple_riding` is a reading of `occupant_count`, never a second opinion about it. The sweep goes
    through the same derive step every other write path does, so answering one answers both."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.routers.attrsweep import SweepApplyIn, attr_apply
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, class_name="motorcycle", tracks=1, per_track=3)
        tid, oids = tracks[0]
        await attr_apply(SweepApplyIn(attr="occupant_count", value=4, unit="track", ids=[str(tid)]),
                         db, await _user(db))
        vals = (await db.execute(select(Object.attrs).where(Object.object_id.in_(oids)))).scalars().all()
        assert all(v["occupant_count"] == 4 for v in vals)
        assert all(v["triple_riding"] is True for v in vals), "three or more is the legal threshold"
        await db.rollback()


@pytest.mark.asyncio
async def test_a_derived_attribute_cannot_be_swept_directly():
    """Two independently settable fields produce objects that say three occupants and not triple riding,
    and no consumer can tell which to believe."""
    from fastapi import HTTPException

    from db.session import get_sessionmaker
    from services.api.routers.attrsweep import SweepApplyIn, attr_apply
    from services.autolabel.ontology import get_ontology
    from services.labelops.attr_sweep import sweep_queue

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, class_name="motorcycle", tracks=1, per_track=2)
        with pytest.raises(HTTPException) as exc:
            await attr_apply(SweepApplyIn(attr="triple_riding", value=True, unit="track",
                                          ids=[str(tracks[0][0])]), db, await _user(db))
        assert exc.value.status_code == 400
        with pytest.raises(ValueError):
            await sweep_queue(db, onto, attr="triple_riding", session_id=sid)
        await db.rollback()


@pytest.mark.asyncio
async def test_a_value_outside_the_ontology_is_refused_with_the_reason():
    """The sweep's keys come from the ontology, so a wrong one means a client bug rather than a typo, and
    the response has to be specific enough to find it."""
    from fastapi import HTTPException

    from db.session import get_sessionmaker
    from services.api.routers.attrsweep import SweepApplyIn, attr_apply
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, tracks=1, per_track=2)
        with pytest.raises(HTTPException) as exc:
            await attr_apply(SweepApplyIn(attr="load_type", value="antimatter", unit="track",
                                          ids=[str(tracks[0][0])]), db, await _user(db))
        assert exc.value.status_code == 400
        assert "attr_errors" in exc.value.detail
        await db.rollback()


@pytest.mark.asyncio
async def test_re_sweeping_a_finished_track_says_so_rather_than_reporting_success():
    """"Did nothing" and "done" are different answers and a mode that works a shrinking queue will hit
    the first one."""
    from db.session import get_sessionmaker
    from services.api.routers.attrsweep import SweepApplyIn, attr_apply
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, tracks = await _seed(db, onto, tracks=1, per_track=3)
        tid = tracks[0][0]
        body = SweepApplyIn(attr="load_type", value="goods", unit="track", ids=[str(tid)])
        await attr_apply(body, db, await _user(db))
        again = await attr_apply(body, db, await _user(db))
        assert again["updated"] == 0 and again["run_id"] is None
        assert "already carries" in again["reason"]
        await db.rollback()


@pytest.mark.asyncio
async def test_coverage_reports_scope_set_and_missing():
    """The "what work exists" view. In scope minus set is missing, computed from a single scan rather than
    from a negated JSONB predicate, so the trap in the first test cannot reappear here."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelops.attr_sweep import coverage

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, _ = await _seed(db, onto, tracks=2, per_track=3,
                             attrs={(0, 0): {"load_type": "goods"}})
        cov = await coverage(db, onto, session_id=sid)
        by = {e["attribute"]: e for e in cov["attributes"]}
        assert by["load_type"]["in_scope"] == 6
        assert by["load_type"]["set"] == 1
        assert by["load_type"]["missing"] == 5
        assert by["load_type"]["track_constant"] is True
        # occupant_count does not apply to a four_wheeler, so it must contribute nothing at all rather
        # than showing six objects of phantom work.
        assert by["occupant_count"]["in_scope"] == 0
        assert by["load_type"]["classes"][0]["class_name"] == "sedan"
        await db.rollback()


@pytest.mark.asyncio
async def test_a_sweep_is_revertible_as_one_run():
    """A page of track-wide answers can touch thousands of objects. It has to come back in one move."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.runs import revert_run
    from services.api.routers.attrsweep import SweepApplyIn, attr_apply
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    maker = get_sessionmaker()
    async with maker() as db:
        _sid, tracks = await _seed(db, onto, tracks=1, per_track=4)
        await db.commit()
        tid, oids = tracks[0]
        out = await attr_apply(SweepApplyIn(attr="load_type", value="water", unit="track",
                                            ids=[str(tid)]), db, await _user(db))
        run_id = out["run_id"]
        assert run_id

    async with maker() as db:
        await revert_run(db, run_id)
        await db.commit()

    async with maker() as db:
        vals = (await db.execute(select(Object.attrs)
                                 .where(Object.object_id.in_(oids)))).scalars().all()
        assert all(not (v or {}).get("load_type") for v in vals), f"revert left {vals}"
