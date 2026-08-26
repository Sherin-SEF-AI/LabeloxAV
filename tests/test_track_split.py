"""Cutting tracks where they stop being one object.

59.2% of tracks contain a centre jump of more than a quarter of the frame width. Everything downstream reads
a track as one object - gap filling interpolates between its endpoints, class corrections propagate along
it, events span it - so a track that is really five objects corrupts all of them at once.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _track(db, boxes, *, classes=None, dt_s=0.33):
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.models import Track
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    t0, sid, tid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="SPL-1", start_ts_ns=t0,
                     end_ts_ns=t0 + seconds_to_ns(60), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Track(track_id=tid, session_id=sid, class_id=onto.by_name("sedan").id,
                 first_ts_ns=t0, last_ts_ns=t0 + seconds_to_ns(60)))
    await db.flush()
    for i, bb in enumerate(boxes):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t0 + int(i * dt_s * 1e9), cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9))
        await db.flush()
        cname = (classes or {}).get(i, "sedan")
        db.add(Object(frame_id=fid, track_id=tid, class_id=onto.by_name(cname).id,
                      bbox=[float(v) for v in bb], conf=0.5, source="fused", state="review"))
    await db.commit()
    return tid


def _drift(n, x0=100, step=25):
    """A well-behaved track: a box moving steadily across the frame."""
    return [[x0 + i * step, 300, x0 + i * step + 90, 400] for i in range(n)]


class TestFindingTheCuts:
    @pytest.mark.asyncio
    async def test_a_continuous_track_is_left_alone(self):
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts

        async with get_sessionmaker()() as db:
            tid = await _track(db, _drift(8))
            cuts, n = await _cuts(db, tid)
        assert cuts == [] and n == 8

    @pytest.mark.asyncio
    async def test_a_teleport_is_a_cut(self):
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts

        async with get_sessionmaker()() as db:
            tid = await _track(db, _drift(4) + [[1700, 300, 1790, 400]] + _drift(3, x0=1700, step=20))
            cuts, _n = await _cuts(db, tid)
        assert len(cuts) == 1, cuts

    @pytest.mark.asyncio
    async def test_a_class_change_alone_never_cuts(self):
        """The detector renames one object between frames constantly - a single receding vehicle here is
        truck, rider, autorickshaw, suv and motorcycle on five consecutive frames. Cutting on that would
        shred correct tracks."""
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts

        async with get_sessionmaker()() as db:
            tid = await _track(db, _drift(6),
                               classes={1: "truck", 2: "rider", 3: "autorickshaw", 4: "bus"})
            cuts, _n = await _cuts(db, tid)
        assert cuts == [], cuts

    @pytest.mark.asyncio
    async def test_a_track_too_short_to_judge_is_left_alone(self):
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 300, 190, 400], [1700, 300, 1790, 400]])
            cuts, _n = await _cuts(db, tid)
        assert cuts == []


class TestSplitting:
    @pytest.mark.asyncio
    async def test_the_pieces_are_each_continuous(self):
        """The property that matters: after the cut, nothing downstream can interpolate across a join."""
        from db.models import AgentRun
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts, split_one

        async with get_sessionmaker()() as db:
            tid = await _track(db, _drift(4) + _drift(4, x0=1700, step=15))
            rid = uuid.uuid4()
            db.add(AgentRun(run_id=rid, kind="track_split", status="running", scope={}, policy={},
                            counts={}, changes={}, critic={}))
            await db.commit()
        async with get_sessionmaker()() as db:
            cuts, _n = await _cuts(db, tid)
            assert cuts
            await split_one(db, tid, cuts, rid)
            await db.commit()
        async with get_sessionmaker()() as db:
            run = await db.get(AgentRun, rid)
            remaining = (await _cuts(db, tid))[0]
            for t in run.policy["created_tracks"]:
                remaining += (await _cuts(db, uuid.UUID(t)))[0]
        assert remaining == [], "a piece still contains a discontinuity"

    @pytest.mark.asyncio
    async def test_every_object_survives_the_split(self):
        from sqlalchemy import select

        from db.models import AgentRun, Object
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts, split_one

        async with get_sessionmaker()() as db:
            tid = await _track(db, _drift(4) + _drift(4, x0=1700, step=15))
            rid = uuid.uuid4()
            db.add(AgentRun(run_id=rid, kind="track_split", status="running", scope={}, policy={},
                            counts={}, changes={}, critic={}))
            await db.commit()
        async with get_sessionmaker()() as db:
            await split_one(db, tid, (await _cuts(db, tid))[0], rid)
            await db.commit()
        async with get_sessionmaker()() as db:
            run = await db.get(AgentRun, rid)
            ids = [tid, *[uuid.UUID(t) for t in run.policy["created_tracks"]]]
            n = len((await db.execute(select(Object.object_id)
                                      .where(Object.track_id.in_(ids)))).scalars().all())
        assert n == 8, "a split must move objects, never lose them"

    @pytest.mark.asyncio
    async def test_it_writes_no_review_rows(self):
        """`review` means a person ruled. Three hundred thousand machine splits in it would corrupt every
        reader of that table the same way a VLM verdict written there would."""
        from sqlalchemy import func, select

        from db.models import AgentRun, Review
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts, split_one

        async with get_sessionmaker()() as db:
            before = (await db.execute(select(func.count(Review.review_id)))).scalar()
            tid = await _track(db, _drift(4) + _drift(4, x0=1700, step=15))
            rid = uuid.uuid4()
            db.add(AgentRun(run_id=rid, kind="track_split", status="running", scope={}, policy={},
                            counts={}, changes={}, critic={}))
            await db.commit()
        async with get_sessionmaker()() as db:
            await split_one(db, tid, (await _cuts(db, tid))[0], rid)
            await db.commit()
            after = (await db.execute(select(func.count(Review.review_id)))).scalar()
        assert after == before


class TestRevert:
    @pytest.mark.asyncio
    async def test_it_restores_every_object_and_removes_the_tracks_it_made(self):
        from sqlalchemy import select

        from db.models import AgentRun, Object, Track
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts, revert_split, split_one

        async with get_sessionmaker()() as db:
            tid = await _track(db, _drift(4) + _drift(4, x0=1700, step=15))
            rid = uuid.uuid4()
            db.add(AgentRun(run_id=rid, kind="track_split", status="running", scope={}, policy={},
                            counts={}, changes={}, critic={}))
            await db.commit()
        async with get_sessionmaker()() as db:
            await split_one(db, tid, (await _cuts(db, tid))[0], rid)
            run = await db.get(AgentRun, rid)
            run.status = "committed"
            await db.commit()
        async with get_sessionmaker()() as db:
            res = await revert_split(db, await db.get(AgentRun, rid))
        assert res["restored"] == 4 and res["tracks_removed"] >= 1

        async with get_sessionmaker()() as db:
            back = (await db.execute(select(Object.object_id)
                                     .where(Object.track_id == tid))).scalars().all()
            run = await db.get(AgentRun, rid)
            leftover = [t for t in run.policy["created_tracks"]
                        if await db.get(Track, uuid.UUID(t)) is not None]
        assert len(back) == 8, "every object must return to its original track"
        assert leftover == [], "revert left empty tracks behind"

    @pytest.mark.asyncio
    async def test_it_leaves_alone_an_object_something_else_now_owns(self):
        from db.models import AgentRun, Object
        from db.session import get_sessionmaker
        from services.quality.track_split import _cuts, revert_split, split_one

        async with get_sessionmaker()() as db:
            tid = await _track(db, _drift(4) + _drift(4, x0=1700, step=15))
            rid = uuid.uuid4()
            db.add(AgentRun(run_id=rid, kind="track_split", status="running", scope={}, policy={},
                            counts={}, changes={}, critic={}))
            await db.commit()
        async with get_sessionmaker()() as db:
            await split_one(db, tid, (await _cuts(db, tid))[0], rid)
            run = await db.get(AgentRun, rid)
            run.status = "committed"
            # A later run takes ownership of one moved object.
            oid = uuid.UUID(next(iter(run.changes)))
            obj = await db.get(Object, oid)
            obj.provenance = {**(obj.provenance or {}), "agent_run_id": str(uuid.uuid4())}
            await db.commit()
        async with get_sessionmaker()() as db:
            res = await revert_split(db, await db.get(AgentRun, rid))
        assert res["skipped"] >= 1, res
