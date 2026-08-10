"""Bulk review was a convenience sweep; a keyboard grid makes it the main way labels arrive.

Single review has always refused a stale write, advanced the lock version, recorded real elapsed time and
written to the activity feed. Bulk review did none of those. That was tolerable while it ran occasionally
over a filter and is not once most of the corpus's review flows through it, because every throughput and
cost-per-label figure is built on `Review.time_spent_ms`, and a hardcoded zero reports the work as free.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns


async def _objects(db, onto, n: int, *, class_name: str = "rider"):
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    cid = next(c.id for c in onto.classes if c.name == class_name)
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts, sid, fid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="BULK-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/b.jpg",
                 width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    out = []
    for _ in range(n):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[1.0, 1.0, 50.0, 90.0],
                      conf=0.5, source="fused", state="review", attrs={}, provenance={}, version=1))
        out.append(oid)
    await db.flush()
    return out


@pytest.mark.asyncio
async def test_batch_time_is_divided_across_its_members():
    """A grid triaging sixty crops in a minute must report a minute, not zero and not sixty minutes."""
    from sqlalchemy import select

    from db.models import Object, Review
    from db.session import get_sessionmaker
    from services.api.deps import BulkReviewIn
    from services.api.routers.review import bulk_review
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oids = await _objects(db, get_ontology(), 4)
        await bulk_review(BulkReviewIn(object_ids=[str(o) for o in oids], action="confirm",
                                       state="accepted", reviewer="grid", time_spent_ms=8000), db, None)
        times = (await db.execute(select(Review.time_spent_ms)
                                  .where(Review.object_id.in_(oids)))).scalars().all()
        assert times and all(t == 2000 for t in times), f"expected 8000/4 per object, got {times}"
        assert all(o.state == "accepted" for o in
                   (await db.execute(select(Object).where(Object.object_id.in_(oids)))).scalars())
        await db.rollback()


@pytest.mark.asyncio
async def test_the_lock_version_advances_so_other_clients_notice():
    """Without this a bulk edit is invisible to every optimistic check and gets overwritten back."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import BulkReviewIn
    from services.api.routers.review import bulk_review
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oids = await _objects(db, get_ontology(), 2)
        await bulk_review(BulkReviewIn(object_ids=[str(o) for o in oids], action="confirm",
                                       state="accepted", reviewer="grid"), db, None)
        versions = (await db.execute(select(Object.version)
                                     .where(Object.object_id.in_(oids)))).scalars().all()
        assert all(v == 2 for v in versions), f"version must advance from 1, got {versions}"
        await db.rollback()


@pytest.mark.asyncio
async def test_a_stale_member_is_skipped_and_named_not_clobbered():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import BulkReviewIn
    from services.api.routers.review import bulk_review
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oids = await _objects(db, get_ontology(), 3)
        contended = await db.get(Object, oids[1])
        contended.version = 9          # somebody else edited it since the grid loaded
        await db.flush()

        out = await bulk_review(BulkReviewIn(
            object_ids=[str(o) for o in oids], action="confirm", state="accepted", reviewer="grid",
            expected_versions={str(o): 1 for o in oids}), db, None)

        assert out["updated"] == 2, "one contended object must not discard the other two verdicts"
        assert [s["object_id"] for s in out["skipped_stale"]] == [str(oids[1])]
        assert out["skipped_stale"][0] == {"object_id": str(oids[1]), "expected": 1, "current": 9}
        assert (await db.get(Object, oids[1])).state == "review", "the stale object is untouched"
        await db.rollback()


@pytest.mark.asyncio
async def test_an_object_that_vanished_is_reported_rather_than_silently_dropped():
    from db.session import get_sessionmaker
    from services.api.deps import BulkReviewIn
    from services.api.routers.review import bulk_review
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oids = await _objects(db, get_ontology(), 2)
        ghost = str(uuid.uuid4())
        out = await bulk_review(BulkReviewIn(object_ids=[str(oids[0]), ghost, str(oids[1])],
                                             action="confirm", state="accepted"), db, None)
        assert out["updated"] == 2
        assert out["skipped_missing"] == [ghost]
        await db.rollback()


@pytest.mark.asyncio
async def test_without_expected_versions_nothing_is_locked():
    """The parameter is opt-in, so existing callers keep working exactly as they did."""
    from db.models import Object
    from db.session import get_sessionmaker
    from services.api.deps import BulkReviewIn
    from services.api.routers.review import bulk_review
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oids = await _objects(db, get_ontology(), 2)
        obj = await db.get(Object, oids[0])
        obj.version = 42
        await db.flush()
        out = await bulk_review(BulkReviewIn(object_ids=[str(o) for o in oids],
                                             action="confirm", state="accepted"), db, None)
        assert out["updated"] == 2 and out["skipped_stale"] == []
        await db.rollback()


@pytest.mark.asyncio
async def test_the_batch_writes_one_activity_entry_not_one_per_object():
    """Sixty identical feed rows is not a record of what somebody did, it buries everything either side."""
    from sqlalchemy import func, select

    from db.models import ActivityEvent
    from db.session import get_sessionmaker
    from services.api.deps import BulkReviewIn
    from services.api.routers.review import bulk_review
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        before = (await db.execute(select(func.count()).select_from(ActivityEvent))).scalar_one()
        oids = await _objects(db, get_ontology(), 5)
        await bulk_review(BulkReviewIn(object_ids=[str(o) for o in oids], action="confirm",
                                       state="accepted", reviewer="grid"), db, None)
        after = (await db.execute(select(func.count()).select_from(ActivityEvent))).scalar_one()
        assert after == before + 1, f"expected one entry for the batch, got {after - before}"
        await db.rollback()
