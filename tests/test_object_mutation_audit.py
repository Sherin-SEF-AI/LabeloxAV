"""Two object mutations left no trace, and one of them destroyed the trace of everything else.

`DELETE /objects/{id}` took no user, checked no role, and wrote nothing. Deleting an object cascades its
Review rows, so the object and its entire review history left the database together: the action most worth
auditing was the one that erased the audit. Nothing afterwards could say the object had existed, let alone
who removed it.

`PUT /objects/{id}/mask` replaced a segment in place with no reviewer, no before-state and no role check,
while every other geometry change in the editor writes a Review row.

The delete record goes in AuditDecision rather than Review precisely because AuditDecision has no foreign
key to the object: a record that cascades with the thing it records is not a record.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns, seconds_to_ns
from db.models import (
    AuditDecision, Frame, Object, OntologyClass, OntologyVersion, Review, User,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology


async def _user(db, name="asha", role="annotator") -> User:
    """A real row: Review.user_id is a foreign key, so a fabricated id would fail the insert here for a
    reason that has nothing to do with what is being tested."""
    row = User(user_id=uuid.uuid4(), name=f"{name}-{uuid.uuid4().hex[:6]}", role=role)
    db.add(row)
    await db.flush()
    return row


async def _object(db) -> Object:
    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts = now_ns()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="AUDIT-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                 img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    obj = Object(object_id=uuid.uuid4(), frame_id=fid,
                 class_id=next(c.id for c in onto.classes if c.name == "rider"),
                 bbox=[10.0, 20.0, 110.0, 220.0], conf=0.42, source="fused", state="review",
                 attrs={"occluded": True}, provenance={"model": "yolo11"}, version=3)
    db.add(obj)
    await db.commit()
    return obj


@pytest.mark.asyncio
async def test_deleting_an_object_leaves_a_record_that_outlives_it():
    from services.api.routers.objects import delete_object

    async with get_sessionmaker()() as db:
        obj = await _object(db)
        oid = str(obj.object_id)
        await delete_object(oid, db=db, user=await _user(db))

    async with get_sessionmaker()() as db:
        assert await db.get(Object, uuid.UUID(oid)) is None
        rows = (await db.execute(
            select(AuditDecision).where(AuditDecision.subject == oid))).scalars().all()

    assert len(rows) == 1
    rec = rows[0]
    assert rec.decision == "delete_object"
    assert rec.actor.startswith("asha")
    # The whole state, because after the delete there is nothing left to join against.
    assert rec.rationale["bbox"] == [10.0, 20.0, 110.0, 220.0]
    assert rec.rationale["conf"] == 0.42
    assert rec.rationale["source"] == "fused"
    assert rec.rationale["attrs"] == {"occluded": True}
    assert rec.rationale["provenance"] == {"model": "yolo11"}


@pytest.mark.asyncio
async def test_a_failed_delete_does_not_leave_an_orphan_audit_row():
    """The record and the delete are one transaction. A record of a delete that did not happen is a lie
    with the same shape as the truth."""
    from services.api.routers.objects import delete_object

    async with get_sessionmaker()() as db:
        obj = await _object(db)
        oid = str(obj.object_id)
        await db.rollback()

    async with get_sessionmaker()() as db:
        with pytest.raises(Exception):
            await delete_object(str(uuid.uuid4()), db=db, user=await _user(db))
        await db.rollback()

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(AuditDecision).where(AuditDecision.subject == oid))).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_replacing_a_mask_records_who_replaced_it_and_what_was_there():
    from services.api.deps import MaskIn
    from services.api.routers.objects import update_mask

    async with get_sessionmaker()() as db:
        obj = await _object(db)
        oid = str(obj.object_id)
        before_version = obj.version
        out = await update_mask(oid, MaskIn(polygons=[[10, 20, 110, 20, 110, 220, 10, 220]]),
                                db=db, user=await _user(db, name="ravi"))

    assert out["version"] == before_version + 1, "an unversioned edit defeats the optimistic lock"

    async with get_sessionmaker()() as db:
        reviews = (await db.execute(
            select(Review).where(Review.object_id == uuid.UUID(oid)))).scalars().all()

    assert [r.action for r in reviews] == ["edit_mask"]
    r = reviews[0]
    assert r.reviewer.startswith("ravi")
    assert r.before["source"] == "fused" and r.before["conf"] == 0.42
    assert r.after["n_polygons"] == 1
    assert r.after["mask_uri"] != r.before["mask_uri"]
