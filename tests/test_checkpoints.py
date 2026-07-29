"""Named saves of a frame's annotations.

The interesting cases are all about not losing work: a restore that replaces an hour must leave that hour
recoverable, and a stale foreign key from a track deleted since must not take the whole restore down with it.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from db.models import AnnotationCheckpoint, Frame, Object, OntologyClass, Track
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.annotate.checkpoints import (
    create_checkpoint,
    delete_checkpoint,
    list_checkpoints,
    restore_checkpoint,
)


async def _frame_with(objects: int, *, track: bool = False):
    """A session, a frame, and n objects on it. Returns (frame_id, [object_id], track_id)."""
    async with get_sessionmaker()() as db:
        cls_id = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "sedan"))).scalar()
        if cls_id is None:
            pytest.skip("the ontology in this database has no sedan class")
        s = DbSession(vehicle_id="veh-ckpt", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=1280, height=960, quality=0.9)
        db.add(f)
        await db.flush()
        tid = None
        if track:
            t = Track(session_id=s.session_id, class_id=cls_id, first_ts_ns=0, last_ts_ns=1)
            db.add(t)
            await db.flush()
            tid = t.track_id
        ids = []
        for i in range(objects):
            o = Object(frame_id=f.frame_id, class_id=cls_id, track_id=tid,
                       bbox=[float(i), 0.0, float(i) + 10, 10.0], conf=0.8, source="fused",
                       attrs={}, state="review")
            db.add(o)
            await db.flush()
            ids.append(o.object_id)
        await db.commit()
        return f.frame_id, ids, tid


@pytest.mark.db
async def test_a_save_records_the_objects_as_they_stand():
    fid, ids, _ = await _frame_with(3)
    async with get_sessionmaker()() as db:
        c = await create_checkpoint(db, fid, name="three objects", created_by="tester")
    assert c["object_count"] == 3
    assert c["auto"] is False
    assert c["name"] == "three objects"

    async with get_sessionmaker()() as db:
        listing = await list_checkpoints(db, fid)
    assert listing["count"] == 1
    assert {i["checkpoint_id"] for i in listing["checkpoints"]} == {c["checkpoint_id"]}
    assert len(ids) == 3


@pytest.mark.db
async def test_restoring_brings_back_deleted_objects_and_removes_added_ones():
    fid, ids, _ = await _frame_with(3)
    async with get_sessionmaker()() as db:
        saved = await create_checkpoint(db, fid, name="before the mess")

    # Delete one, add one, change another. All three kinds of divergence at once.
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Object).where(Object.frame_id == fid))).scalars().all()
        await db.delete(rows[0])
        rows[1].state = "rejected"
        db.add(Object(frame_id=fid, class_id=rows[1].class_id, bbox=[99.0, 99.0, 109.0, 109.0],
                      conf=0.5, source="fused", attrs={}, state="review"))
        await db.commit()

    async with get_sessionmaker()() as db:
        out = await restore_checkpoint(db, saved["checkpoint_id"], created_by="tester")
        after = (await db.execute(select(Object).where(Object.frame_id == fid))).scalars().all()

    assert out["created"] == 1, "the deleted object comes back"
    assert out["deleted"] == 1, "the one added since is removed"
    assert out["updated"] == 2
    assert len(after) == 3
    assert all(o.state == "review" for o in after), "the changed state is put back too"


@pytest.mark.db
async def test_a_restore_saves_what_it_replaced_so_it_can_itself_be_undone():
    """The one destructive operation here. It must not be possible to lose work to the feature whose whole
    purpose is not losing work."""
    fid, _ids, _ = await _frame_with(2)
    async with get_sessionmaker()() as db:
        early = await create_checkpoint(db, fid, name="two objects")

    async with get_sessionmaker()() as db:
        db.add(Object(frame_id=fid,
                      class_id=(await db.execute(select(Object.class_id)
                                                 .where(Object.frame_id == fid))).scalar(),
                      bbox=[50.0, 50.0, 60.0, 60.0], conf=0.9, source="fused",
                      attrs={}, state="review"))
        await db.commit()

    async with get_sessionmaker()() as db:
        out = await restore_checkpoint(db, early["checkpoint_id"])
    assert out["undo_with"], "the replaced state comes back as a checkpoint id"

    # Undoing the restore is itself a restore, and it gets the third object back.
    async with get_sessionmaker()() as db:
        await restore_checkpoint(db, out["undo_with"])
        rows = (await db.execute(select(Object).where(Object.frame_id == fid))).scalars().all()
    assert len(rows) == 3, "the state the restore replaced is fully recoverable"


@pytest.mark.db
async def test_the_automatic_save_is_flagged_and_can_be_hidden():
    fid, _ids, _ = await _frame_with(1)
    async with get_sessionmaker()() as db:
        c = await create_checkpoint(db, fid, name="mine")
        await restore_checkpoint(db, c["checkpoint_id"])
        every = await list_checkpoints(db, fid, include_auto=True)
        mine_only = await list_checkpoints(db, fid, include_auto=False)

    assert every["count"] == 2
    assert mine_only["count"] == 1
    assert mine_only["checkpoints"][0]["name"] == "mine"
    assert any(x["auto"] for x in every["checkpoints"])


@pytest.mark.db
async def test_a_track_deleted_since_does_not_take_the_restore_down_with_it():
    """track_id is a foreign key. A track merged away or deleted after the save would fail the whole restore
    on a detail nobody cares about, losing the recovery to a housekeeping change."""
    fid, _ids, tid = await _frame_with(2, track=True)
    async with get_sessionmaker()() as db:
        saved = await create_checkpoint(db, fid, name="with a track")

    async with get_sessionmaker()() as db:
        # The FK is SET NULL, so deleting the track clears it off the objects. The snapshot still names it.
        await db.delete(await db.get(Track, tid))
        await db.commit()

    async with get_sessionmaker()() as db:
        out = await restore_checkpoint(db, saved["checkpoint_id"])
        rows = (await db.execute(select(Object).where(Object.frame_id == fid))).scalars().all()

    assert out["dropped_stale_tracks"] == 1
    assert len(rows) == 2, "the restore still happened"
    assert all(o.track_id is None for o in rows), "the dead reference is dropped, not resurrected"


@pytest.mark.db
async def test_deleting_a_save_leaves_the_annotations_alone():
    fid, _ids, _ = await _frame_with(2)
    async with get_sessionmaker()() as db:
        c = await create_checkpoint(db, fid, name="disposable")
        assert await delete_checkpoint(db, c["checkpoint_id"]) is True
        assert await delete_checkpoint(db, c["checkpoint_id"]) is False
        rows = (await db.execute(select(Object).where(Object.frame_id == fid))).scalars().all()
        left = (await db.execute(select(AnnotationCheckpoint)
                                 .where(AnnotationCheckpoint.frame_id == fid))).scalars().all()
    assert len(rows) == 2, "a save is a record of the annotations, not the annotations"
    assert left == []


@pytest.mark.db
async def test_a_save_of_a_frame_that_does_not_exist_is_refused():
    async with get_sessionmaker()() as db:
        with pytest.raises(ValueError, match="frame not found"):
            await create_checkpoint(db, uuid.uuid4(), name="nowhere")
