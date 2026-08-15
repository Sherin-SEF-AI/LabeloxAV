"""A fifty-object bulk review was fifty manual reversals.

Bulk review writes one `Review` row per object, which is the audit trail: it answers who changed what and
what the value had been. It is not an undo. Nothing tied the batch together, so taking back a correction
applied to fifty objects meant reversing each one by hand, from memory, one at a time. The correction dialog
makes this sharper, because its whole purpose is applying one decision widely.

`AgentRun` is already the repo's revertible unit, but its generic restore refuses to touch anything whose
`source` is `human`, which is correct when undoing an agent's work over a person's and wrong here: a human
review is exactly what sets that column. Every object in the batch would be skipped. So this kind gets its
own revert, keyed on the run id stamped into provenance rather than on the source column.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import AgentRun, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.agent.runs import revert_run
from services.review_batch import KIND, change_record, record_batch

pytestmark = pytest.mark.db


async def _objects(db, n: int, *, class_id: int = 1, source: str = "auto_accept", state: str = "review"):
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="BATCH-01", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()
    ids = []
    for _ in range(n):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=class_id, bbox=[1, 1, 9, 9],
                      conf=0.5, source=source, state=state, attrs={"colour": "red"}))
        ids.append(oid)
    await db.commit()
    return ids


async def _apply(db, ids, *, to_class: int, to_state: str = "accepted"):
    """What bulk review does to each object, plus the batch record that makes it undoable."""
    changes = {}
    for oid in ids:
        obj = await db.get(Object, oid)
        changes[str(oid)] = change_record(obj)
        obj.class_id = to_class
        obj.state = to_state
        obj.source = "human"
        obj.version = (obj.version or 1) + 1
    await db.commit()
    return await record_batch(db, changes, created_by="tester", policy={"action": "reclassify"})


class TestTheBatchIsOneThing:
    async def test_a_batch_gets_a_single_run_id(self):
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 3)
            run_id = await _apply(db, ids, to_class=42)
        assert run_id is not None

    async def test_an_empty_batch_records_nothing(self):
        # A run claiming zero objects is a row nobody can act on and a revert that means nothing.
        async with get_sessionmaker()() as db:
            assert await record_batch(db, {}, created_by="tester") is None

    async def test_every_object_carries_the_run_id(self):
        """Ownership is checked against this, not against the source column, which the batch itself sets."""
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 2)
            run_id = await _apply(db, ids, to_class=42)
            for oid in ids:
                obj = await db.get(Object, oid)
                await db.refresh(obj)
                assert (obj.provenance or {}).get("agent_run_id") == run_id


class TestUndo:
    async def test_the_whole_batch_comes_back(self):
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 4, class_id=7, state="review")
            run_id = await _apply(db, ids, to_class=42, to_state="accepted")
            out = await revert_run(db, uuid.UUID(run_id))
            assert out["reverted"] == 4
            for oid in ids:
                obj = await db.get(Object, oid)
                await db.refresh(obj)
                assert obj.class_id == 7 and obj.state == "review"

    async def test_the_source_column_is_restored_too(self):
        """Otherwise the corpus quietly gains human-authored rows nobody authored, and every later query
        that trusts `source = human` inherits the lie."""
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 2, source="auto_accept")
            run_id = await _apply(db, ids, to_class=42)
            await revert_run(db, uuid.UUID(run_id))
            for oid in ids:
                obj = await db.get(Object, oid)
                await db.refresh(obj)
                assert obj.source == "auto_accept"

    async def test_attributes_are_restored(self):
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 1)
            run_id = await _apply(db, ids, to_class=42)
            obj = await db.get(Object, ids[0])
            obj.attrs = {"colour": "blue"}
            await db.commit()
            await revert_run(db, uuid.UUID(run_id))
            obj = await db.get(Object, ids[0])
            await db.refresh(obj)
            assert obj.attrs == {"colour": "red"}

    async def test_an_object_edited_afterwards_is_left_alone(self):
        """A later decision, by a person or another batch, wins. The revert reports it as skipped rather
        than rolling somebody's work back underneath them."""
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 2)
            run_id = await _apply(db, ids, to_class=42)
            later = await db.get(Object, ids[0])
            prov = dict(later.provenance or {})
            prov["agent_run_id"] = str(uuid.uuid4())      # a subsequent batch took ownership
            later.provenance = prov
            later.class_id = 99
            await db.commit()

            out = await revert_run(db, uuid.UUID(run_id))
            assert out["reverted"] == 1 and out["skipped"] == 1
            untouched = await db.get(Object, ids[0])
            await db.refresh(untouched)
            assert untouched.class_id == 99

    async def test_the_run_is_marked_reverted_so_it_cannot_be_undone_twice(self):
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 1)
            run_id = await _apply(db, ids, to_class=42)
            await revert_run(db, uuid.UUID(run_id))
            run = await db.get(AgentRun, uuid.UUID(run_id))
            await db.refresh(run)
            assert run.status == "reverted"
            with pytest.raises(ValueError):
                await revert_run(db, uuid.UUID(run_id))

    async def test_the_provenance_stamp_is_cleared_on_the_way_back(self):
        # Otherwise the object still claims to belong to a run that no longer holds it.
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 1)
            run_id = await _apply(db, ids, to_class=42)
            await revert_run(db, uuid.UUID(run_id))
            obj = await db.get(Object, ids[0])
            await db.refresh(obj)
            assert "agent_run_id" not in (obj.provenance or {})
            assert "review_batch" not in (obj.provenance or {})

    async def test_a_deleted_object_is_skipped_rather_than_crashing_the_undo(self):
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 2)
            run_id = await _apply(db, ids, to_class=42)
            await db.delete(await db.get(Object, ids[0]))
            await db.commit()
            out = await revert_run(db, uuid.UUID(run_id))
        assert out["reverted"] == 1 and out["skipped"] == 1


class TestItRoutesToItsOwnRevert:
    async def test_the_generic_restore_would_have_skipped_everything(self):
        """The reason this kind needs its own path: the generic revert refuses anything `source = human`,
        and a human review is what set it. Without the branch, reverted would be 0."""
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 3)
            run_id = await _apply(db, ids, to_class=42)
            run = await db.get(AgentRun, uuid.UUID(run_id))
            assert run.kind == KIND
            for oid in ids:
                obj = await db.get(Object, oid)
                await db.refresh(obj)
                assert obj.source == "human"
            out = await revert_run(db, uuid.UUID(run_id))
        assert out["reverted"] == 3
