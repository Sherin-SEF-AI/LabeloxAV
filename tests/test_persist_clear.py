"""Section 1.6: an autolabel re-run clears the derived machine rows (interpolated/propagated/relabel) as well
as fused/auto_accept, so no stale artifact of a replaced machine pass lingers in the corpus. Human-reviewed
objects and recall candidates awaiting review are never touched.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.autolabel.persist import _clear_machine_objects

pytestmark = pytest.mark.db


async def test_rerun_clears_derived_machine_rows_keeps_human_and_recall():
    onto = get_ontology()
    cid = onto.by_name("pedestrian").id
    async with get_sessionmaker()() as db:
        sess = DbSession(session_id=uuid.uuid4(), vehicle_id="T", start_ts_ns=0, end_ts_ns=1,
                         ontology_version=onto.version)
        db.add(sess)
        frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="c",
                      img_uri="s3://x", width=10, height=10)
        db.add(frame)
        await db.flush()
        for src in ("fused", "auto_accept", "interpolated", "propagated", "relabel", "human", "recall"):
            db.add(Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=cid,
                          bbox=[1.0, 1.0, 2.0, 2.0], conf=0.5, source=src, state="review"))
        await db.commit()

        await _clear_machine_objects(db, get_object_store(), SimpleNamespace(frame_id=frame.frame_id))
        await db.commit()

        remaining = set((await db.execute(
            select(Object.source).where(Object.frame_id == frame.frame_id))).scalars().all())
    assert remaining == {"human", "recall"}


async def test_rerun_keeps_machine_rows_a_person_decided_on():
    """Re-classing a track leaves the rows "propagated" or "fused" with a review row beside them. A re-run
    keyed on source alone deleted 1,650 such corrections on the live corpus before this was caught."""
    from db.models import Review

    onto = get_ontology()
    cid = onto.by_name("pedestrian").id
    async with get_sessionmaker()() as db:
        sess = DbSession(session_id=uuid.uuid4(), vehicle_id="T", start_ts_ns=0, end_ts_ns=1,
                         ontology_version=onto.version)
        db.add(sess)
        frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="c",
                      img_uri="s3://x", width=10, height=10)
        db.add(frame)
        await db.flush()
        reclassed, accepted, plain = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        db.add(Object(object_id=reclassed, frame_id=frame.frame_id, class_id=cid, bbox=[1.0, 1.0, 2.0, 2.0],
                      conf=0.5, source="propagated", state="review"))
        db.add(Object(object_id=accepted, frame_id=frame.frame_id, class_id=cid, bbox=[3.0, 3.0, 4.0, 4.0],
                      conf=0.5, source="fused", state="accepted"))
        db.add(Object(object_id=plain, frame_id=frame.frame_id, class_id=cid, bbox=[5.0, 5.0, 6.0, 6.0],
                      conf=0.5, source="fused", state="review"))
        await db.flush()
        db.add(Review(object_id=reclassed, reviewer="admin", action="reclassify_track", before={}, after={},
                      ts_ns=now_ns()))
        await db.commit()

        await _clear_machine_objects(db, get_object_store(), SimpleNamespace(frame_id=frame.frame_id))
        await db.commit()
        remaining = set((await db.execute(
            select(Object.object_id).where(Object.frame_id == frame.frame_id))).scalars().all())
    assert remaining == {reclassed, accepted}


async def test_a_detection_on_top_of_kept_work_is_not_written_twice():
    from core.bus import EventBus
    from services.autolabel.persist import persist_frame_objects

    onto = get_ontology()
    ped, car = onto.by_name("pedestrian").id, onto.by_name("sedan").id
    async with get_sessionmaker()() as db:
        sess = DbSession(session_id=uuid.uuid4(), vehicle_id="T", start_ts_ns=0, end_ts_ns=1,
                         ontology_version=onto.version)
        db.add(sess)
        frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="c",
                      img_uri="s3://x", width=100, height=100)
        db.add(frame)
        await db.flush()
        # A person corrected this object to sedan; the model still calls it a pedestrian.
        db.add(Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=car,
                      bbox=[10.0, 10.0, 40.0, 40.0], conf=1.0, source="human", state="accepted"))
        await db.commit()

        def det(box):
            return SimpleNamespace(mask=None, obj=SimpleNamespace(
                object_id=None, class_id=ped, class_name="pedestrian", conf=0.4, attrs={}, state=_State(),
                bbox=SimpleNamespace(as_list=lambda b=box: b),
                provenance=SimpleNamespace(model_dump=lambda mode=None: {})))

        fm = SimpleNamespace(frame_id=frame.frame_id, session_id=sess.session_id, width=100, height=100)
        bus = EventBus()
        await persist_frame_objects(db, get_object_store(), bus, fm,
                                    [det([11.0, 11.0, 40.0, 41.0]), det([60.0, 60.0, 80.0, 90.0])])
        await db.commit()
        rows = (await db.execute(select(Object.source, Object.bbox).where(
            Object.frame_id == frame.frame_id))).all()
    assert sorted(r[0] for r in rows) == ["fused", "human"]
    assert any(r[1][0] == 60.0 for r in rows if r[0] == "fused")


class _State:
    value = "review"

    def __eq__(self, other):
        return False
