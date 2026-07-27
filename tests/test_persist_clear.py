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
