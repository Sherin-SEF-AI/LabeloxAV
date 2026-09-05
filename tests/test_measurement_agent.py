"""Phase 1 of the autonomy work: measurements refresh on staleness, and the control queue has a door.

`stale_precision_classes` is the predicate that decides what a night re-judges, so it is pinned
directly: never-measured outranks stale, fresh is left alone, and the bound is the bound. The triage
`control` scope is the door to the 601-pending queue: an auto-accepted control object is invisible to
the default review filter (that is HOW the queue starved), so the scope must ignore the states filter
and say why each crop is there.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select as _sa_select

from core.timebase import now_ns
from db.models import ControlSample, Frame, MachineVerdict, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker

pytestmark = pytest.mark.db


async def _one_object(db) -> uuid.UUID:
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="MEAS-01", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()
    oid = uuid.uuid4()
    db.add(Object(object_id=oid, frame_id=fid, class_id=1, bbox=[1, 1, 9, 9], conf=0.9,
                  source="fused", state="review"))
    await db.flush()
    return oid


async def test_never_measured_outranks_stale_and_fresh_is_left_alone(monkeypatch):
    import services.labelops.class_precision as cp
    from services.agent.measurement_agent import stale_precision_classes

    tag = uuid.uuid4().hex[:6]
    fresh_cls, stale_cls, never_cls = f"fresh-{tag}", f"stale-{tag}", f"never-{tag}"

    async def fake_targets(db, **kw):
        return [{"class_name": c, "n_objects": 20_000} for c in (fresh_cls, stale_cls, never_cls)]

    monkeypatch.setattr(cp, "class_targets", fake_targets)

    async with get_sessionmaker()() as db:
        oid = await _one_object(db)
        for cls, age_days in ((fresh_cls, 1), (stale_cls, 30)):
            db.add(MachineVerdict(object_id=oid, judge="vlm", provider="ollama", model_version="t",
                                  verdict="correct", batch_id=cp.batch_id_for(cls), ts_ns=now_ns(),
                                  created_at=datetime.now(UTC) - timedelta(days=age_days)))
        await db.commit()

        out = await stale_precision_classes(db, bound_days=14, limit=5)
        names = [r["class_name"] for r in out]
        assert never_cls in names and stale_cls in names
        assert fresh_cls not in names, "re-judging a fresh class spends the night's VLM budget on nothing"
        assert names[0] == never_cls, "an unmeasured class is worse than a stale one: its number is not " \
                                      "old, it does not exist"
        assert next(r for r in out if r["class_name"] == never_cls)["reason"] == "never measured"
        assert "30d ago" in next(r for r in out if r["class_name"] == stale_cls)["reason"]


async def test_the_limit_caps_a_nights_judging(monkeypatch):
    import services.labelops.class_precision as cp
    from services.agent.measurement_agent import stale_precision_classes

    async def fake_targets(db, **kw):
        return [{"class_name": f"c{i}-{uuid.uuid4().hex[:4]}", "n_objects": 20_000} for i in range(9)]

    monkeypatch.setattr(cp, "class_targets", fake_targets)
    async with get_sessionmaker()() as db:
        out = await stale_precision_classes(db, limit=3)
    assert len(out) == 3, "batch-by-batch is the contract: the sweep converges across nights, it does " \
                          "not own the card for one"


async def test_the_control_scope_reaches_objects_the_default_filter_hides():
    from services.api.routers.triage import triage

    async with get_sessionmaker()() as db:
        sess = DbSession(session_id=uuid.uuid4(), vehicle_id=f"CTRLQ-{uuid.uuid4().hex[:4]}",
                         start_ts_ns=0, end_ts_ns=1, ontology_version="test")
        db.add(sess)
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                     img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
        await db.flush()
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=1, bbox=[1, 1, 9, 9], conf=0.95,
                      source="auto_accept", state="auto_accept"))
        await db.flush()
        db.add(ControlSample(object_id=oid, was_auto_accepted=True))
        await db.commit()

        plain = await triage(db=db, states="review,annotate", session_id=str(sess.session_id))
        assert not any(r.object_id == str(oid) for r in plain), \
            "auto_accept is outside the default states filter, which is exactly why the queue starved"

        scoped = await triage(db=db, states="review,annotate", session_id=str(sess.session_id),
                              control=True)
        mine = [r for r in scoped if r.object_id == str(oid)]
        assert mine, "the control scope must reach what the default filter hides"
        assert mine[0].flags[0].code == "control_sample", "the reviewer is told this verdict scores the gate"

        # a judged sample leaves the queue
        cs = (await db.execute(
            _sa_select(ControlSample).where(ControlSample.object_id == oid)
        )).scalars().first()
        cs.human_verdict = "correct"
        await db.commit()
        after = await triage(db=db, states="review,annotate", session_id=str(sess.session_id),
                             control=True)
        assert not any(r.object_id == str(oid) for r in after)
