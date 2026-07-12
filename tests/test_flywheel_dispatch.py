"""Flywheel dispatch integration test: a cycle's work order is actioned into the review queue as one reversible
run. Seeds cattle objects and a corpus cycle, dispatches, and asserts the candidates are bumped to review and
stamped with the cycle provenance under a kind='flywheel' AgentRun; then reverts and asserts the prior state is
restored exactly and a human-accepted object is never touched."""

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns, seconds_to_ns
from services.agent.runs import revert_run
from services.flywheel.dispatch import dispatch_worklist


@pytest.mark.asyncio
async def test_dispatch_bumps_to_review_and_reverts():
    from db.models import AgentRun, Frame, FlywheelCycle, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cattle = next(c.id for c in onto.classes if c.name == "cattle")
    maker = get_sessionmaker()
    ts = now_ns()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    annotate_id, review_id, human_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/f.jpg",
                     width=320, height=240, quality=0.9, scene={}))
        await db.flush()
        # a candidate to bump (annotate), one already in review, and a human-accepted one that must be untouched
        db.add(Object(object_id=annotate_id, frame_id=fid, class_id=cattle, bbox=[1.0, 1.0, 30.0, 30.0],
                      conf=0.3, source="fused", state="annotate", attrs={}, provenance={}, version=1))
        db.add(Object(object_id=review_id, frame_id=fid, class_id=cattle, bbox=[1.0, 1.0, 30.0, 30.0],
                      conf=0.5, source="fused", state="review", attrs={}, provenance={}, version=1))
        db.add(Object(object_id=human_id, frame_id=fid, class_id=cattle, bbox=[1.0, 1.0, 30.0, 30.0],
                      conf=0.9, source="human", state="accepted", attrs={}, provenance={}, version=1))
        cycle = FlywheelCycle(
            label_budget=100,
            signals={"regressions": [{"slice": "cattle", "class_id": cattle, "protected": True}], "source": "corpus"},
            allocation=[{"slice": "cattle", "labels": 50, "reason": "protected"}],
            collection_tasks=[], rationale="test")
        db.add(cycle)
        await db.commit()
        cid = str(cycle.cycle_id)

    async with maker() as db:
        res = await dispatch_worklist(db, cid, per_slice_cap=300, created_by="flywheel")
        assert res["dispatched"] == 2                       # the annotate + review candidates, not the accepted one
        assert res["by_slice"]["cattle"] == 2

    async with maker() as db:
        bumped = await db.get(Object, annotate_id)
        assert bumped.state == "review"                     # annotate -> review
        assert bumped.provenance["flywheel"]["cycle_id"] == cid
        human = await db.get(Object, human_id)
        assert human.state == "accepted" and human.source == "human"   # never touched
        run = (await db.execute(select(AgentRun).where(AgentRun.kind == "flywheel"))).scalars().first()
        assert run is not None and run.status == "committed"
        run_id = run.run_id

    async with maker() as db:
        await revert_run(db, run_id)
    async with maker() as db:
        restored = await db.get(Object, annotate_id)
        assert restored.state == "annotate"                 # revert restored the prior state exactly
