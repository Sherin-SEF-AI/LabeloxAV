"""Drift Investigator: the hypothesis synthesis (pure) and the end-to-end localization of a forced
control-precision breach to the class that caused it."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from core.timebase import now_ns, seconds_to_ns


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


def test_synthesize_names_the_driving_class_and_proposes_relabel():
    from services.agent.drift_investigator import _synthesize

    findings = [{"metric": "control_precision", "precision": 0.6, "floor": 0.97,
                 "worst_classes": [("autorickshaw", 8), ("e_auto", 2)],
                 "worst_scenes": [("rain", 6)], "sessions": ["s1", "s2"],
                 "common_factor": {"vehicle": "DASHCAM-01"}}]
    hypothesis, action = _synthesize(findings)
    assert "autorickshaw" in hypothesis and "DASHCAM-01" in hypothesis
    assert action["kind"] == "narrow_relabel" and action["target_class"] == "autorickshaw"


async def _seed_incorrect_controls(class_name: str, n: int = 6):
    from db.models import ControlSample, Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from sqlalchemy import delete
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    maker = get_sessionmaker()
    cid = next(c.id for c in onto.classes if c.name == class_name)
    ts = now_ns()
    async with maker() as db:
        await db.execute(delete(ControlSample))   # deterministic precision for this run
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        sid = uuid.uuid4()
        db.add(DbSession(session_id=sid, vehicle_id="DASHCAM-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/f.jpg", width=320,
                     height=240, quality=0.9, scene={"weather": "rain"}))
        await db.flush()
        for _ in range(n):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[1.0, 1.0, 20.0, 20.0], conf=0.98,
                          source="auto_accept", state="auto_accept", attrs={}, provenance={}, version=1))
            await db.flush()
            db.add(ControlSample(object_id=oid, was_auto_accepted=True, human_verdict="incorrect"))
        await db.commit()
    return class_name


@requires_infra
def test_investigates_control_precision_breach():
    from db.models import AgentRun
    from db.session import get_sessionmaker
    from services.agent.drift_investigator import investigate
    from services.govern.drift import run_drift_scan

    run_async(_seed_incorrect_controls("autorickshaw", n=6))

    async def _flow():
        async with get_sessionmaker()() as db:
            drift = await run_drift_scan(db)          # control precision = 0 -> breach
        assert "control_precision" in drift["breached"]
        run_id = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(AgentRun(run_id=run_id, kind="drift_investigator", scope={}, status="running", policy={},
                            counts={}, changes={}, critic={}, created_by="test"))
            await db.commit()
        await investigate(run_id, drift)
        async with get_sessionmaker()() as db:
            run = await db.get(AgentRun, run_id)
        rep = run.counts
        assert run.status == "committed"
        cp = next(f for f in rep["findings"] if f["metric"] == "control_precision")
        assert any(c[0] == "autorickshaw" for c in cp["worst_classes"])
        assert "autorickshaw" in rep["hypothesis"]
        assert rep["proposed_action"]["kind"] == "narrow_relabel"

    run_async(_flow())
