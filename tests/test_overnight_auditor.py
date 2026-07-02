"""Overnight Auditor: the token budget, the suspect-queue + report logic against a stubbed VLM, and the
once-per-day scheduling marker."""

from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
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


def test_token_budget():
    from services.agent.runtime.budget import TokenBudget

    b = TokenBudget(3)
    assert not b.exhausted and b.remaining == 3
    b.charge(); b.charge(2)
    assert b.exhausted and b.remaining == 0
    assert b.as_dict() == {"max_calls": 3, "used": 3, "remaining": 0}


async def _seed_auto_accept():
    from db.models import ControlSample, Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from sqlalchemy import delete

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    # deterministic sample: no other auto-accepts or control samples left over from sibling tests
    async with maker() as db:
        await db.execute(delete(ControlSample))
        await db.execute(delete(Object).where(Object.state == "auto_accept"))
        await db.commit()
    sid, fid, oid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(7).integers(20, 230, size=(240, 320, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=320, height=240,
                     quality=0.9, scene={"weather": "rain", "time_of_day": "night"}))
        db.add(Object(object_id=oid, frame_id=fid, class_id=sedan, bbox=[10.0, 10.0, 120.0, 120.0], conf=0.97,
                      source="fused", state="auto_accept", attrs={}, provenance={}, version=1))
        await db.commit()
    return str(fid), str(oid), sedan


def _stub_vlm(monkeyfree_to_class: str):
    """Force the VLM verifier to confidently disagree, calling every sampled sedan a `to_class`."""
    import services.autolabel.grounding as grounding
    import services.autolabel.paths.path_c_qwen3vl as pc

    async def _supported():
        return None

    class _FakeVerifier:
        def __init__(self, *a, **k):
            pass

        def verify_object(self, img, bbox, class_id, votes=None):
            return pc.VlmResult(class_name=monkeyfree_to_class, confident=True, votes=1, agreement=1.0)

    grounding.supported_concept_ids = _supported
    pc.VlmVerifier = _FakeVerifier
    pc.make_vlm_client = lambda settings=None: object()


@requires_infra
def test_audit_queues_vlm_suspects_and_reports_reversibly():
    from db.models import AgentRun, Object
    from db.session import get_sessionmaker
    from services.agent.overnight_auditor import run_audit
    from services.agent.runs import revert_run

    fid, oid, sedan = run_async(_seed_auto_accept())
    _stub_vlm("autorickshaw")

    async def _flow():
        run_id = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(AgentRun(run_id=run_id, kind="overnight_auditor", scope={}, status="running", policy={},
                            counts={}, changes={}, critic={}, created_by="test"))
            await db.commit()
        await run_audit(run_id, sample_size=50, vlm_calls=10, since_hours=48)
        async with get_sessionmaker()() as db:
            run = await db.get(AgentRun, run_id)
            rep = run.counts
            obj = await db.get(Object, uuid.UUID(oid))
        assert run.status == "committed"
        assert rep["vlm_checked"] >= 1 and rep["vlm_disagreements"] >= 1
        assert rep["suspects_queued"] >= 1
        assert any(m["from"] == "sedan" and m["to"] == "autorickshaw" for m in rep["confusion_movers"])
        assert obj.state == "review" and obj.provenance.get("agent_run_id") == str(run_id)
        # reversible: revert restores the auto_accept state exactly
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, run_id)
        assert rev["reverted"] >= 1
        async with get_sessionmaker()() as db:
            obj2 = await db.get(Object, uuid.UUID(oid))
        assert obj2.state == "auto_accept"

    run_async(_flow())


@requires_infra
def test_maybe_run_nightly_is_once_per_day():
    from db.session import get_sessionmaker
    from services.agent.overnight_auditor import maybe_run_nightly

    from sqlalchemy import delete

    from db.models import AgentRun

    run_async(_seed_auto_accept())
    _stub_vlm("autorickshaw")

    async def _flow():
        async with get_sessionmaker()() as db:   # clean slate: no auditor run recorded for today yet
            await db.execute(delete(AgentRun).where(AgentRun.kind == "overnight_auditor"))
            await db.commit()
        async with get_sessionmaker()() as db:
            first = await maybe_run_nightly(db)
        async with get_sessionmaker()() as db:
            second = await maybe_run_nightly(db)
        assert first["ran"] is True
        assert second["ran"] is False and "already ran" in second["reason"]

    run_async(_flow())
