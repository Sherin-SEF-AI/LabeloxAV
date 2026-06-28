"""M4.4 closed-loop governance. The pure champion gate never promotes a model that regresses a safety
class or Safe-mIoU. End to end: a challenger is promoted only when it beats the champion on gold without a
safety regression; the control sample reports a true auto-accept precision; a simulated drift breach
pauses auto-promotion; the kill switch pauses the loop and rolls back; every automated decision is
audited. Safety classes are VRU (pedestrian, child). Single asyncio.run for the DB part."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.govern.champion import champion_gate


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")

CHAMP = {"map": 0.70, "safe_miou": 0.90, "per_class": {"pedestrian": 0.80, "child": 0.78, "motorcycle": 0.65}}


def test_champion_gate_blocks_safety_regression():
    onto, cfg = get_ontology(), get_settings().phase4.govern
    # beats mAP, no safety regression -> promote
    good = {"map": 0.74, "safe_miou": 0.91, "per_class": {"pedestrian": 0.82, "child": 0.79, "motorcycle": 0.70}}
    assert champion_gate(good, CHAMP, onto, cfg)["promote"] is True
    # higher overall mAP BUT a VRU class collapses -> never promote
    unsafe = {"map": 0.80, "safe_miou": 0.91, "per_class": {"pedestrian": 0.50, "child": 0.79, "motorcycle": 0.95}}
    g = champion_gate(unsafe, CHAMP, onto, cfg)
    assert g["promote"] is False and "pedestrian" in g["regressed_safety"]
    # Safe-mIoU regression alone blocks promotion
    sm = {"map": 0.80, "safe_miou": 0.80, "per_class": {"pedestrian": 0.82, "child": 0.79, "motorcycle": 0.9}}
    assert champion_gate(sm, CHAMP, onto, cfg)["promote"] is False
    # first model (no incumbent) is promoted
    assert champion_gate(good, None, onto, cfg)["promote"] is True


@requires_infra
def test_governance_end_to_end():
    from db.models import AuditDecision, ControlSample, DriftMetric, GovernanceState, ModelRegistry
    from db.session import get_sessionmaker
    from services.govern import killswitch as K
    from services.govern.champion import evaluate_and_promote
    from services.govern.control_sample import measured_precision, record_verdict
    from services.govern.drift import run_drift_scan
    from services.govern.registry import register

    tag = uuid.uuid4().hex[:6]
    task = f"det-{tag}"  # isolate from any real registered champion (hermetic per run)
    v_champ, v_unsafe, v_good, v_paused = (f"m-champ-{tag}", f"m-unsafe-{tag}", f"m-good-{tag}", f"m-paused-{tag}")
    good = {"map": 0.74, "safe_miou": 0.91, "per_class": {"pedestrian": 0.82, "child": 0.79, "motorcycle": 0.70}}
    unsafe = {"map": 0.80, "safe_miou": 0.91, "per_class": {"pedestrian": 0.50, "child": 0.79, "motorcycle": 0.95}}

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            # reset the singleton to a clean baseline for the test
            st = await K.get_state(db)
            st.loop_enabled = st.auto_accept_enabled = st.auto_promote_enabled = True
            st.champion_version, st.paused_reason = None, None
            await db.commit()

            # first model becomes champion (no incumbent)
            await register(db, v_champ, task, CHAMP)
            r0 = await evaluate_and_promote(db, v_champ, task)
            assert r0["promoted"] is True

            # an unsafe challenger is rejected despite higher mAP
            await register(db, v_unsafe, task, unsafe)
            r1 = await evaluate_and_promote(db, v_unsafe, task)
            assert r1["promoted"] is False and "pedestrian" in r1["gate"]["regressed_safety"]

            # a good challenger is promoted, recording what it beat
            await register(db, v_good, task, good)
            r2 = await evaluate_and_promote(db, v_good, task)
            assert r2["promoted"] is True and r2["promoted_from"] == v_champ

            # control sample: 5 auto-accepted controls (real object FKs), 2 judged incorrect -> precision 0.6
            from sqlalchemy import select
            from db.models import Object
            sids = []
            obj_ids = (await db.execute(select(Object.object_id).limit(5))).scalars().all()
            for oid in obj_ids:
                cs = ControlSample(object_id=oid, was_auto_accepted=True)
                db.add(cs)
                await db.flush()
                sids.append(str(cs.sample_id))
            await db.commit()
            for i, sid in enumerate(sids):
                await record_verdict(db, sid, "incorrect" if i < 2 else "correct")
            prec = await measured_precision(db)
            assert prec["reviewed"] == 5 and prec["precision"] == 0.6  # a true, measured precision number

            # a drift breach (control precision 0.6 < floor 0.97) pauses auto-promotion
            scan = await run_drift_scan(db)
            assert "control_precision" in scan["breached"] and scan["paused"] is True
            stt = await K.get_state(db)
            assert stt.auto_promote_enabled is False

            # a further good challenger is now PAUSED, not promoted
            await register(db, v_paused, task,
                           {"map": 0.9, "safe_miou": 0.95, "per_class": {"pedestrian": 0.9, "child": 0.9, "motorcycle": 0.9}})
            r3 = await evaluate_and_promote(db, v_paused, task)
            assert r3["promoted"] is False and r3.get("paused") is True

            # kill switch pauses the loop and rolls back to the prior champion (v_champ)
            ks = await K.engage(db, "test kill", task)
            assert ks["rollback"]["rolled_back"] is True and ks["rollback"]["to"] == v_champ
            stt2 = await K.get_state(db)
            assert stt2.loop_enabled is False and stt2.champion_version == v_champ

            # every automated decision is in the audit log
            from services.govern.audit import list_audit
            decisions = {a["decision"] for a in await list_audit(db, limit=200)}
            assert {"promote", "reject", "promotion_paused", "engage"} <= decisions

            # cleanup: registry rows, controls, drift, audit for this run; reset governance
            for v in (v_champ, v_unsafe, v_good, v_paused):
                reg = await db.get(ModelRegistry, v)
                if reg:
                    await db.delete(reg)
            for sid in sids:
                cs = await db.get(ControlSample, uuid.UUID(sid))
                if cs:
                    await db.delete(cs)
            from sqlalchemy import delete
            await db.execute(delete(DriftMetric))
            st = await K.get_state(db)
            st.loop_enabled = st.auto_accept_enabled = st.auto_promote_enabled = True
            st.champion_version, st.paused_reason = None, None
            await db.commit()

    asyncio.run(run())


@requires_infra
def test_drift_pause_recovers_but_not_killswitch():
    """R1.4: a drift-induced soft pause lifts when the breach clears; a non-drift pause does not."""
    from db.session import get_sessionmaker
    from services.govern import killswitch as K

    async def run():
        async with get_sessionmaker()() as db:
            st = await K.get_state(db)
            st.loop_enabled, st.auto_promote_enabled = True, False
            st.paused_reason = "drift breach: control_precision=0.5"
            await db.commit()
            assert await K.resume_auto_promote(db) is True
            st = await K.get_state(db)
            assert st.auto_promote_enabled is True and st.paused_reason is None

            # a non-drift pause (operator/kill switch) is never auto-resumed
            st.loop_enabled, st.auto_promote_enabled = True, False
            st.paused_reason = "manual operator hold"
            await db.commit()
            assert await K.resume_auto_promote(db) is False

            st = await K.get_state(db)
            st.auto_promote_enabled, st.paused_reason = True, None
            await db.commit()

    asyncio.run(run())
