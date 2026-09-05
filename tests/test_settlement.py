"""Phase 2 of the autonomy work: the settlement engine, proven on real rows.

The order mirrors the plan's own verification list: the sample-size math, the role clamp, the lot
lifecycle (plan -> human verdicts -> tally -> settle -> revert) on seeded rows, every guard's refusal,
the spot-check auto-revert, and the contamination umbrella - settlement changes no calibration input.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from core.config import get_settings
from core.timebase import now_ns, seconds_to_ns
from db.models import AgentRun, Frame, Object, SettlementLot, SettlementSpot
from db.models import Session as DbSession
from db.session import get_sessionmaker

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine
    from db.session import get_sessionmaker as _gsm

    get_engine.cache_clear()
    _gsm.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


EPOCH = "test-model-A"


async def _seed_stratum(class_name: str, n: int, *, human: int = 0, other_epoch: int = 0):
    """One session, one frame per object, n review objects whose winning proposal names EPOCH."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = onto.by_name(class_name).id
    sid = uuid.uuid4()
    start = now_ns()
    oids, human_ids, other_ids = [], [], []
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id=f"SETL-{uuid.uuid4().hex[:4]}", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        await db.flush()

        def _add(source, prov):
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start + len(oids + human_ids + other_ids),
                         cam_id="cam_f", img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[10, 10, 60, 60],
                          conf=0.8, source=source, state="review", provenance=prov, attrs={},
                          version=1))
            return oid

        win = {"proposals": [{"verdict": "agree", "class_name": class_name,
                              "model_version": EPOCH, "conf": 0.8, "path": "path_a"}]}
        other = {"proposals": [{"verdict": "agree", "class_name": class_name,
                                "model_version": "test-model-B", "conf": 0.8, "path": "path_a"}]}
        for _ in range(n):
            oids.append(_add("fused", win))
        for _ in range(human):
            human_ids.append(_add("human", win))
        for _ in range(other_epoch):
            other_ids.append(_add("fused", other))
        await db.commit()
    return sid, cid, oids, human_ids, other_ids


def _shrink(monkeypatch, *, far: float = 0.35, sample: int = 10, min_pop: int = 20,
            spot_fraction: float = 0.5):
    """Test-sized lot parameters. The real math is pinned separately in TestSampleMath."""
    import services.labelops.settlement as st

    monkeypatch.setattr(st, "MIN_POPULATION", min_pop)
    monkeypatch.setattr(st, "sample_target", lambda f, **kw: sample)
    monkeypatch.setattr(st, "tier_for", lambda cn: ("default", far))
    monkeypatch.setattr(st, "SPOT_FRACTION", spot_fraction)


async def _judge_sample(lot_id: str, *, defects: int = 0, leave: int = 0) -> None:
    """Human verdicts on a lot's sample through the real review path: rejects first, accepts after,
    `leave` crops unjudged."""
    from services.autolabel.ontology import get_ontology
    from services.review_apply import apply_review_batch

    async with get_sessionmaker()() as db:
        lot = await db.get(SettlementLot, uuid.UUID(lot_id))
        ids = [uuid.UUID(s) for s in lot.sample_object_ids]
        judgeable = ids[:len(ids) - leave]
        objs = (await db.execute(select(Object).where(
            Object.object_id.in_(judgeable)))).scalars().all()
        onto = get_ontology()
        bad, good = objs[:defects], objs[defects:]
        if bad:
            await apply_review_batch(db, bad, action="reject", onto=onto, role="reviewer",
                                     reviewer="lot-judge")
        if good:
            await apply_review_batch(db, good, action="accept", onto=onto, role="reviewer",
                                     reviewer="lot-judge")
        await db.commit()


async def _governance(settlement: bool | None = None, loop: bool | None = None) -> tuple:
    from services.govern.killswitch import get_state

    async with get_sessionmaker()() as db:
        st = await get_state(db)
        prior = (st.settlement_enabled, st.loop_enabled)
        if settlement is not None:
            st.settlement_enabled = settlement
        if loop is not None:
            st.loop_enabled = loop
        await db.commit()
    return prior


class TestSampleMath:
    def test_sample_sizes_match_the_plan_and_the_wilson_property(self):
        """~120 clean-ish verdicts for far 0.05, ~280 for 0.02, ~565 for 0.01 - the numbers stated
        to the user before any human is asked. Pinned as ranges plus the defining property."""
        from services.labelops.sampling import wilson_interval
        from services.labelops.settlement import sample_target

        for far, lo, hi in ((0.05, 100, 140), (0.02, 250, 320), (0.01, 520, 640)):
            n = sample_target(far)
            assert lo <= n <= hi, f"far {far}: n={n} outside the planned range"
            assert wilson_interval(1, n)["hi"] <= far, "n must survive one defect"
            assert wilson_interval(1, n - 25)["hi"] > far, "n must not be padded far past need"

    def test_tiers_come_from_the_pack_policy(self):
        from packs.registry import default_pack_id, get_pack
        from services.labelops.settlement import tier_for

        policy = get_pack(default_pack_id()).safety_policy
        crit = next(iter(policy.critical_class_names()))
        tier, far = tier_for(crit)
        assert tier == "critical" and far == policy.accept_far_bound(crit)


class TestRoleClamp:
    def test_a_person_asking_for_settled_gets_accepted(self):
        """'settled' means "the machine closed this under a passed lot". A person's ruling is the
        stronger claim, so it lands as accepted - and the review API can never mint machine-settled
        rows."""
        from services.review_policy import state_for

        assert state_for(None, "settled", "reviewer", None) == "accepted"
        assert state_for(None, "settled", None, None) == "accepted"

    def test_an_annotator_asking_for_settled_gets_submitted(self):
        from services.review_policy import state_for

        assert state_for(None, "settled", "annotator", None) == "submitted"


@requires_infra
def test_the_full_lot_cycle_settles_and_reverts_real_rows(monkeypatch):
    from services.govern.class_autonomy import set_level
    from services.labelops.settlement import plan_lot, revert_lot, settle_lot, tally_lot

    _shrink(monkeypatch)

    async def _flow():
        sid, cid, oids, human_ids, other_ids = await _seed_stratum("sedan", 30, human=2,
                                                                   other_epoch=3)
        prior = await _governance(settlement=True, loop=True)
        try:
            async with get_sessionmaker()() as db:
                plan = await plan_lot(db, "sedan", epoch=EPOCH, created_by="test")
            assert "error" not in plan, plan
            assert plan["sample_n"] == 10 and plan["population"] >= 30
            lot_id = plan["lot_id"]

            # completion floor: half judged is not a measurement
            await _judge_sample(lot_id, defects=0, leave=6)
            async with get_sessionmaker()() as db:
                waiting = await tally_lot(db, lot_id)
            assert waiting["status"] == "judging" and "unjudged" in waiting["detail"]

            await _judge_sample(lot_id, defects=0, leave=0)
            async with get_sessionmaker()() as db:
                res = await tally_lot(db, lot_id)
            assert res["status"] == "accepted", res
            assert res["decision"]["verdict"] == "accept" and res["defects"] == 0

            async with get_sessionmaker()() as db:
                await set_level(db, cid, 2, set_by="test", basis={"lot": lot_id})
                settled = await settle_lot(db, lot_id, created_by="test")
            assert "error" not in settled, settled
            assert settled["settled"] == 20, "30 in stratum minus the 10-crop sample"

            async with get_sessionmaker()() as db:
                lot = await db.get(SettlementLot, uuid.UUID(lot_id))
                assert lot.status == "settled" and lot.run_ids
                rows = (await db.execute(select(Object).where(
                    Object.object_id.in_(oids)))).scalars().all()
                by_state: dict[str, int] = {}
                for o in rows:
                    by_state[o.state] = by_state.get(o.state, 0) + 1
                    if o.state == "settled":
                        assert o.source == "fused", "settling never touches source; provenance is " \
                                                    "the point"
                        assert o.provenance["settlement"]["lot_id"] == lot_id
                        assert o.provenance.get("agent_run_id") in lot.run_ids
                assert by_state.get("settled") == 20
                assert by_state.get("accepted") == 10, "the sample carries the humans' rulings"

                for hid in human_ids:
                    assert (await db.get(Object, hid)).state == "review", \
                        "a human-sourced object is never settled by a machine"
                for oid_ in other_ids:
                    assert (await db.get(Object, oid_)).state == "review", \
                        "another epoch's labels are another population; the lot proves nothing " \
                        "about them"

                spots = (await db.execute(select(SettlementSpot).where(
                    SettlementSpot.lot_id == uuid.UUID(lot_id)))).scalars().all()
                assert spots, "no spot mirror means no continuous check and no auto-revert trigger"
                assert all(str(s.object_id) not in set(lot.sample_object_ids) for s in spots), \
                    "the spot mirror must not re-examine the crops the decision was made on"

            async with get_sessionmaker()() as db:
                rev = await revert_lot(db, lot_id, reason="test revert")
            assert rev["reverted"] == 20 and rev["status"] == "reverted"
            async with get_sessionmaker()() as db:
                back = (await db.execute(select(Object.state).where(
                    Object.object_id.in_(oids)))).scalars().all()
                assert sum(1 for s in back if s == "review") == 20
        finally:
            await _governance(settlement=prior[0], loop=prior[1])

    run_async(_flow())


@requires_infra
def test_a_defective_sample_rejects_the_lot(monkeypatch):
    from services.labelops.settlement import plan_lot, tally_lot

    _shrink(monkeypatch)

    async def _flow():
        await _seed_stratum("truck", 30)
        async with get_sessionmaker()() as db:
            plan = await plan_lot(db, "truck", epoch=EPOCH, created_by="test")
        assert "error" not in plan, plan
        await _judge_sample(plan["lot_id"], defects=8)
        async with get_sessionmaker()() as db:
            res = await tally_lot(db, plan["lot_id"])
        assert res["status"] == "rejected", res
        assert res["defects"] == 8
        assert "above" in res["decision"]["reason"]

    run_async(_flow())


@requires_infra
def test_every_guard_refuses_with_its_reason(monkeypatch):
    from services.govern.class_autonomy import set_level
    from services.labelops.settlement import plan_lot, settle_lot, tally_lot

    _shrink(monkeypatch)

    async def _flow():
        sid, cid, *_ = await _seed_stratum("bus", 30)
        async with get_sessionmaker()() as db:
            plan = await plan_lot(db, "bus", epoch=EPOCH, created_by="test")
            dup = await plan_lot(db, "bus", epoch=EPOCH, created_by="test")
        assert "already" in dup["error"], "one open lot per stratum"
        await _judge_sample(plan["lot_id"])
        async with get_sessionmaker()() as db:
            assert (await tally_lot(db, plan["lot_id"]))["status"] == "accepted"
            await set_level(db, cid, 2, set_by="test")

        prior = await _governance(settlement=False, loop=True)
        try:
            async with get_sessionmaker()() as db:
                off = await settle_lot(db, plan["lot_id"])
            assert "settlement_enabled is off" in off["error"]

            await _governance(settlement=True, loop=False)
            async with get_sessionmaker()() as db:
                killed = await settle_lot(db, plan["lot_id"])
            assert "kill switch" in killed["error"]

            # critical never auto-applies, whatever the switches say
            await _governance(settlement=True, loop=True)
            async with get_sessionmaker()() as db:
                lot = await db.get(SettlementLot, uuid.UUID(plan["lot_id"]))
                lot.tier = "critical"
                await db.commit()
                crit = await settle_lot(db, plan["lot_id"])
            assert "critical" in crit["error"] and "person" in crit["error"]
        finally:
            await _governance(settlement=prior[0], loop=prior[1])

    run_async(_flow())


@requires_infra
def test_population_below_the_floor_is_refused(monkeypatch):
    import services.labelops.settlement as st
    from services.labelops.settlement import plan_lot

    monkeypatch.setattr(st, "sample_target", lambda f, **kw: 5)

    async def _flow():
        await _seed_stratum("autorickshaw", 12)
        async with get_sessionmaker()() as db:
            res = await plan_lot(db, "autorickshaw", epoch=EPOCH)
        assert "error" in res and "below" in res["error"] and "2000" in res["error"].replace(",", "")

    run_async(_flow())


@requires_infra
def test_spot_check_reject_auto_reverts_and_steps_down(monkeypatch):
    from services.govern.class_autonomy import effective_level, set_level
    from services.govern.settlement_agent import maybe_spot_check
    from services.labelops.settlement import plan_lot, settle_lot, tally_lot

    _shrink(monkeypatch)

    async def _flow():
        from sqlalchemy import delete

        sid, cid, oids, *_ = await _seed_stratum("tempo", 30)
        prior = await _governance(settlement=True, loop=True)
        try:
            async with get_sessionmaker()() as db:
                await db.execute(delete(AgentRun).where(AgentRun.kind == "settlement_spot_check"))
                await db.commit()
                plan = await plan_lot(db, "tempo", epoch=EPOCH, created_by="test")
            await _judge_sample(plan["lot_id"])
            async with get_sessionmaker()() as db:
                assert (await tally_lot(db, plan["lot_id"]))["status"] == "accepted"
                await set_level(db, cid, 2, set_by="test")
                settled = await settle_lot(db, plan["lot_id"], created_by="test")
            assert settled["spots"] > 0

            # every spot verdict incorrect: the settled population is provably worse than its bound
            async with get_sessionmaker()() as db:
                spots = (await db.execute(select(SettlementSpot).where(
                    SettlementSpot.lot_id == uuid.UUID(plan["lot_id"])))).scalars().all()
                for s in spots:
                    s.human_verdict = "incorrect"
                    s.verdict_at = datetime.now(UTC)
                await db.commit()
                res = await maybe_spot_check(db)
            assert res["ran"] is True and res["breaches"] == 1

            workers = [t for t in asyncio.all_tasks()
                       if t.get_name() == "worker" and t is not asyncio.current_task()]
            await asyncio.gather(*workers)

            async with get_sessionmaker()() as db:
                lot = await db.get(SettlementLot, uuid.UUID(plan["lot_id"]))
                assert lot.status == "reverted", "a failed spot check is the one automatic revert"
                states = (await db.execute(select(Object.state).where(
                    Object.object_id.in_(oids)))).scalars().all()
                assert all(s in ("review", "accepted") for s in states), \
                    "settled rows return to review; the sample keeps its human rulings"
                lvl = await effective_level(db, cid)
                assert lvl["level"] == 0 and lvl["set_by"] == "spot_check"
                assert lvl["cooldown_until"] is not None, "re-promotion waits out the cooldown AND " \
                                                          "needs fresh lot evidence"
        finally:
            await _governance(settlement=prior[0], loop=prior[1])

    run_async(_flow())


@requires_infra
def test_settlement_changes_no_calibration_input(monkeypatch):
    """The umbrella: settle a stratum and prove every reader that means "a person ruled" reads the
    same bytes before and after. The three intended readers (filmstrip done-states, session done
    count, embedding-outlier mining) must move; everything else must not."""
    from sqlalchemy import text as sql

    from services.govern.class_autonomy import set_level
    from services.labelops.judge_calibration import build_calibration_set
    from services.labelops.precision_batch import MACHINE_STATES
    from services.labelops.settlement import plan_lot, settle_lot, tally_lot

    _shrink(monkeypatch)

    async def _snapshot(db, sid):
        fixture = select(Object.object_id).join(Frame, Frame.frame_id == Object.frame_id).where(
            Frame.session_id == sid)
        async def ids(where):
            return sorted(str(i) for i in (await db.execute(
                fixture.where(where))).scalars().all())
        cal = await build_calibration_set(db)
        return {
            "calibration_decisions": cal["decisions"] if "decisions" in cal else cal.get(
                "independent_decisions", len(cal.get("positives", []) or [])
                + len(cal.get("negatives", []) or [])),
            "gold_pool": await ids((Object.source == "human") & (Object.state == "accepted")),
            "precision_pool": await ids(Object.state.in_(MACHINE_STATES)),
            "compat_pool": await ids(Object.state == "accepted"),
            "control_pool": await ids(Object.state == "auto_accept"),
            "auditor_pool": await ids((Object.state == "auto_accept") & (Object.source != "human")),
        }

    async def _flow():
        sid, cid, oids, *_ = await _seed_stratum("suv", 30)
        prior = await _governance(settlement=True, loop=True)
        try:
            async with get_sessionmaker()() as db:
                plan = await plan_lot(db, "suv", epoch=EPOCH, created_by="test")
            await _judge_sample(plan["lot_id"])
            async with get_sessionmaker()() as db:
                assert (await tally_lot(db, plan["lot_id"]))["status"] == "accepted"
                await set_level(db, cid, 2, set_by="test")
                before = await _snapshot(db, sid)
                done_before = (await db.execute(sql(
                    "select count(*) from object o join frame f on f.frame_id=o.frame_id "
                    "where f.session_id=:s and o.state in ('accepted','auto_accept','settled')"),
                    {"s": str(sid)})).scalar_one()

                settled = await settle_lot(db, plan["lot_id"], created_by="test")
                assert settled["settled"] == 20

                after = await _snapshot(db, sid)
                done_after = (await db.execute(sql(
                    "select count(*) from object o join frame f on f.frame_id=o.frame_id "
                    "where f.session_id=:s and o.state in ('accepted','auto_accept','settled')"),
                    {"s": str(sid)})).scalar_one()

            for key in before:
                if key == "precision_pool":
                    continue
                assert before[key] == after[key], \
                    f"settlement leaked into {key}: a reader that means 'a person ruled' now reads " \
                    "machine-settled rows"
            # The machine-precision pool is the one reader that legitimately shrinks: the settled rows
            # left 'review'. What must hold is the direction - they LEFT and nothing entered - and that
            # 'settled' is not itself a draw state, or settlement would grade its own homework.
            settled_ids = {s2 for s2 in before["precision_pool"] if s2 not in after["precision_pool"]}
            assert set(after["precision_pool"]) <= set(before["precision_pool"]), \
                "settling must never ADD to the machine-precision pool"
            assert len(settled_ids) == 20, "exactly the settled remainder leaves the pool"
            assert "settled" not in MACHINE_STATES, \
                "a settled label back in the precision-sample pool would let settlement grade itself"
            assert done_after == done_before + 20, \
                "the intended readers (done counts) must see settled labels as closed - that IS the " \
                "feature"
        finally:
            await _governance(settlement=prior[0], loop=prior[1])

    run_async(_flow())
