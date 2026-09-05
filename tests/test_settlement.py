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


# ---- WP1: the sequential rule on real rows ----
# `_shrink(sample=10)` makes the cap 10, so every lot above draws its first increment AT the cap and
# decides by Wilson alone, which is the legacy path and stays covered. The tests below raise the cap
# so the sequential machinery has room: increments, the completed-prefix tally, the early accept,
# the draw-order grid, the spot writer, and the migration round-trip.


async def _permutation(class_id: int, epoch: str, n: int, *, exclude: list[str] = ()) -> list[str]:
    """The first n judgeable review-state crops of the stratum in the engine's own permutation,
    after `exclude`. Judged crops leave the review state, so a later increment is the head of the
    permutation over what remains, which is the same order restricted to the same set."""
    from sqlalchemy import text as sql

    from services.autolabel.ontology import get_ontology
    from services.labelops.settlement import _EPOCH_SQL, MIN_SIDE_PX, SAMPLE_SALT

    cname = get_ontology().by_id(class_id).name
    async with get_sessionmaker()() as db:
        return [str(i) for i in (await db.execute(sql(f"""
            select o.object_id from object o
            where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
              and least(o.bbox[3] - o.bbox[1], o.bbox[4] - o.bbox[2]) >= :minside
              and {_EPOCH_SQL} = :epoch
              and not (o.object_id::text = any(cast(:have as text[])))
            order by md5(o.object_id::text || :salt) limit :n"""),
            {"cid": class_id, "class_name": cname, "epoch": epoch, "minside": MIN_SIDE_PX,
             "salt": SAMPLE_SALT, "n": n, "have": list(exclude)})).scalars().all()]


@requires_infra
def test_a_clean_sequential_lot_accepts_on_its_first_increment(monkeypatch):
    """Cap 60, first increment 25, all clean: the SPRT crosses its accept bound and Wilson agrees, so
    the lot accepts after 25 verdicts instead of 60. The decision names both rules."""
    from services.labelops.settlement import SPRT_INCREMENT, plan_lot, tally_lot

    _shrink(monkeypatch, sample=60)

    async def _flow():
        sid, cid, oids, *_ = await _seed_stratum("sedan", 80)
        await _clear_stratum(cid)
        async with get_sessionmaker()() as db:
            plan = await plan_lot(db, "sedan", epoch=EPOCH, created_by="test")
        assert "error" not in plan, plan
        try:
            assert plan["rule"] == "sprt" and plan["cap_n"] == 60
            assert plan["sample_n"] == SPRT_INCREMENT == 25
            assert plan["human_minutes_estimate"] == round(25 / 10)
            assert (await _permutation(cid, EPOCH, 25)) == (await _lot_ids(plan["lot_id"])), \
                "the first increment is the head of the stratum's permutation"

            await _judge_sample(plan["lot_id"])
            async with get_sessionmaker()() as db:
                res = await tally_lot(db, plan["lot_id"])
                lot = await db.get(SettlementLot, uuid.UUID(plan["lot_id"]))
            assert res["status"] == "accepted", res
            d = res["decision"]
            assert d["rule"] == "sprt" and d["sprt"]["verdict"] == "accept" \
                and d["wilson"]["verdict"] == "accept"
            assert d["n"] == 25 and d["defects"] == 0 and d["increments_complete"] == 1
            assert lot.llr is not None and lot.llr <= lot.sprt["bound_accept"]
            assert lot.sprt["trajectory"][-1]["n"] == 25
            assert len(lot.increments) == 1 and lot.increments[0]["added"] == 25
        finally:
            await _drop_lot(plan.get("lot_id"))

    run_async(_flow())


async def _lot_ids(lot_id: str) -> list[str]:
    async with get_sessionmaker()() as db:
        return list((await db.get(SettlementLot, uuid.UUID(lot_id))).sample_object_ids)


async def _clear_stratum(class_id: int) -> None:
    """Open lots a crashed earlier run left on the stratum would refuse the next plan."""
    async with get_sessionmaker()() as db:
        for lot in (await db.execute(select(SettlementLot).where(
                SettlementLot.class_id == class_id, SettlementLot.model_epoch == EPOCH,
                SettlementLot.status.in_(("sampling", "judging", "accepted"))))).scalars().all():
            await db.delete(lot)
        await db.commit()


async def _drop_lot(lot_id: str | None) -> None:
    """A lot left accepted or judging blocks the next plan on its stratum; tests that stop short of
    settle-and-revert remove theirs."""
    if not lot_id:
        return
    async with get_sessionmaker()() as db:
        lot = await db.get(SettlementLot, uuid.UUID(lot_id))
        if lot is not None:
            await db.delete(lot)
            await db.commit()


@requires_infra
def test_a_partial_increment_never_tallies_and_continue_draws_the_next_prefix(monkeypatch):
    """Judged 20 of 25: nothing tallies (floor 0.9). Judged 25 with 5 defects at far 0.35: the SPRT
    says continue, the engine reports inconclusive, and the top-up draws the NEXT 25 of the same
    permutation. While that increment is half judged the next one is refused, and the tally over
    the completed prefix is unchanged by the partial one."""
    from services.labelops.settlement import plan_lot, tally_lot, top_up_lot

    _shrink(monkeypatch, sample=100)

    async def _flow():
        sid, cid, oids, *_ = await _seed_stratum("sedan", 120)
        await _clear_stratum(cid)
        async with get_sessionmaker()() as db:
            plan = await plan_lot(db, "sedan", epoch=EPOCH, created_by="test")
        assert "error" not in plan and plan["cap_n"] == 100, plan
        lot_id = plan["lot_id"]
        try:

            await _judge_sample(lot_id, defects=5, leave=5)      # 20 of 25 judged
            async with get_sessionmaker()() as db:
                res = await tally_lot(db, lot_id)
            assert res["status"] == "judging" and "waiting" in res["detail"], res
            assert "decision" not in res

            await _judge_sample(lot_id, defects=5)               # all 25 judged, 5 defects
            async with get_sessionmaker()() as db:
                res = await tally_lot(db, lot_id)
            d = res["decision"]
            assert res["status"] == "judging" and d["verdict"] == "inconclusive", res
            assert d["sprt"]["verdict"] == "continue" and d["rule"] == "sprt"
            assert d["increments_complete"] == 1 and d["next_increment"] == 25
            llr_after_one = d["sprt"]["llr"]

            first = await _lot_ids(lot_id)
            expected_next = await _permutation(cid, EPOCH, 25, exclude=first)
            async with get_sessionmaker()() as db:
                top = await top_up_lot(db, lot_id)
            assert top["added"] == 25 and top["increments"] == 2 and top["sample_total"] == 50, top
            assert (await _lot_ids(lot_id)) == [*first, *expected_next], \
                "the second increment is the next 25 of the same permutation, appended in order"

            # Judge only the first 10 of the second increment: its 25 are 40% judged, so the prefix
            # is still one increment and the llr does not move; the next draw is refused.
            ids = await _lot_ids(lot_id)
            await _judge_ids(ids[25:35], defects=0)
            async with get_sessionmaker()() as db:
                res = await tally_lot(db, lot_id)
                top = await top_up_lot(db, lot_id)
            assert res["decision"]["increments_complete"] == 1
            assert res["decision"]["sprt"]["llr"] == llr_after_one
            assert "still being judged" in top.get("error", ""), top

            # Complete the second increment clean: 5 defects in 50 at far 0.35 accepts (both rules).
            await _judge_ids(ids[35:50], defects=0)
            async with get_sessionmaker()() as db:
                res = await tally_lot(db, lot_id)
                lot = await db.get(SettlementLot, uuid.UUID(lot_id))
            assert res["status"] == "accepted", res
            assert res["decision"]["n"] == 50 and res["decision"]["increments_complete"] == 2
            assert len(lot.sprt["trajectory"]) == 2 and lot.topups == 0, \
                "sequential increments are not top-ups; the counter is for the post-cap rule"
        finally:
            await _drop_lot(lot_id)

    run_async(_flow())


async def _judge_ids(ids: list[str], *, defects: int) -> None:
    from services.autolabel.ontology import get_ontology
    from services.review_apply import apply_review_batch

    async with get_sessionmaker()() as db:
        objs = (await db.execute(select(Object).where(
            Object.object_id.in_([uuid.UUID(s) for s in ids])))).scalars().all()
        by = {str(o.object_id): o for o in objs}
        ordered = [by[s] for s in ids]
        onto = get_ontology()
        if ordered[:defects]:
            await apply_review_batch(db, ordered[:defects], action="reject", onto=onto,
                                     role="reviewer", reviewer="lot-judge")
        if ordered[defects:]:
            await apply_review_batch(db, ordered[defects:], action="accept", onto=onto,
                                     role="reviewer", reviewer="lot-judge")
        await db.commit()


@requires_infra
def test_five_straight_defects_reject_a_sequential_lot_early(monkeypatch):
    """The saving the rule was built for: a bad class fails on its first increment. Cap 100 at far
    0.05 (the real default tier) would have asked for 100 verdicts; 25 with 8 defects rejects."""
    from services.labelops.settlement import plan_lot, tally_lot

    _shrink(monkeypatch, far=0.05, sample=100)

    async def _flow():
        sid, cid, oids, *_ = await _seed_stratum("sedan", 120)
        await _clear_stratum(cid)
        async with get_sessionmaker()() as db:
            plan = await plan_lot(db, "sedan", epoch=EPOCH, created_by="test")
        await _judge_sample(plan["lot_id"], defects=8)
        async with get_sessionmaker()() as db:
            res = await tally_lot(db, plan["lot_id"])
        assert res["status"] == "rejected", res
        assert res["decision"]["sprt"]["verdict"] == "reject"
        assert res["decision"]["n"] == 25 < 100

    run_async(_flow())


@requires_infra
def test_at_the_cap_the_fixed_rule_decides_verbatim(monkeypatch):
    """Cap 10 (the legacy shrink): the first increment IS the cap, and the decision is
    `acceptance_decision` byte for byte, with the SPRT recorded beside it for the record."""
    from services.labelops.sampling import acceptance_decision
    from services.labelops.settlement import plan_lot, tally_lot

    _shrink(monkeypatch)

    async def _flow():
        sid, cid, oids, *_ = await _seed_stratum("sedan", 30)
        await _clear_stratum(cid)
        async with get_sessionmaker()() as db:
            plan = await plan_lot(db, "sedan", epoch=EPOCH, created_by="test")
        assert plan["sample_n"] == plan["cap_n"] == 10
        await _judge_sample(plan["lot_id"], defects=2)
        async with get_sessionmaker()() as db:
            res = await tally_lot(db, plan["lot_id"])
        d = res["decision"]
        assert d["rule"] == "wilson_at_cap"
        expected = acceptance_decision(2, 10, max_defect_rate=0.35)
        assert d["verdict"] == expected["verdict"] and d["wilson"] == expected
        assert "sprt" in d

    run_async(_flow())


@requires_infra
def test_the_grid_serves_a_settlement_batch_in_draw_order_and_nothing_else_changes(monkeypatch):
    """Triage ranks by (1-conf)*rarity*boost everywhere; a `settle-` batch is served in the order it
    was drawn so every judged prefix is a random sample. A batch with any other cycle id keeps the
    formula exactly."""
    from sqlalchemy import text as sql

    from services.api.routers.triage import _why_and_priority, triage
    from services.autolabel.ontology import get_ontology
    from services.labelops.settlement import plan_lot

    _shrink(monkeypatch, sample=60)

    async def _flow():
        sid, cid, oids, *_ = await _seed_stratum("sedan", 80)
        await _clear_stratum(cid)
        async with get_sessionmaker()() as db:
            # make the confidences differ so the formula would reorder the batch
            await db.execute(sql("update object set conf = random() * 0.5 + 0.4 "
                                 "where object_id = any(cast(:ids as uuid[]))"),
                             {"ids": [str(o) for o in oids]})
            await db.commit()
            plan = await plan_lot(db, "sedan", epoch=EPOCH, created_by="test")
            assert "error" not in plan, plan
            # no session filter: the stratum spans every session that seeded this epoch
            rows = await triage(db=db, states="review", flywheel=plan["batch_id"], limit=200)
        drawn = await _lot_ids(plan["lot_id"])
        await _drop_lot(plan["lot_id"])
        served = [r.object_id for r in rows]
        assert served == drawn, "a settlement batch is served in draw order"
        assert all(r.flags[0].code == "settlement_sample" for r in rows)
        assert rows[0].priority > rows[-1].priority

        # the same objects under an ordinary cycle id: the formula, untouched
        onto = get_ontology()
        async with get_sessionmaker()() as db:
            await db.execute(sql("""
                update object set provenance = provenance || jsonb_build_object('flywheel',
                    jsonb_build_object('cycle_id', 'cycle-plain'))
                where object_id = any(cast(:ids as uuid[]))"""),
                {"ids": [str(o) for o in oids[:20]]})
            await db.commit()
            rows = await triage(db=db, states="review", session_id=str(sid),
                                flywheel="cycle-plain", limit=200)
            objs = {str(o.object_id): o for o in (await db.execute(select(Object).where(
                Object.object_id.in_(oids[:20])))).scalars().all()}
        assert len(rows) == 20
        for r in rows:
            _why, prio, _flags = _why_and_priority(objs[r.object_id], onto)
            assert r.priority == prio
        assert [r.priority for r in rows] == sorted((r.priority for r in rows), reverse=True)

    run_async(_flow())


@requires_infra
def test_a_spot_verdict_lands_through_the_review_path(monkeypatch):
    """Spots were dead: settle_lot minted them on settled objects no grid ever listed and nothing
    wrote human_verdict. Now the spot objects carry a `spot-` cycle id the grid serves under
    states=settled, and a person's ruling through apply_review_batch writes the spot verdict."""
    from services.api.routers.triage import triage
    from services.autolabel.ontology import get_ontology
    from services.govern.class_autonomy import set_level
    from services.labelops.settlement import plan_lot, settle_lot, tally_lot
    from services.review_apply import apply_review_batch

    _shrink(monkeypatch)

    async def _flow():
        sid, cid, oids, *_ = await _seed_stratum("tempo", 30)
        await _clear_stratum(cid)
        prior = await _governance(settlement=True, loop=True)
        try:
            async with get_sessionmaker()() as db:
                plan = await plan_lot(db, "tempo", epoch=EPOCH, created_by="test")
            await _judge_sample(plan["lot_id"])
            async with get_sessionmaker()() as db:
                assert (await tally_lot(db, plan["lot_id"]))["status"] == "accepted"
                await set_level(db, cid, 2, set_by="test")
                settled = await settle_lot(db, plan["lot_id"], created_by="test")
            assert settled["spots"] > 0
            assert settled["spot_review_at"] == \
                f"/review/grid?flywheel={settled['spot_batch']}&states=settled"

            async with get_sessionmaker()() as db:
                rows = await triage(db=db, states="settled", flywheel=settled["spot_batch"],
                                    limit=200)
                assert len(rows) == settled["spots"], "every spot is on the grid"
                assert all(r.flags[0].code == "settlement_spot" for r in rows)

                spots = (await db.execute(select(SettlementSpot).where(
                    SettlementSpot.lot_id == uuid.UUID(plan["lot_id"])))).scalars().all()
                assert all(s.human_verdict is None for s in spots)
                first, second = spots[0], spots[1] if len(spots) > 1 else None
                obj = await db.get(Object, first.object_id)
                res = await apply_review_batch(db, [obj], action="accept", onto=get_ontology(),
                                               role="reviewer", reviewer="spot-judge")
                await db.commit()
                assert res.spot_judged == 1
                await db.refresh(first)
                assert first.human_verdict == "correct" and first.verdict_at is not None
                await db.refresh(obj)
                assert obj.state == "accepted", "a person's ruling on a settled object upgrades it"
                if second is not None:
                    obj2 = await db.get(Object, second.object_id)
                    res = await apply_review_batch(db, [obj2], action="reject", onto=get_ontology(),
                                                   role="reviewer", reviewer="spot-judge")
                    await db.commit()
                    await db.refresh(second)
                    assert second.human_verdict == "incorrect" and res.spot_judged == 1
        finally:
            await _governance(settlement=prior[0], loop=prior[1])

    run_async(_flow())
