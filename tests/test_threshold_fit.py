"""Fitted auto-accept thresholds, end to end, and the loud fallback when there is nothing fitted.

The behavioural claim is narrow and worth stating precisely: before this, `class_auto_accept` returned a
configured constant and the gate's docstring called it a calibrated precision floor. It was neither
calibrated nor measured. Now it prefers a threshold fitted from recorded outcomes, and when there is none
it still returns the constant but says so once per class.

That last part is the test that matters most. The fallback is the ordinary case today and will be for a
while, so a silent fallback would mean the engine cannot tell you which of its thresholds were measured,
which is indistinguishable from the state this work set out to leave behind.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.config import get_settings
from core.timebase import now_ns
from db.models import EvalPatch, Frame, InferenceRun, ModelRegistry, Prediction, ThresholdFit
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.gate import class_auto_accept
from services.autolabel.ontology import get_ontology
from services.oraclyx.threshold_fit import (
    activate_fit,
    active_thresholds,
    fit_thresholds,
)

pytestmark = pytest.mark.db


async def _run_with_outcomes(db, spec: dict[str, list[tuple[float, bool]]], *,
                             calibrated: bool = False) -> dict:
    """One inference run whose EvalPatch rows carry exactly `spec`: class name -> (score, was-right)."""
    onto = get_ontology()
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="THR", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=onto.version)
    db.add(sess)
    mv = f"thr-{uuid.uuid4().hex[:8]}"
    db.add(ModelRegistry(model_version=mv, task="detection"))
    await db.flush()
    run = InferenceRun(model_version=mv, gold_id=None, status="complete", params={}, code_sha="0" * 40)
    db.add(run)
    await db.flush()
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="front",
                  width=1920, height=1080, img_uri="s3://x/f.jpg")
    db.add(frame)
    await db.flush()

    eval_id = uuid.uuid4()
    for name, pairs in spec.items():
        cid = onto.by_name(name).id
        for score, right in pairs:
            p = Prediction(run_id=run.run_id, frame_id=frame.frame_id, class_id=cid,
                           bbox=[0.0, 0.0, 10.0, 10.0], conf=score,
                           conf_calibrated=score if calibrated else None)
            db.add(p)
            await db.flush()
            db.add(EvalPatch(eval_id=eval_id, run_id=run.run_id, prediction_id=p.prediction_id,
                             frame_id=frame.frame_id, outcome="tp" if right else "fp",
                             gt_class_id=cid if right else None, pred_class_id=cid, conf=score))
    await db.commit()
    return {"run": run, "model_version": mv, "onto": onto}


# Eighty outcomes for one class, wrong every fifth from the top. Enough support to fit, and structured so
# the answer is checkable: FAR at depth k is roughly k/5 over k, so a 0.05 bound bites near the top and a
# 0.25 bound admits the whole sample.
_MOSTLY_RIGHT = [(1.0 - 0.01 * i, i % 5 != 4) for i in range(80)]


class TestFitting:
    async def test_it_fits_a_threshold_and_records_the_delta_from_the_constant(self):
        async with get_sessionmaker()() as db:
            fx = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT})
            res = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200)
            assert "error" not in res, res
            row = res["per_class"][0]
            assert row["class_name"] == "pedestrian"
            assert row["measured"] is True, row["reason"]
            # A VRU carries the pack's critical bound, not a config number.
            assert row["alpha"] == 0.01
            assert row["config_threshold"] == get_settings().gate.safety_auto_accept
            assert row["delta"] == round(row["threshold"] - row["config_threshold"], 4)
            assert row["far_at"] <= row["alpha"]

    async def test_a_class_with_too_few_outcomes_is_stored_refused_not_omitted(self):
        """The gate has to tell "earned no threshold" from "nobody looked", and only rows can say that."""
        async with get_sessionmaker()() as db:
            fx = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT,
                                               "bus": [(0.9, True), (0.8, False)]})
            res = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200)
            by_name = {r["class_name"]: r for r in res["per_class"]}
            assert set(by_name) == {"pedestrian", "bus"}
            assert by_name["bus"]["measured"] is False
            assert "below the 50 needed" in by_name["bus"]["reason"]
            assert by_name["bus"]["threshold"] is None
            assert res["n_fitted"] == 1 and res["n_refused"] == 1

    async def test_the_bound_comes_from_the_pack_and_differs_by_class(self):
        async with get_sessionmaker()() as db:
            fx = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT, "sedan": _MOSTLY_RIGHT})
            res = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200)
            by_name = {r["class_name"]: r for r in res["per_class"]}
            assert by_name["pedestrian"]["alpha"] < by_name["sedan"]["alpha"]
            # Identical outcomes, different bounds, so the stricter class must not accept more.
            assert by_name["pedestrian"]["accept_rate"] <= by_name["sedan"]["accept_rate"]

    async def test_it_records_which_confidence_column_it_read(self):
        """A threshold fitted on calibrated confidence and applied to raw is arbitrary, not conservative."""
        async with get_sessionmaker()() as db:
            raw = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT}, calibrated=False)
            cal = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT}, calibrated=True)
            assert (await fit_thresholds(db, run_id=str(raw["run"].run_id),
                                         n_boot=200))["score_field"] == "conf"
            assert (await fit_thresholds(db, run_id=str(cal["run"].run_id),
                                         n_boot=200))["score_field"] == "conf_calibrated"

    async def test_a_reconstructed_run_cannot_be_fitted(self):
        """Its predictions were backfilled from review history and never had a real confidence.

        The same reason the promotion gate refuses AP on one: there is no score distribution to place a
        threshold inside.
        """
        async with get_sessionmaker()() as db:
            fx = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT})
            fx["run"].params = {"reconstructed": True}
            await db.commit()
            res = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200)
            assert "error" in res and "reconstructed" in res["error"]


class TestActivation:
    async def test_a_fit_is_stored_inactive(self):
        """Fitting is a measurement. Activating changes what enters the corpus unseen, across every class.

        gold_calibrate set this precedent: fit, report whether it is trustworthy, leave the switch to a
        person.
        """
        async with get_sessionmaker()() as db:
            fx = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT})
            res = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200)
            rows = (await db.execute(select(ThresholdFit).where(
                ThresholdFit.fit_id == uuid.UUID(res["fit_id"])))).scalars().all()
            assert all(r.active is False for r in rows)
            assert (await active_thresholds(db, fx["model_version"]))["by_class"] == {}

    async def test_activating_retires_the_previous_fit_for_that_model(self):
        """Wholesale, because a threshold set half from one evaluation and half from another is not an
        operating point."""
        async with get_sessionmaker()() as db:
            fx = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT})
            first = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200)
            second = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200)
            await activate_fit(db, first["fit_id"])
            assert (await active_thresholds(db, fx["model_version"]))["fit_id"] == first["fit_id"]
            await activate_fit(db, second["fit_id"])
            act = await active_thresholds(db, fx["model_version"])
            assert act["fit_id"] == second["fit_id"]
            still_on = (await db.execute(select(ThresholdFit).where(
                ThresholdFit.fit_id == uuid.UUID(first["fit_id"]),
                ThresholdFit.active.is_(True)))).scalars().all()
            assert still_on == []

    async def test_a_refused_class_never_reaches_the_gate_map(self):
        """Absent, not present with its config value: the gate must fall back explicitly and say so.

        Filling it in here would put a constant into a map the gate treats as measured, which is precisely
        the confusion being removed.
        """
        async with get_sessionmaker()() as db:
            fx = await _run_with_outcomes(db, {"pedestrian": _MOSTLY_RIGHT,
                                               "bus": [(0.9, True), (0.8, False)]})
            res = await fit_thresholds(db, run_id=str(fx["run"].run_id), n_boot=200, activate=True)
            act = await active_thresholds(db, fx["model_version"])
            onto = get_ontology()
            assert onto.by_name("pedestrian").id in act["by_class"]
            assert onto.by_name("bus").id not in act["by_class"]
            assert act["n_active"] == 2 and act["n_usable"] == 1
            assert res["fit_id"] == act["fit_id"]


class TestTheGateReadsIt:
    def test_a_fitted_threshold_wins_over_the_constant(self):
        onto, cfg = get_ontology(), get_settings().gate
        cid = onto.by_name("pedestrian").id
        assert class_auto_accept(cid, onto, cfg) == cfg.safety_auto_accept
        assert class_auto_accept(cid, onto, cfg, {cid: 0.61}) == 0.61

    def test_an_unfitted_class_still_gets_the_constant(self):
        """Falling back is correct. Falling back silently is not, and the warning is the difference.

        Every class is unfitted today, so a gate that could not report this would be indistinguishable
        from one that never learned to measure anything.
        """
        import services.autolabel.gate as gate_mod

        onto, cfg = get_ontology(), get_settings().gate
        cid = onto.by_name("sedan").id
        gate_mod._WARNED_UNFITTED.discard(cid)
        seen: list[dict] = []
        real = gate_mod.log.warning
        gate_mod.log.warning = lambda ev, **kw: seen.append({"event": ev, **kw})  # type: ignore[assignment]
        try:
            assert class_auto_accept(cid, onto, cfg, {}) == cfg.auto_accept
            # Once per class per process, not once per object: a run gates tens of thousands of times.
            class_auto_accept(cid, onto, cfg, {})
            class_auto_accept(cid, onto, cfg, {})
        finally:
            gate_mod.log.warning = real  # type: ignore[assignment]
        assert len(seen) == 1, seen
        assert seen[0]["event"] == "gate.threshold_unfitted"
        assert seen[0]["class_id"] == cid and seen[0]["had_fit"] is True

    def test_the_gate_state_actually_moves_with_a_fitted_threshold(self):
        """The behavioural end of it: an object that was held for review now auto-accepts, or the reverse.

        A threshold that is read but changes no decision is a number in a table.
        """
        from core.schemas import BBox, GateState, Provenance, UnifiedObject
        from services.autolabel.gate import gate_object

        onto, cfg = get_ontology(), get_settings().gate
        cid = onto.by_name("sedan").id
        # Relative to the live constant, not to the value the docstring quotes. This deployment runs
        # auto_accept at 0.45 rather than the documented 0.95, which is exactly the kind of drift that
        # makes "the threshold is a precision floor" an unverifiable claim.
        conf = (cfg.auto_accept + cfg.review_low) / 2.0
        obj = UnifiedObject(class_id=cid, class_name="sedan",
                            bbox=BBox(x1=0, y1=0, x2=10, y2=10), conf=conf,
                            provenance=Provenance(agreement=True))
        assert conf < cfg.auto_accept
        assert gate_object(obj, onto, cfg, fitted={}) == GateState.review
        # Measured below the object's score, the same object clears it.
        assert gate_object(obj, onto, cfg, fitted={cid: conf - 0.01}) == GateState.auto_accept
        # And a fit stricter than the constant holds back an object the constant would have let through.
        loud = UnifiedObject(class_id=cid, class_name="sedan",
                             bbox=BBox(x1=0, y1=0, x2=10, y2=10), conf=cfg.auto_accept + 0.01,
                             provenance=Provenance(agreement=True))
        assert gate_object(loud, onto, cfg, fitted={}) == GateState.auto_accept
        assert gate_object(loud, onto, cfg, fitted={cid: 0.999}) == GateState.review
