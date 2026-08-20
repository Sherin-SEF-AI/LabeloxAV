"""Calibration conditioned on scene density, and the fallback it refuses to hide.

One isotonic curve per class assumes a detector's confidence means the same thing on an empty highway and
at a crowded junction. It does not: in a crowded frame objects occlude each other and a 0.8 is far more
often wrong. Averaging the two gives a curve that is optimistic where it matters, and the error cancels in
the aggregate ECE so nothing reports it.

The synthetic corpus below builds exactly that: the same confidence is right 90% of the time in sparse
frames and 40% in dense ones. A per-class curve can only land between them; a density-conditioned one
recovers both. The interesting test is the last one, where a cell is too thin to fit and the table has to
say so rather than quietly serving the class curve as though it were conditioned.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.timebase import now_ns
from db.models import EvalPatch, Frame, InferenceRun, ModelRegistry, Prediction
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.oraclyx.density_calibration import (
    Curve,
    DensityCalibration,
    fit_density_calibration,
    write_calibrated_confidence,
)
from services.verdyx.blind_audit import DENSITY_BOUNDS, density_stratum

pytestmark = pytest.mark.db


class TestTheSharedBuckets:
    def test_density_means_the_same_thing_here_as_in_the_audit(self):
        """A calibration cell and an audit stratum called "dense" must describe the same frames.

        Two independently-chosen boundary sets with the same names would be the worst outcome: every
        comparison between an audit stratum and a calibration cell would be quietly wrong.
        """
        assert [n for n, _lo, _hi in DENSITY_BOUNDS] == ["sparse", "moderate", "dense"]
        assert density_stratum(0) == "sparse" and density_stratum(9) == "sparse"
        assert density_stratum(10) == "moderate" and density_stratum(29) == "moderate"
        assert density_stratum(30) == "dense" and density_stratum(500) == "dense"


class TestTheCurve:
    def test_a_curve_with_no_knots_is_the_identity(self):
        # Never silently a constant: an empty curve means nothing was fitted, and the raw score is the
        # only honest answer.
        c = Curve((), (), 0, "class:1", False)
        assert c(0.3) == 0.3 and c(0.9) == 0.9

    def test_it_interpolates_and_clips(self):
        c = Curve((0.0, 0.5, 1.0), (0.0, 0.25, 1.0), 100, "class:1", False)
        assert abs(c(0.25) - 0.125) < 1e-9
        assert c(-5.0) == 0.0 and c(5.0) == 1.0

    def test_a_class_with_no_curve_at_all_is_named_uncalibrated(self):
        """Not silently the raw score under a calibrated-looking label.

        Every consumer that prefers conf_calibrated would then read raw confidence believing otherwise,
        which is the same defect as an unmeasured threshold called a precision floor.
        """
        cal = DensityCalibration({}, {})
        value, scope = cal.calibrate(0.7, 42, "dense")
        assert value == 0.7 and scope == "uncalibrated"

    def test_a_thin_cell_falls_back_to_the_class_curve_and_says_which(self):
        cls = Curve((0.0, 1.0), (0.0, 1.0), 500, "class:1", False)
        cell = Curve((0.0, 1.0), (0.0, 0.5), 500, "class:1|density:dense", False)
        cal = DensityCalibration({DensityCalibration.key(1, "dense"): cell}, {1: cls})
        assert cal.calibrate(0.8, 1, "dense") == (pytest.approx(0.4), "class:1|density:dense")
        # sparse was never fitted, so it serves the class curve and the scope says so.
        assert cal.calibrate(0.8, 1, "sparse") == (pytest.approx(0.8), "class:1")

    def test_it_round_trips_through_json(self):
        cls = Curve((0.0, 0.5, 1.0), (0.0, 0.3, 1.0), 500, "class:1", False)
        cal = DensityCalibration({DensityCalibration.key(1, "dense"): cls}, {1: cls},
                                 meta={"run_id": "r"})
        back = DensityCalibration.from_json(cal.to_json())
        assert back.calibrate(0.4, 1, "dense") == cal.calibrate(0.4, 1, "dense")
        assert back.meta["run_id"] == "r"


async def _corpus(db, *, n_sparse: int = 60, n_dense: int = 60, dense_per_frame: int = 35,
                  sparse_reliability: float = 0.9, dense_reliability: float = 0.4,
                  seed: int = 3):
    """A run where the same confidence means two different things depending on how crowded the frame is."""
    rng = np.random.default_rng(seed)
    onto = get_ontology()
    cid = onto.by_name("sedan").id
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="DENS", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=onto.version)
    db.add(sess)
    mv = f"dens-{uuid.uuid4().hex[:8]}"
    db.add(ModelRegistry(model_version=mv, task="detection"))
    await db.flush()
    run = InferenceRun(model_version=mv, status="complete", params={}, code_sha="0" * 40)
    db.add(run)
    await db.flush()

    eval_id = uuid.uuid4()
    for kind, n_frames, per_frame, reliability in (
            ("sparse", n_sparse, 3, sparse_reliability),
            ("dense", n_dense, dense_per_frame, dense_reliability)):
        for i in range(n_frames):
            f = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns() + i,
                      cam_id=kind, width=1920, height=1080, img_uri=f"s3://x/{kind}{i}.jpg")
            db.add(f)
            await db.flush()
            for _ in range(per_frame):
                conf = float(rng.uniform(0.3, 0.99))
                p = Prediction(run_id=run.run_id, frame_id=f.frame_id, class_id=cid,
                               bbox=[0.0, 0.0, 10.0, 10.0], conf=conf)
                db.add(p)
                await db.flush()
                right = rng.uniform(0, 1) < reliability
                db.add(EvalPatch(eval_id=eval_id, run_id=run.run_id, prediction_id=p.prediction_id,
                                 frame_id=f.frame_id, outcome="tp" if right else "fp",
                                 gt_class_id=cid if right else None, pred_class_id=cid, conf=conf))
    await db.commit()
    return run, cid


class TestFitting:
    async def test_it_recovers_two_different_reliabilities_that_one_curve_cannot(self):
        """The whole claim, on data built so a single curve provably cannot express it.

        The same confidence is right 90% of the time in sparse frames and 40% in dense ones. A per-class
        curve has to land between; the conditioned cells land near each truth.
        """
        async with get_sessionmaker()() as db:
            run, cid = await _corpus(db)
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            assert out["measured"] is True, out.get("reason")
            cal = out["calibration"]
            sparse, _ = cal.calibrate(0.7, cid, "sparse")
            dense, _ = cal.calibrate(0.7, cid, "dense")
            assert sparse > dense + 0.2, (sparse, dense)
            assert 0.8 < sparse < 1.0 and 0.25 < dense < 0.55

    async def test_conditioning_is_judged_on_the_worst_bucket_not_the_average(self):
        """The aggregate cannot see this defect, and the fixture proves it rather than asserting it.

        A per-class curve that predicts the pooled base rate everywhere scores a near-perfect aggregate
        ECE while being badly wrong in both buckets: it is over-confident in the dense half by as much as
        it is under-confident in the sparse half, and the binning averages the two away. That
        cancellation is the failure, so the verdict compares the worst bucket, which cannot cancel.
        """
        async with get_sessionmaker()() as db:
            run, _cid = await _corpus(db)
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            # Per bucket the conditioned curve wins, which is the comparison the verdict is made on and
            # the only one that cannot cancel. The aggregate is deliberately NOT asserted in either
            # direction: whether it flatters the per-class curve depends on how the split falls, and that
            # it can flatter it at all is the reason it is not the criterion.
            assert out["worst_bucket_density_ece"] < out["worst_bucket_per_class_ece"], out["per_bucket"]
            assert out["beats_per_class"] is True
            assert out["trustworthy"] is True

    async def test_every_bucket_reports_its_own_error_not_just_the_average(self):
        """The aggregate hides exactly the failure this fixes: the two errors cancel."""
        async with get_sessionmaker()() as db:
            run, _cid = await _corpus(db)
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            assert set(out["per_bucket"]) == {"sparse", "moderate", "dense"}
            for name in ("sparse", "dense"):
                b = out["per_bucket"][name]
                assert b["n"] > 0 and b["cal_ece"] is not None

    async def test_a_thin_cell_is_recorded_rather_than_silently_falling_back(self):
        """A partly-conditioned table read as fully conditioned is the failure mode.

        Here the dense bucket has only a handful of detections, far below the cell minimum, so it must
        appear in thin_cells with its count and what it fell back to.
        """
        async with get_sessionmaker()() as db:
            # Three frames at 30 predictions each: 30 is the dense boundary, so the cell really is the
            # dense one, and 90 observations is under the 100-observation minimum however the split falls.
            # (Five per frame would have put these in the SPARSE bucket, since the bucket is decided by
            # the prediction count, which is what made the first version of this test assert nothing.)
            run, _cid = await _corpus(db, n_dense=3, dense_per_frame=30)
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            assert out["measured"] is True
            thin = {t["bucket"]: t for t in out["thin_cells"]}
            assert "dense" in thin, out["thin_cells"]
            assert thin["dense"]["n"] < thin["dense"]["min"]
            assert thin["dense"]["fell_back_to"] in ("class", "uncalibrated")
            assert "cells had at least" in out["coverage_note"]

    async def test_every_cell_in_the_data_is_accounted_for_however_the_split_falls(self):
        """A cell is fitted or thin. It used to be possible to be neither.

        The inventory was taken over the training half, so a cell whose every observation landed in
        validation appeared in no total at all and the coverage note undercounted its own denominator -
        the exact dishonesty this module says it is avoiding. Asserted as an invariant rather than by
        engineering a particular split, because which cell lands where is not the property under test.
        """
        async with get_sessionmaker()() as db:
            run, cid = await _corpus(db, n_sparse=40, n_dense=3, dense_per_frame=30)
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            # One class, two populated buckets: sparse (3 per frame) and dense (30 per frame).
            assert out["n_cells"] + out["n_thin_cells"] == 2, out["thin_cells"]
            assert f"of {out['n_cells'] + out['n_thin_cells']}" in out["coverage_note"]
            for t in out["thin_cells"]:
                assert t["reason"] in ("below the cell minimum",
                                       "none of this cell's observations landed in the training split")
                assert t["n_all"] >= t["n"], "the all-pairs count cannot be under the training count"

    async def test_a_reconstructed_run_is_refused(self):
        async with get_sessionmaker()() as db:
            run, _cid = await _corpus(db, n_sparse=5, n_dense=5)
            run.params = {"reconstructed": True}
            await db.commit()
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            assert out["measured"] is False and "reconstructed" in out["reason"]

    async def test_a_run_with_nothing_scored(self):
        async with get_sessionmaker()() as db:
            onto = get_ontology()
            sess = DbSession(session_id=uuid.uuid4(), vehicle_id="EMPTY", start_ts_ns=0, end_ts_ns=1,
                             ontology_version=onto.version)
            db.add(sess)
            mv = f"empty-{uuid.uuid4().hex[:8]}"
            db.add(ModelRegistry(model_version=mv, task="detection"))
            await db.flush()
            run = InferenceRun(model_version=mv, status="complete", params={}, code_sha="0" * 40)
            db.add(run)
            await db.commit()
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            assert out["measured"] is False and "no scored predictions" in out["reason"]


class TestWritingTheCalibratedColumn:
    async def test_it_fills_conf_calibrated_which_nothing_had_ever_written(self):
        """The column has existed unwritten since migration 0069.

        That is why every threshold fitted so far reads raw confidence: threshold_fit prefers the
        calibrated column and has never found one.
        """
        from sqlalchemy import func, select

        async with get_sessionmaker()() as db:
            run, _cid = await _corpus(db)
            before = (await db.execute(select(func.count()).select_from(Prediction).where(
                Prediction.run_id == run.run_id,
                Prediction.conf_calibrated.is_not(None)))).scalar_one()
            assert before == 0

            out = await fit_density_calibration(db, run_id=str(run.run_id))
            res = await write_calibrated_confidence(db, run_id=str(run.run_id),
                                                    calibration=out["calibration"])
            assert res["n_written"] > 0
            after = (await db.execute(select(func.count()).select_from(Prediction).where(
                Prediction.run_id == run.run_id,
                Prediction.conf_calibrated.is_not(None)))).scalar_one()
            assert after == res["n_written"]

    async def test_the_raw_confidence_is_never_touched(self):
        """The append-only invariant protects the model's raw output; the derived column exists to be filled."""
        from sqlalchemy import select

        async with get_sessionmaker()() as db:
            run, _cid = await _corpus(db)
            raw_before = sorted((await db.execute(select(Prediction.conf).where(
                Prediction.run_id == run.run_id))).scalars().all())
            out = await fit_density_calibration(db, run_id=str(run.run_id))
            await write_calibrated_confidence(db, run_id=str(run.run_id),
                                              calibration=out["calibration"])
            raw_after = sorted((await db.execute(select(Prediction.conf).where(
                Prediction.run_id == run.run_id))).scalars().all())
            assert raw_before == raw_after

    async def test_an_uncalibrated_class_is_left_null_rather_than_filled_with_the_raw_score(self):
        """Filling it would make an uncalibrated prediction indistinguishable from a calibrated one."""
        async with get_sessionmaker()() as db:
            run, _cid = await _corpus(db, n_sparse=5, n_dense=5)
            res = await write_calibrated_confidence(db, run_id=str(run.run_id),
                                                    calibration=DensityCalibration({}, {}))
            assert res["n_written"] == 0
            assert res["n_uncalibrated"] > 0
