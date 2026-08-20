"""Drift detection has to be able to detect drift.

Two defects made it structurally blind:

1. Input drift projected 768-dimensional embeddings onto basis vector 0 and binned over a fixed [-1, 1]
   range. One coordinate of 768 measures a single arbitrary feature, so any shift in the other 767
   directions, or any shift orthogonal to that axis, was invisible no matter how large.
2. The label histogram was hardcoded to 64 class slots while the ontology carries far more, so every class
   with a higher id was dropped and a shift confined to them could not register.

Pure unit tests: the statistics are computed on synthetic matrices, so no database and no embeddings service."""
from __future__ import annotations

import numpy as np
import pytest

from services.govern.drift import (
    _quantile_bins,
    ontology_class_slots,
    projection_axes,
    psi,
)

DIM = 768


def _worst_psi(ref: np.ndarray, cur: np.ndarray, axes: np.ndarray) -> float:
    """The statistic input_embedding_drift reports: the worst PSI across the projection ensemble."""
    ref_p, cur_p = ref @ axes.T, cur @ axes.T
    vals = []
    for j in range(axes.shape[0]):
        bins = _quantile_bins(ref_p[:, j])
        if len(bins) < 3:
            continue
        r = np.histogram(ref_p[:, j], bins=bins)[0].astype(float)
        c = np.histogram(cur_p[:, j], bins=bins)[0].astype(float)
        vals.append(psi(r.tolist(), c.tolist()))
    return float(max(vals)) if vals else 0.0


def _old_single_axis_psi(ref: np.ndarray, cur: np.ndarray) -> float:
    """The previous implementation, reproduced here so the regression is demonstrated, not asserted."""
    axis = np.zeros(DIM, dtype=np.float32)
    axis[0] = 1.0
    bins = np.linspace(-1.0, 1.0, 11)
    r = np.histogram(ref @ axis, bins=bins)[0].astype(float)
    c = np.histogram(cur @ axis, bins=bins)[0].astype(float)
    return psi(r.tolist(), c.tolist())


def test_drift_orthogonal_to_the_old_axis_is_now_detected():
    # A world that shifts in every dimension EXCEPT coordinate 0: the old metric is blind by construction.
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, size=(600, DIM)).astype(np.float32)
    cur = ref.copy()
    cur[:, 1:] += 1.5                       # large shift, none of it on axis 0

    assert _old_single_axis_psi(ref, cur) < 0.01, "sanity: the old metric could not see this"
    assert _worst_psi(ref, cur, projection_axes(DIM, 32)) > 0.5, "the ensemble must see it"


def test_a_broad_distribution_shift_is_detected():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, size=(500, DIM)).astype(np.float32)
    cur = (ref + 0.9).astype(np.float32)
    assert _worst_psi(ref, cur, projection_axes(DIM, 32)) > 1.0


def test_identical_windows_report_no_drift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0.0, 1.0, size=(400, DIM)).astype(np.float32)
    assert _worst_psi(ref, ref.copy(), projection_axes(DIM, 32)) < 1e-6


def test_a_variance_only_change_is_detected():
    # A world that keeps its mean but spreads out: a mean-shift test would miss it, quantile binning does not.
    rng = np.random.default_rng(3)
    ref = rng.normal(0.0, 1.0, size=(600, DIM)).astype(np.float32)
    cur = rng.normal(0.0, 3.0, size=(600, DIM)).astype(np.float32)
    assert _worst_psi(ref, cur, projection_axes(DIM, 32)) > 0.3


def test_projection_axes_are_deterministic_and_unit_norm():
    # Reproducibility matters: the same two windows must yield the same number on every run, or a "breach"
    # is just resampling noise.
    a, b = projection_axes(DIM, 16), projection_axes(DIM, 16)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert a.shape == (16, DIM)


def test_quantile_bins_span_the_reference_and_are_open_ended():
    rng = np.random.default_rng(4)
    vals = rng.normal(size=2000)
    bins = _quantile_bins(vals, n_bins=10)
    assert bins[0] == -np.inf and bins[-1] == np.inf   # current-window outliers always land in a bin
    assert np.all(np.diff(bins) > 0)                    # strictly increasing, valid for np.histogram


def test_label_histogram_covers_the_whole_ontology():
    # The hardcoded 64 dropped every class above it; the slot count must follow the real ontology.
    slots = ontology_class_slots()
    assert slots > 64, "the ontology is larger than the old hardcoded histogram"
    from services.autolabel.ontology import get_ontology
    assert slots == max(c.id for c in get_ontology().classes) + 1


def test_psi_is_zero_for_identical_distributions_and_positive_otherwise():
    assert psi([10, 10, 10], [10, 10, 10]) < 1e-9
    assert psi([90, 5, 5], [5, 5, 90]) > 1.0


@pytest.mark.db
async def test_an_unmeasured_gate_is_not_reported_as_a_perfect_one():
    """The absence of evidence was being recorded as the strongest possible evidence.

    control_precision_drift substituted 1.0 when no control sample carried a human verdict, and
    run_drift_scan persisted it. Measured on this corpus: 601 samples waiting, none judged, and the drift
    ledger recording the gate's realized precision as 100% - a flat line at perfect for a number that had
    never been taken. This is the one figure a buyer is meant to be able to trust over a self-reported one.
    """
    from sqlalchemy import delete, func, select

    from db.models import ControlSample, DriftMetric
    from db.session import get_sessionmaker
    from services.govern.drift import control_precision_drift, run_drift_scan

    async with get_sessionmaker()() as db:
        # No verdicts anywhere: the starved state.
        await db.execute(delete(ControlSample))
        await db.commit()

        r = await control_precision_drift(db)
        assert r["unmeasured"] is True
        assert r["value"] is None, "an unmeasured precision must not carry a number at all"
        assert r["breach"] is False, (
            "starvation is not a breach: it is not the same as being measured below the floor, and turning "
            "it into one would pause every promotion for a reason nobody can fix in the moment")

        before = (await db.execute(select(func.count()).select_from(DriftMetric)
                                   .where(DriftMetric.metric == "control_precision"))).scalar_one()
        await run_drift_scan(db)
        after = (await db.execute(select(func.count()).select_from(DriftMetric)
                                  .where(DriftMetric.metric == "control_precision"))).scalar_one()
        assert after == before, (
            "an unmeasured metric was persisted; a gap in the series is honest, a substituted value is a "
            "reading that was never taken")


@pytest.mark.db
async def test_a_measured_gate_still_reports_and_breaches():
    """The other half: once verdicts exist the metric behaves exactly as before."""
    import uuid as _uuid

    from sqlalchemy import delete

    from db.models import ControlSample, Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.govern.drift import control_precision_drift

    async with get_sessionmaker()() as db:
        await db.execute(delete(ControlSample))
        sess = DbSession(session_id=_uuid.uuid4(), vehicle_id="CTL-01", start_ts_ns=0, end_ts_ns=1,
                         ontology_version="test")
        db.add(sess)
        fid = _uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                     img_uri="s3://x/y.jpg", width=64, height=64))
        await db.flush()
        # Four judged auto-accepts, one of them wrong -> 0.75, well under the 0.97 floor.
        for i in range(4):
            oid = _uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=1, bbox=[1, 1, 9, 9], conf=0.9,
                          source="auto_accept", state="auto_accept"))
            await db.flush()
            db.add(ControlSample(object_id=oid, was_auto_accepted=True,
                                 human_verdict="incorrect" if i == 0 else "correct"))
        await db.commit()

        r = await control_precision_drift(db)
        assert r["unmeasured"] is False
        assert r["value"] == 0.75
        assert r["breach"] is True and r["reviewed"] == 4
