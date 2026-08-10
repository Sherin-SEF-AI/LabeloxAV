"""The consensus endpoint was reading a table that has never had a row in it.

`/calyx/rig/{vehicle}/consensus` fused `calibration_override`, which is written only when somebody corrects a
calibration by hand. There are zero of those on this deployment, so for DASHCAM-01, a vehicle with
ninety-seven calibrations across ninety-seven sessions, it answered `n_overrides: 0` and a prior of nothing.
The estimates were in `camera_calibration` the whole time.

This is the wiring failure that keeps recurring here: working code reading the wrong end of the spine, and
returning an empty answer that looks like a fact about the fleet instead of a fact about the query.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import CameraCalibration
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.calyx.run import rig_prior_for_vehicle

pytestmark = pytest.mark.db


async def _seed(db, vehicle: str, pitches: list[float], *, cam: str = "cam_f") -> None:
    """The corpus shape: intrinsics and five of six extrinsic axes identical, only pitch moving."""
    for p in pitches:
        sid = uuid.uuid4()
        db.add(DbSession(session_id=sid, vehicle_id=vehicle, start_ts_ns=0, end_ts_ns=1,
                         ontology_version="test"))
        await db.flush()
        db.add(CameraCalibration(session_id=sid, cam_id=cam, model="pinhole",
                                 fx=2870.0, fy=2870.0, cx=960.0, cy=540.0, dist=[0.0] * 5,
                                 ref_width=1920, rpy_deg=[0.0, p, 0.0], xyz_m=[0.0, 0.0, 1.5],
                                 source="estimated", quality=0.6))
    await db.commit()


async def test_a_vehicle_with_calibrations_and_no_overrides_still_gets_a_prior():
    """The bug, exactly. Ninety-seven calibrations and no overrides used to produce an empty prior."""
    veh = f"TEST-CAL-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        await _seed(db, veh, [0.50, 0.54, 0.48, 0.60, 0.51, 0.55])
        out = await rig_prior_for_vehicle(db, veh)
    assert out["n_calibrations"] == 6
    assert out["n_overrides"] == 0
    assert out["prior"]["n"] == 6


async def test_the_prior_names_the_one_axis_this_corpus_measures():
    veh = f"TEST-CAL-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        await _seed(db, veh, [0.50, 0.54, 0.48, 0.60, 0.51, 0.55])
        out = await rig_prior_for_vehicle(db, veh)
    assert out["prior"]["measured_axes"] == ["pitch"]
    assert "z" in out["prior"]["constant_axes"]


async def test_a_session_whose_pitch_is_far_from_the_fleet_is_surfaced():
    """The output an operator can act on: this session, this many sigmas, on this axis."""
    veh = f"TEST-CAL-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        await _seed(db, veh, [0.50, 0.54, 0.48, 0.60, 0.51, 0.55, 0.49, 0.52, 9.0])
        out = await rig_prior_for_vehicle(db, veh)
    assert out["n_outliers"] == 1
    assert out["outliers"][0]["flagged_axes"] == ["pitch"]


async def test_confidence_reflects_that_five_axes_were_never_measured():
    """A prior over many sessions that only ever observed pitch must not read like a full calibration."""
    veh = f"TEST-CAL-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        await _seed(db, veh, [0.5 + 0.01 * i for i in range(20)])
        out = await rig_prior_for_vehicle(db, veh)
    assert 0.0 < out["confidence"] <= 1 / 6 + 1e-6


async def test_an_unknown_vehicle_returns_empty_rather_than_raising():
    async with get_sessionmaker()() as db:
        out = await rig_prior_for_vehicle(db, f"NOPE-{uuid.uuid4().hex[:6]}")
    assert out["n_calibrations"] == 0 and out["confidence"] == 0.0
