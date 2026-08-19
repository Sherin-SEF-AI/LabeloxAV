"""M-CAL.3a: writing real calibration into the store. spec_to_fields turns a human rig spec (focal or FOV,
mount height, pitch) into stored fields; upsert_calibration respects source precedence so a measured spec is
never downgraded by an estimate; and resolve_calibration then reads the stored measured row, scaled to the
image, instead of the nominal default."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from db.models import CameraCalibration
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.calibration.resolve import resolve_calibration
from services.calibration.store import spec_to_fields, upsert_calibration

pytestmark = pytest.mark.db

_CAM = "cam_test_mcal3"


async def _a_session(db) -> uuid.UUID:
    """A session of this test's own, rather than whichever one another test left behind.

    These two tests took the first row of `session` and asserted it existed. Under the isolated test
    database that row exists only when some earlier test happened to seed one, so the assertion passed in a
    full run and failed under any filter that did not happen to include the seeding test. An order
    dependency that only shows up under filtering is worse than a plain failure: it makes the suite pass
    for reasons nobody chose.
    """
    from services.autolabel.ontology import get_ontology

    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="CALIB-FIXTURE", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=get_ontology().version))
    await db.commit()
    return sid

def test_spec_fov_to_focal():
    f = spec_to_fields(1920, 1080, {"hfov_deg": 37.0, "height_m": 1.6, "pitch_deg": -2.0})
    assert abs(f["fx"] - 2870.0) < 6.0          # 1920 / (2 tan(18.5deg))
    assert f["fy"] == f["fx"]
    assert f["cx"] == 960.0 and f["cy"] == 540.0
    assert f["xyz_m"] == [0.0, 0.0, 1.6]
    assert f["rpy_deg"] == [0.0, -2.0, 0.0]


def test_spec_explicit_fx_and_principal_point():
    f = spec_to_fields(1280, 960, {"fx": 1000.0, "cx": 640.0})
    assert f["fx"] == 1000.0 and f["fy"] == 1000.0
    assert f["cx"] == 640.0 and f["cy"] == 480.0   # cy defaults to image centre


def test_spec_requires_a_focal_source():
    with pytest.raises(ValueError):
        spec_to_fields(1920, 1080, {"height_m": 1.5})


async def test_upsert_precedence_then_resolve_reads_measured():
    async with get_sessionmaker()() as db:
        sid = await _a_session(db)
    try:
        est = await upsert_calibration(sid, _CAM, spec_to_fields(1920, 1080, {"fx": 2000.0, "height_m": 1.4}),
                                       "estimated")
        assert est["stored"]
        meas = await upsert_calibration(sid, _CAM, spec_to_fields(1920, 1080, {"fx": 2870.0, "height_m": 1.6}),
                                        "measured")
        assert meas["stored"]
        # an estimate must not downgrade the measured row
        down = await upsert_calibration(sid, _CAM, spec_to_fields(1920, 1080, {"fx": 1500.0}), "estimated")
        assert not down["stored"] and "higher-trust" in down["reason"]

        c = await resolve_calibration(sid, _CAM, 960, 540)   # half the reference width
        assert c.source == "measured"
        assert abs(c.fx - 1435.0) < 1.0                      # 2870 * (960 / 1920)
        assert abs(c.xyz_m[2] - 1.6) < 1e-6
    finally:
        async with get_sessionmaker()() as db:
            await db.execute(delete(CameraCalibration).where(
                CameraCalibration.session_id == sid, CameraCalibration.cam_id == _CAM))
            await db.commit()
