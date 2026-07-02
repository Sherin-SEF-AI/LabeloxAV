"""M-MC.3 annotate-once propagate (Tier 2): a ground object drawn in one camera projects into the other rig
view through real calibration, creating a propagated object (source=propagated, routed to review, stamped with
the origin and rig-linked). An uncalibrated session is gated back to Tier 1. Single asyncio.run so the cached
engine binds to one loop.

The rig is a synthetic forward-facing stereo pair (both cameras look forward, offset 0.6 m laterally) so a
forward ground point is in both fields of view, which keeps the geometry easy to reason about in the test."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _seed_rig(db, sid, calibrated: bool):
    """Calibration rows for a session that has already been flushed (so the FK to session resolves)."""
    from db.models import CalibrationValidation, CameraCalibration

    # forward-facing stereo: same orientation, 0.6 m apart in the ego y (left) axis
    for cam, y in (("cam_f", 0.3), ("cam_l", -0.3)):
        db.add(CameraCalibration(session_id=sid, cam_id=cam, model="pinhole", fx=1500.0, fy=1500.0,
                                 cx=960.0, cy=540.0, dist=[], ref_width=1920, rpy_deg=[0.0, 0.0, 0.0],
                                 xyz_m=[0.0, y, 1.5], source="measured", quality=0.9))
        if calibrated:
            db.add(CalibrationValidation(session_id=sid, cam_id=cam, model="pinhole", status="pass"))


@requires_infra
def test_propagate_projects_and_gates():
    from sqlalchemy import delete, select
    from db.models import CalibrationValidation, CameraCalibration, Frame, FrameGroup, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.multicam.propagate import propagate_object
    from services.multicam.sync import persist_groups

    sid = uuid.uuid4()
    sid_uncal = uuid.uuid4()

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add_all([DbSession(session_id=s, vehicle_id="RIG-MMC3", start_ts_ns=0, end_ts_ns=1,
                                  ontology_version="labelox-in-0.1.0") for s in (sid, sid_uncal)])
            await db.flush()
            _seed_rig(db, sid, calibrated=True)
            _seed_rig(db, sid_uncal, calibrated=False)
            await db.flush()
            oids = {}
            for s in (sid, sid_uncal):
                f1 = Frame(session_id=s, ts_ns=1000, cam_id="cam_f", img_uri="s3://x/f.jpg", width=1920, height=1080)
                f2 = Frame(session_id=s, ts_ns=1005, cam_id="cam_l", img_uri="s3://x/l.jpg", width=1920, height=1080)
                db.add_all([f1, f2])
                await db.flush()
                # a ground object in cam_f: bottom-centre at (960, 800) lifts to ~8.7 m forward on the road
                o = Object(frame_id=f1.frame_id, class_id=11, bbox=[900, 600, 1020, 800], conf=0.9,
                           state="accepted", source="human")
                db.add(o)
                await db.flush()
                oids[s] = o.object_id
            await db.commit()

        await persist_groups(sid)
        await persist_groups(sid_uncal)

        # calibrated session: propagates into cam_l
        res = await propagate_object(oids[sid], use_sam=False)
        assert res.get("gated") is not True, res
        cam_l = next((t for t in res["targets"] if t["cam"] == "cam_l"), None)
        assert cam_l is not None and cam_l["in_view"] is True, res
        assert len(res["created"]) == 1
        assert res["metric"]["range_m"] > 3.0  # a real forward distance

        async with maker() as db:
            new = await db.get(Object, uuid.UUID(res["created"][0]))
            assert new.source == "propagated" and new.state == "review"
            assert new.provenance["from_object_id"] == str(oids[sid])
            assert new.rig_object_id is not None  # linked into the source's rig identity
            src = await db.get(Object, oids[sid])
            assert src.rig_object_id == new.rig_object_id

        # uncalibrated session: gated back to Tier 1 (manual linking)
        g = await propagate_object(oids[sid_uncal], use_sam=False)
        assert g.get("gated") is True and g.get("tier") == 1, g

        async with maker() as db:
            from db.models import RigObject

            for s in (sid, sid_uncal):
                await db.execute(delete(RigObject).where(RigObject.session_id == s))
                await db.execute(delete(FrameGroup).where(FrameGroup.session_id == s))
                await db.execute(delete(CalibrationValidation).where(CalibrationValidation.session_id == s))
                await db.execute(delete(CameraCalibration).where(CameraCalibration.session_id == s))
                await db.delete(await db.get(DbSession, s))
            await db.commit()

    asyncio.run(run())
