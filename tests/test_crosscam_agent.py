"""Cross-camera propagation: project an object's 3D cuboid into a synchronized overlapping camera, create
the 2D box there, and revert (delete). cam_f -> cam_rt is a real front/side FOV overlap on the rig."""

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


def test_project_box_front_and_side():
    """Pure geometry: a forward cuboid is visible in the front-ish right camera, not the rear/left."""
    from services.agent.crosscam_agent import _project_box

    cub = {"center": [10.0, 0.0, 0.75], "size": [1.8, 4.2, 1.5], "yaw": 0.0}
    box_rt, vis_rt = _project_box(cub, "cam_rt", 1920, 1080)
    assert box_rt is not None and vis_rt > 0.5
    assert _project_box(cub, "cam_b", 1920, 1080)[0] is None    # behind the rear camera


async def _seed_two_cam():
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, ff, fr, oid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(4).integers(30, 220, size=(1080, 1920, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="XCAM-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        u1 = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
        u2 = store.put_bytes(f"frames/{sid}/cam_rt/{ts}.jpg", buf.tobytes(), "image/jpeg")
        db.add(Frame(frame_id=ff, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=u1, width=1920, height=1080, quality=0.9))
        db.add(Frame(frame_id=fr, session_id=sid, ts_ns=ts, cam_id="cam_rt", img_uri=u2, width=1920, height=1080, quality=0.9))
        db.add(Object(object_id=oid, frame_id=ff, class_id=sedan, bbox=[850.0, 500.0, 1120.0, 760.0], conf=0.9,
                      source="fused", state="review", cuboid_3d={"center": [10.0, 0.0, 0.75], "size": [1.8, 4.2, 1.5], "yaw": 0.0},
                      provenance={}, attrs={}, version=1))
        await db.commit()
    return oid, fr


@requires_infra
def test_crosscam_commit_and_revert():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.crosscam_agent import commit_cross_camera, plan_cross_camera
    from services.agent.runs import revert_run
    from sqlalchemy import select

    oid, cam_rt_frame = run_async(_seed_two_cam())

    async def _flow():
        async with get_sessionmaker()() as db:
            plan = await plan_cross_camera(db, oid)
            assert plan["counts"]["targets"] == 1
            assert plan["counts"]["auto_accept"] + plan["counts"]["review"] == 1   # visible in cam_rt
        async with get_sessionmaker()() as db:
            run = await commit_cross_camera(db, oid, created_by="t")
            assert run["created"] == 1
        async with get_sessionmaker()() as db:
            made = (await db.execute(select(Object).where(Object.frame_id == cam_rt_frame, Object.source == "propagated"))).scalars().all()
            assert len(made) == 1
            assert (made[0].provenance or {}).get("target_cam") == "cam_rt"
            assert made[0].cuboid_3d is not None
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(run["run_id"]))
            assert rev["reverted"] == 1
        async with get_sessionmaker()() as db:
            assert not (await db.execute(select(Object).where(Object.frame_id == cam_rt_frame, Object.source == "propagated"))).scalars().all()

    run_async(_flow())
