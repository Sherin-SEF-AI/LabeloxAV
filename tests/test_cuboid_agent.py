"""2D->3D auto-cuboid agent: monocular fit + reprojection validation, attach + reversible revert."""

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


def test_fit_mono_returns_valid_cuboid():
    """Pure: a sedan box near the bottom lifts to a plausible ground cuboid with a positive reprojection IoU."""
    from services.agent.cuboid_agent import _fit_mono
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    sedan = next(c.id for c in onto.classes if c.name == "sedan")

    class _O:
        class_id = sedan
        bbox = [820.0, 620.0, 1080.0, 860.0]

    fit = _fit_mono(_O(), onto, "cam_front", 1920, 1080)
    assert fit is not None
    cuboid, iou = fit
    assert iou > 0.0 and cuboid["size"] == [1.8, 4.2, 1.5]
    assert cuboid["center"][0] > 0            # ahead of the ego
    assert abs(cuboid["center"][2] - 0.75) < 1e-6   # sits on the ground (z = height/2)


async def _seed_frame_with_vehicle():
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, fid, oid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    start = now_ns()
    img = np.random.default_rng(9).integers(30, 220, size=(1080, 1920, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_front/{start}.jpg", buf.tobytes(), "image/jpeg")
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="CUB-01", start_ts_ns=start, end_ts_ns=start + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_front", img_uri=uri, width=1920, height=1080, quality=0.9))
        db.add(Object(object_id=oid, frame_id=fid, class_id=sedan, bbox=[820.0, 620.0, 1080.0, 860.0],
                      conf=0.9, source="fused", state="review", provenance={}, attrs={}, version=1))
        await db.commit()
    return fid, oid


@requires_infra
def test_cuboid_commit_and_revert():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.cuboid_agent import commit_cuboids, plan_cuboids
    from services.agent.runs import revert_run

    fid, oid = run_async(_seed_frame_with_vehicle())

    async def _flow():
        async with get_sessionmaker()() as db:
            plan = await plan_cuboids(db, fid, min_iou=0.0)   # accept any positive fit for the test
            assert plan["counts"]["total"] == 1
            assert plan["counts"]["auto_accept"] + plan["counts"]["review"] == 1
        async with get_sessionmaker()() as db:
            run = await commit_cuboids(db, fid, min_iou=0.0, created_by="t")
            assert run["attached"] == 1
        async with get_sessionmaker()() as db:
            o = await db.get(Object, oid)
            assert o.cuboid_3d is not None and o.cuboid_3d["size"] == [1.8, 4.2, 1.5]
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(run["run_id"]))
            assert rev["reverted"] == 1
        async with get_sessionmaker()() as db:
            assert (await db.get(Object, oid)).cuboid_3d is None

    run_async(_flow())
