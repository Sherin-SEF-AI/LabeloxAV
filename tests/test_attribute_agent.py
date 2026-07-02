"""Auto-attribute fill: derive occlusion/truncation, fill valid attrs, revert restores prior attrs."""

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


def test_occlusion_and_truncation_rules():
    from services.agent.attribute_agent import _occlusion, _truncation

    class _B:
        def __init__(self, b):
            self.bbox = b
            self.object_id = uuid.uuid4()

    far = [100, 100, 200, 300]
    near = [120, 250, 220, 400]   # nearer (lower), overlaps far's lower part
    assert _occlusion(far, [_B(near)]) > 0        # far is occluded by the nearer box
    assert _occlusion(near, [_B(far)]) == 0       # nearer box is not occluded by the one behind
    assert _truncation([0, 100, 80, 300], 1920, 1080) > 0   # touches the left edge
    assert _truncation([800, 400, 900, 600], 1920, 1080) == 0


async def _seed_two_overlapping():
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    far_id, near_id = uuid.uuid4(), uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(2).integers(30, 220, size=(1080, 1920, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="ATTR-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=1920, height=1080, quality=0.9))
        db.add(Object(object_id=far_id, frame_id=fid, class_id=sedan, bbox=[600.0, 400.0, 800.0, 700.0], conf=0.9,
                      source="fused", state="review", provenance={}, attrs={}, version=1))
        db.add(Object(object_id=near_id, frame_id=fid, class_id=sedan, bbox=[650.0, 600.0, 900.0, 950.0], conf=0.9,
                      source="fused", state="review", provenance={}, attrs={}, version=1))
        await db.commit()
    return fid, far_id


@requires_infra
def test_attribute_commit_and_revert():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.attribute_agent import commit_attributes, plan_attributes
    from services.agent.runs import revert_run

    fid, far_id = run_async(_seed_two_overlapping())

    async def _flow():
        async with get_sessionmaker()() as db:
            plan = await plan_attributes(db, fid)
            assert plan["counts"]["attrs_filled"] > 0
            assert "occlusion" in plan["counts"]["by_attr"]
        async with get_sessionmaker()() as db:
            run = await commit_attributes(db, fid, created_by="t")
            assert run["objects_updated"] >= 1
        async with get_sessionmaker()() as db:
            far = await db.get(Object, far_id)
            assert "occlusion" in (far.attrs or {}) and far.attrs["occlusion"] > 0   # occluded by the nearer box
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(run["run_id"]))
            assert rev["reverted"] == run["objects_updated"]
        async with get_sessionmaker()() as db:
            assert (await db.get(Object, far_id)).attrs == {}

    run_async(_flow())
