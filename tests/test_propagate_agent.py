"""Track auto-propagation agent: propagate a keyframe box across shifted frames, then revert (delete)."""

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


async def _seed_ontology(db):
    from db.models import OntologyClass, OntologyVersion
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is not None:
        return
    db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
    await db.flush()
    for c in onto.classes:
        db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
    await db.flush()


async def _seed_clip(n_frames=6, shift=4):
    """A session of n frames: the same textured noise shifted `shift` px/frame (so optical flow has real
    motion to track), plus a source object box on the first frame. Returns (session, first_frame, obj_id)."""
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid = uuid.uuid4()
    start = now_ns()
    rng = np.random.default_rng(11)
    base = rng.integers(0, 255, size=(480, 720, 3), dtype=np.uint8)  # textured -> trackable corners
    cls = [c.id for c in __import__("services.autolabel.ontology", fromlist=["get_ontology"]).get_ontology().classes if c.l1 == "four_wheeler"][0]
    frames = []
    async with maker() as db:
        await _seed_ontology(db)
        db.add(DbSession(session_id=sid, vehicle_id="PROP-01", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(2), city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        oid = None
        for i in range(n_frames):
            fid = uuid.uuid4()
            img = np.roll(base, i * shift, axis=1)  # shift right by i*shift px
            _ok, buf = cv2.imencode(".jpg", img)
            uri = store.put_bytes(f"frames/{sid}/cam_f/{start + i}.jpg", buf.tobytes(), "image/jpeg")
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start + i * 1_000_000, cam_id="cam_f",
                         img_uri=uri, width=720, height=480, quality=0.9))
            frames.append(fid)
            if i == 0:
                oid = uuid.uuid4()
                db.add(Object(object_id=oid, frame_id=fid, class_id=cls, bbox=[300.0, 190.0, 430.0, 320.0],
                              conf=1.0, source="human", state="accepted", provenance={}, attrs={}, version=1))
        await db.commit()
    return sid, frames[0], oid


@requires_infra
def test_propagate_and_revert(monkeypatch):
    import services.agent.propagate_agent as pa
    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.propagate_agent import commit_propagate, plan_propagate
    from services.agent.runs import revert_run

    # geometry-only (skip DINOv3) so the test is deterministic + GPU-free
    monkeypatch.setattr(pa, "_source_vec", lambda *a, **k: None)

    _sid, _f0, oid = run_async(_seed_clip())

    async def _flow():
        async with get_sessionmaker()() as db:
            plan = await plan_propagate(db, oid, span=10)
            assert plan["counts"]["total_steps"] > 0
            assert plan["forward"] >= 1          # boxes carried forward along the shift
        async with get_sessionmaker()() as db:
            run = await commit_propagate(db, oid, span=10, created_by="t")
            assert run["created"] >= 1
        async with get_sessionmaker()() as db:
            made = (await db.execute(
                __import__("sqlalchemy").select(Object).where(Object.source == "propagated")
            )).scalars().all()
            assert made and all(o.state in ("auto_accept", "review") for o in made)
            assert all((o.provenance or {}).get("propagated_from") == str(oid) for o in made)
        # revert deletes the propagated boxes
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(run["run_id"]))
            assert rev["reverted"] == run["created"]
        async with get_sessionmaker()() as db:
            left = (await db.execute(
                __import__("sqlalchemy").select(Object).where(Object.source == "propagated")
            )).scalars().all()
            assert not left

    run_async(_flow())
