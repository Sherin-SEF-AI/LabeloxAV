"""Temporal auto-repair: relabel a strong-majority track outlier, skip corrupt static-majority tracks, revert."""

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


async def _seed_tracks():
    """Track A: 4 sedan + 1 outlier vehicle (movable majority -> repairable). Track B: 3 static-class + 1
    sedan (static majority -> a corrupt track, must be skipped). Returns (session_id, outlier_object_id)."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid = uuid.uuid4()
    ts0 = now_ns()
    img = np.random.default_rng(8).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    outlier_cls = next(c.id for c in onto.classes if c.l1 == "two_wheeler")
    static_cls = next((c.id for c in onto.classes if c.l1 in ("fixed", "boundary")), None)
    tA, tB = uuid.uuid4(), uuid.uuid4()
    outlier_oid = uuid.uuid4()
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="TR-01", start_ts_ns=ts0, end_ts_ns=ts0 + seconds_to_ns(2),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Track(track_id=tA, session_id=sid, class_id=sedan, first_ts_ns=ts0, last_ts_ns=ts0 + 5))
        db.add(Track(track_id=tB, session_id=sid, class_id=static_cls or sedan, first_ts_ns=ts0, last_ts_ns=ts0 + 4))
        # Track A: 5 frames, class ids [sedan,sedan,sedan,sedan,outlier]
        for i, cls in enumerate([sedan, sedan, sedan, sedan, outlier_cls]):
            fid = uuid.uuid4()
            uri = store.put_bytes(f"frames/{sid}/cam_f/A{i}.jpg", buf.tobytes(), "image/jpeg")
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts0 + i, cam_id="cam_f", img_uri=uri, width=640, height=480, quality=0.9))
            oid = outlier_oid if cls == outlier_cls else uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, track_id=tA, class_id=cls, bbox=[10.0, 10.0, 50.0, 50.0],
                          conf=0.8, source="fused", state="review", provenance={}, attrs={}, version=1))
        # Track B: static-majority corrupt track
        if static_cls is not None:
            for i, cls in enumerate([static_cls, static_cls, static_cls, sedan]):
                fid = uuid.uuid4()
                uri = store.put_bytes(f"frames/{sid}/cam_f/B{i}.jpg", buf.tobytes(), "image/jpeg")
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts0 + 10 + i, cam_id="cam_f", img_uri=uri, width=640, height=480, quality=0.9))
                db.add(Object(object_id=uuid.uuid4(), frame_id=fid, track_id=tB, class_id=cls, bbox=[60.0, 60.0, 100.0, 100.0],
                              conf=0.8, source="fused", state="review", provenance={}, attrs={}, version=1))
        await db.commit()
    return str(sid), str(outlier_oid), sedan


@requires_infra
def test_temporal_repair_fixes_flip_skips_corrupt():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.temporal_repair import commit_temporal_repair, plan_temporal_repair
    from services.agent.runs import revert_run

    sid, outlier_oid, sedan = run_async(_seed_tracks())

    async def _flow():
        async with get_sessionmaker()() as db:
            plan = await plan_temporal_repair(db, sid, min_majority=0.8)
            # the movable-majority outlier is repaired; the static-majority corrupt track is not
            assert any(i["object_id"] == outlier_oid and i["to_class"] == sedan for i in plan["items"])
            assert not any(i["to_name"] in ("tree", "bus_shelter") for i in plan["items"])
        async with get_sessionmaker()() as db:
            run = await commit_temporal_repair(db, sid, min_majority=0.8, created_by="t")
            assert run["relabeled"] >= 1
        async with get_sessionmaker()() as db:
            assert int((await db.get(Object, uuid.UUID(outlier_oid))).class_id) == sedan
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(run["run_id"]))
            assert rev["reverted"] == run["relabeled"]
        async with get_sessionmaker()() as db:
            assert int((await db.get(Object, uuid.UUID(outlier_oid))).class_id) != sedan

    run_async(_flow())
