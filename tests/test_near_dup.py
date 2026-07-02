"""Near-duplicate consistency: an object present in a frame but absent in its near-identical twin is flagged."""

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


async def _seed_near_dups():
    """Two frames with an identical DINOv3 vector (so they read as near-duplicates). Frame A has a pedestrian
    plus an extra object; frame B has only the pedestrian. The extra object in A should be flagged."""
    from db.models import Frame, FrameEmbedding, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, fa, fb = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    extra_oid = uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(1).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    ped = next(c.id for c in onto.classes if c.name == "pedestrian")
    pole = next((c.id for c in onto.classes if c.name in ("pole", "utility_pole", "light_pole")), None) or next(c.id for c in onto.classes if c.l1 == "fixed")
    vec = [1.0] + [0.0] * 767   # unit vector; both frames identical -> cosine sim 1.0
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="ND-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        for f in (fa, fb):
            uri = store.put_bytes(f"frames/{sid}/cam_f/{f}.jpg", buf.tobytes(), "image/jpeg")
            db.add(Frame(frame_id=f, session_id=sid, ts_ns=ts + (0 if f == fa else 1), cam_id="cam_f",
                         img_uri=uri, width=640, height=480, quality=0.9))
        await db.flush()   # frames must exist before their embeddings (FK)
        for f in (fa, fb):
            db.add(FrameEmbedding(frame_id=f, dino_vec=vec, siglip_vec=None, model_versions={}))
        # A: pedestrian + an extra object; B: pedestrian only
        db.add(Object(object_id=uuid.uuid4(), frame_id=fa, class_id=ped, bbox=[10.0, 10.0, 40.0, 90.0], conf=0.9, source="fused", state="review", provenance={}, attrs={}, version=1))
        db.add(Object(object_id=extra_oid, frame_id=fa, class_id=pole, bbox=[100.0, 10.0, 130.0, 90.0], conf=0.9, source="fused", state="review", provenance={}, attrs={}, version=1))
        db.add(Object(object_id=uuid.uuid4(), frame_id=fb, class_id=ped, bbox=[12.0, 10.0, 42.0, 90.0], conf=0.9, source="fused", state="review", provenance={}, attrs={}, version=1))
        await db.commit()
    return str(sid), str(extra_oid)


@requires_infra
def test_near_dup_flags_extra_object():
    from services.errordetect.near_dup import detect_near_dup_inconsistent
    from db.session import get_sessionmaker

    sid, extra_oid = run_async(_seed_near_dups())

    async def _flow():
        async with get_sessionmaker()() as db:
            found = await detect_near_dup_inconsistent(db, sid, sim_thresh=0.9)
        ids = {c["object_id"] for c in found}
        assert extra_oid in ids     # the object absent from the near-identical twin is flagged
        cand = next(c for c in found if c["object_id"] == extra_oid)
        assert cand["kind"] == "near_dup_inconsistent" and cand["score"] >= 0.9

    run_async(_flow())
