"""Annotation Copilot: detect a repeated correction pattern, find similar still-wrong cases via embeddings,
batch-fix them to review reversibly."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
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


def _unit(rng):
    v = rng.standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)


async def _seed_pattern():
    """A reviewer corrected e_auto -> autorickshaw several times; there are more e_auto that look the same."""
    from db.models import Frame, Object, ObjectEmbedding, OntologyClass, OntologyVersion, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from sqlalchemy import delete

    onto = get_ontology()
    e_auto = next(c.id for c in onto.classes if c.name == "e_auto")
    auto = next(c.id for c in onto.classes if c.name == "autorickshaw")
    maker = get_sessionmaker()
    rng = np.random.default_rng(11)
    base = _unit(rng)
    ts = now_ns()
    corrected, similar = [], []
    async with maker() as db:
        await db.execute(delete(Review))
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        sid, fid = uuid.uuid4(), uuid.uuid4()
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/f.jpg", width=320,
                     height=240, quality=0.9, scene={}))
        await db.flush()

        async def _obj(class_id):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=class_id, bbox=[1.0, 1.0, 30.0, 30.0], conf=0.6,
                          source="fused", state="auto_accept", attrs={}, provenance={}, version=1))
            await db.flush()
            vec = (base + 0.01 * rng.standard_normal(768)).astype(np.float32)
            db.add(ObjectEmbedding(object_id=oid, dino_vec=(vec / np.linalg.norm(vec)).tolist(), model_versions={}))
            return oid

        for _ in range(4):   # the reviewer's corrections: e_auto -> autorickshaw
            oid = await _obj(auto)
            corrected.append(str(oid))
            db.add(Review(object_id=oid, reviewer="rev", user_id=None, action="reclassify",
                          before={"class_id": e_auto}, after={"class_id": auto}, time_spent_ms=1000, ts_ns=ts))
        for _ in range(5):   # more e_auto that look the same and are still mislabeled
            similar.append(str(await _obj(e_auto)))
        await db.commit()
    return e_auto, auto, corrected, similar


@requires_infra
def test_pattern_similar_batch_revert():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.annotation_copilot import apply_batch, suggest_for_reviewer
    from services.agent.runs import revert_run

    e_auto, auto, corrected, similar = run_async(_seed_pattern())

    async def _flow():
        async with get_sessionmaker()() as db:
            sug = await suggest_for_reviewer(db, None)
        assert sug["pattern"] and sug["pattern"]["from_name"] == "e_auto" and sug["pattern"]["to_name"] == "autorickshaw"
        assert sug["pattern"]["count"] >= 3
        cands = sug["candidates"]
        assert set(cands) & set(similar)          # found the still-mislabeled look-alikes
        assert not (set(cands) & set(corrected))  # excluded the already-corrected examples
        async with get_sessionmaker()() as db:
            r = await apply_batch(db, cands, auto)
        assert r["relabeled"] >= 1
        async with get_sessionmaker()() as db:
            obj = await db.get(Object, uuid.UUID(cands[0]))
            assert obj.class_id == auto and obj.state == "review"
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(r["run_id"]))
        assert rev["reverted"] >= 1
        async with get_sessionmaker()() as db:
            obj2 = await db.get(Object, uuid.UUID(cands[0]))
            assert obj2.class_id == e_auto and obj2.state == "auto_accept"

    run_async(_flow())
