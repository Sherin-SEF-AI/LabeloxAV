"""Ontology Steward: the greedy appearance clustering (pure), and the end-to-end scan -> proposal ->
approve (mint class + reversible relabel) -> revert cycle."""

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


def test_cluster_splits_two_appearance_groups():
    from services.agent.ontology_steward import _cluster

    rng = np.random.default_rng(0)
    a = _unit(rng, 768); b = _unit(rng, 768)
    rows = []
    for i in range(20):
        rows.append((str(uuid.uuid4()), (a + 0.005 * rng.standard_normal(768)).tolist()))
    for i in range(20):
        rows.append((str(uuid.uuid4()), (b + 0.005 * rng.standard_normal(768)).tolist()))
    clusters = _cluster(rows, sim_thresh=0.8)
    big = sorted((len(c["members"]) for c in clusters), reverse=True)
    assert len(clusters) == 2 and big[0] >= 15 and big[1] >= 15


def _unit(rng, d):
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


async def _seed_fallback_cluster(n: int = 35):
    from db.models import Frame, Object, ObjectEmbedding, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from sqlalchemy import delete
    from db.models import PromotionProposal

    onto = get_ontology()
    fb = onto.fallback_ids()[0]
    maker = get_sessionmaker()
    rng = np.random.default_rng(3)
    base = _unit(rng, 768)
    ts = now_ns()
    ids = []
    async with maker() as db:
        await db.execute(delete(PromotionProposal))
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
        for _ in range(n):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=fb, bbox=[1.0, 1.0, 30.0, 30.0], conf=0.6,
                          source="fused", state="review", attrs={}, provenance={}, version=1))
            await db.flush()
            vec = (base + 0.01 * rng.standard_normal(768)).astype(np.float32)
            db.add(ObjectEmbedding(object_id=oid, dino_vec=(vec / np.linalg.norm(vec)).tolist(), model_versions={}))
            ids.append(str(oid))
        await db.commit()
    return fb, ids


@requires_infra
def test_scan_proposal_approve_revert():
    from db.models import Object, PromotionProposal
    from db.session import get_sessionmaker
    from services.agent.ontology_steward import approve, scan_fallbacks
    from services.agent.runs import revert_run

    fb, ids = run_async(_seed_fallback_cluster(35))

    async def _flow():
        async with get_sessionmaker()() as db:
            res = await scan_fallbacks(db, sample=2000, min_cluster=30, sim_thresh=0.7)
        assert res["proposals"] >= 1
        async with get_sessionmaker()() as db:
            prop = (await db.execute(
                __import__("sqlalchemy").select(PromotionProposal).where(PromotionProposal.status == "proposed")
                .order_by(PromotionProposal.member_count.desc()).limit(1))).scalar_one()
            pid = prop.proposal_id
            assert prop.member_count >= 30
        async with get_sessionmaker()() as db:
            ap = await approve(db, pid, "test_new_vehicle")
        assert ap["relabeled"] >= 30
        async with get_sessionmaker()() as db:
            obj = await db.get(Object, uuid.UUID(ids[0]))
            assert obj.class_id == ap["class_id"]
            prop2 = await db.get(PromotionProposal, pid)
            assert prop2.status == "approved" and prop2.approved_class == ap["class_id"]
        # reversible: the promotion run restores the fallback class
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(ap["run_id"]))
        assert rev["reverted"] >= 30
        async with get_sessionmaker()() as db:
            obj2 = await db.get(Object, uuid.UUID(ids[0]))
            assert obj2.class_id == fb

    run_async(_flow())
