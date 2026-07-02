"""M-MC.2 rig identity (Tier 1): link the same physical object seen in two views into one RigObject with a
voted class; a cross-view class disagreement flags a conflict and routes the non-human members to review; the
DINOv3 appearance assist proposes the cross-camera pair; unlinking dissolves a two-member identity. No
calibration required for this tier. Single asyncio.run so the cached engine binds to one loop."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
def test_rig_identity_link_vote_conflict_and_unlink():
    from sqlalchemy import delete, select
    from db.models import Frame, FrameGroup, Object, ObjectEmbedding, RigObject
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.multicam.rigident import link_objects, rig_objects, suggest_links, unlink_object
    from services.multicam.sync import persist_groups

    sid = uuid.uuid4()
    vec = np.random.default_rng(11).standard_normal(768).astype("float32")
    vec /= np.linalg.norm(vec)

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="RIG-MMC2", start_ts_ns=0, end_ts_ns=1,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            f1 = Frame(session_id=sid, ts_ns=1000, cam_id="cam_f", img_uri="s3://x/1.jpg", width=1920, height=1080)
            f2 = Frame(session_id=sid, ts_ns=1005, cam_id="cam_l", img_uri="s3://x/2.jpg", width=1920, height=1080)
            db.add_all([f1, f2])
            await db.flush()
            # same appearance (same DINOv3 vec) but DIFFERENT class -> a cross-view conflict
            o1 = Object(frame_id=f1.frame_id, class_id=5, bbox=[10, 10, 50, 50], conf=0.9, state="auto_accept", source="fused")
            o2 = Object(frame_id=f2.frame_id, class_id=6, bbox=[60, 60, 100, 100], conf=0.9, state="auto_accept", source="fused")
            db.add_all([o1, o2])
            await db.flush()
            db.add(ObjectEmbedding(object_id=o1.object_id, dino_vec=vec.tolist(), model_versions={}))
            db.add(ObjectEmbedding(object_id=o2.object_id, dino_vec=vec.tolist(), model_versions={}))
            await db.commit()
            oid1, oid2 = o1.object_id, o2.object_id

        await persist_groups(sid)
        async with maker() as db:
            gid = (await db.execute(select(FrameGroup.group_id).where(FrameGroup.session_id == sid))).scalar_one()

        # appearance assist proposes the cross-camera pair
        sug = await suggest_links(sid, gid)
        assert len(sug["suggestions"]) == 1, sug
        s0 = sug["suggestions"][0]
        assert {s0["cam_a"], s0["cam_b"]} == {"cam_f", "cam_l"}

        # link the two into one rig identity -> conflict (classes 5 vs 6)
        res = await link_objects(sid, gid, [oid1, oid2], source="appearance")
        assert res["members"] == 2 and res["conflict"] is True
        assert res["class_id"] in (5, 6)

        async with maker() as db:
            a, b = await db.get(Object, oid1), await db.get(Object, oid2)
            assert a.rig_object_id is not None and a.rig_object_id == b.rig_object_id
            assert a.state == "review" and b.state == "review"  # conflict routed non-human members to review

        listing = await rig_objects(sid, gid)
        assert len(listing["rig_objects"]) == 1 and listing["singletons"] == []
        assert listing["rig_objects"][0]["conflict"] is True
        assert sorted(listing["rig_objects"][0]["cameras"]) == ["cam_f", "cam_l"]

        # unlink one member dissolves the two-member identity
        u = await unlink_object(oid1)
        assert u["dissolved"] is True
        async with maker() as db:
            a, b = await db.get(Object, oid1), await db.get(Object, oid2)
            assert a.rig_object_id is None and b.rig_object_id is None
            assert (await db.execute(select(RigObject).where(RigObject.session_id == sid))).first() is None

        async with maker() as db:
            await db.execute(delete(FrameGroup).where(FrameGroup.session_id == sid))
            await db.delete(await db.get(DbSession, sid))
            await db.commit()

    asyncio.run(run())
