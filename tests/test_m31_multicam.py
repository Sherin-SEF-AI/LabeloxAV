"""M3.1 multi-camera association: the same physical object seen in two overlapping cameras at one instant
gets one consistent rig track id. The corpus is single-camera, so this constructs a synthetic 2-camera
rig session (two cameras, same ts, same DINOv3 appearance) and verifies the association + the calibration
gate. Single asyncio.run so the cached engine binds to one loop (conftest clears caches around the test).
"""

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
def test_multicam_associates_same_object_across_views():
    from db.models import CalibrationValidation, Frame, Object, ObjectEmbedding
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.multicam.associate import associate_session

    sid = uuid.uuid4()
    vec = np.random.default_rng(7).standard_normal(768).astype("float32")
    vec /= np.linalg.norm(vec)

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="RIG-TEST", start_ts_ns=0, end_ts_ns=1,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            f1 = Frame(session_id=sid, ts_ns=1000, cam_id="cam_f", img_uri="s3://x/1.jpg", width=1920, height=1080)
            f2 = Frame(session_id=sid, ts_ns=1005, cam_id="cam_l", img_uri="s3://x/2.jpg", width=1920, height=1080)
            db.add_all([f1, f2])
            await db.flush()
            o1 = Object(frame_id=f1.frame_id, class_id=5, bbox=[10, 10, 50, 50], conf=0.9)
            o2 = Object(frame_id=f2.frame_id, class_id=5, bbox=[60, 60, 100, 100], conf=0.9)
            db.add_all([o1, o2])
            await db.flush()
            # same DINOv3 appearance = same physical object seen in two views
            db.add(ObjectEmbedding(object_id=o1.object_id, dino_vec=vec.tolist(), model_versions={}))
            db.add(ObjectEmbedding(object_id=o2.object_id, dino_vec=vec.tolist(), model_versions={}))
            db.add(CalibrationValidation(session_id=sid, cam_id="cam_f", model="pinhole", status="pass"))
            db.add(CalibrationValidation(session_id=sid, cam_id="cam_l", model="fisheye", status="pass"))
            await db.commit()
            oid1, oid2 = o1.object_id, o2.object_id

        res = await associate_session(sid)
        assert res["rig_tracks"] == 1 and res["associated"] == 2

        async with maker() as db:
            a, b = await db.get(Object, oid1), await db.get(Object, oid2)
            assert a.rig_track_id is not None and a.rig_track_id == b.rig_track_id  # one rig id across views
            await db.delete(await db.get(DbSession, sid))  # cleanup (cascades frames/objects/embeddings)
            await db.commit()

    asyncio.run(run())
