"""P2 CLIP semantic search: cosine ranking (unit) + embeddings compute / text search / find-similar
(GPU integration, skipped without CUDA + infra)."""

from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns
from services.intelligence.embeddings import cosine_topk


def test_cosine_topk_orders_by_similarity():
    q = np.array([1.0, 0.0], dtype=np.float32)
    mat = np.array([[0.0, 1.0], [0.9, 0.1], [1.0, 0.0]], dtype=np.float32)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    top = cosine_topk(q, mat, 3)
    assert top[0][0] == 2  # exact match first
    assert top[-1][0] == 0  # orthogonal last
    assert top[0][1] > top[1][1] > top[2][1]


def _cuda_infra() -> bool:
    try:
        import torch

        import redis as redis_lib

        return torch.cuda.is_available() and bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _cuda_infra(), reason="needs CUDA + infra")


async def _seed_with_images(n=4) -> uuid.UUID:
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid = uuid.uuid4()
    start = now_ns()
    colors = [(0, 0, 200), (0, 200, 0), (200, 0, 0), (200, 200, 0)]
    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-07", start_ts_ns=start, end_ts_ns=start + seconds_to_ns(n),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        await db.flush()
        for i in range(n):
            ts = start + seconds_to_ns(i)
            img = np.full((480, 640, 3), colors[i % len(colors)], dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", img)
            uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
            db.add(Frame(session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri,
                         width=640, height=480, quality=0.9))
        await db.commit()
        # one object per frame
        frames = (await db.execute(__import__("sqlalchemy").select(Frame).where(Frame.session_id == sid))).scalars().all()
        for f in frames:
            db.add(Object(frame_id=f.frame_id, class_id=6, bbox=[50, 50, 590, 430], conf=0.8, attrs={},
                          source="fused", state="auto_accept", provenance={}))
        await db.commit()
    return sid


@requires_gpu
@pytest.mark.asyncio
async def test_embeddings_compute_text_search_and_similar():
    from db.models import Embedding
    from db.session import get_sessionmaker
    from sqlalchemy import func, select
    from services.intelligence.embeddings import (
        compute_session_embeddings,
        search_objects_by_text,
        similar_objects,
    )

    sid = await _seed_with_images(4)
    res = await compute_session_embeddings(sid)
    assert res["embedded"] == 4

    maker = get_sessionmaker()
    async with maker() as db:
        n = (await db.execute(select(func.count()).select_from(Embedding))).scalar_one()
        assert n >= 4

        hits = await search_objects_by_text(db, "a red traffic object", limit=4, session_id=str(sid))
        assert len(hits) == 4
        assert all(-1.01 <= h["score"] <= 1.01 for h in hits)
        assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)

        first = hits[0]["object_id"]
        sim = await similar_objects(db, first, limit=3)
        assert all(s["object_id"] != first for s in sim)  # self excluded
        assert all("image_url" in s for s in sim)
