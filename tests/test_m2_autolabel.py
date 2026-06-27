"""M2 GPU smoke test. Requires infra up and a CUDA GPU. Seeds a session with real frames in
MinIO, runs Stage 1 (YOLO26 + SAM 3.1) and asserts the pipeline runs and peak VRAM stays under
the 16 GB ceiling. Skipped automatically when CUDA or the ml extra is unavailable.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns
from db.models import Frame
from db.models import Session as DbSession
from db.session import get_sessionmaker

pytestmark = pytest.mark.asyncio


def _cuda_ready() -> bool:
    try:
        import torch  # noqa: F401

        import torch as t

        return t.cuda.is_available()
    except Exception:
        return False


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not (_cuda_ready() and _infra_up()), reason="needs CUDA GPU + infra (ml extra, make up)"
)


async def _seed_session(n_frames: int = 3) -> uuid.UUID:
    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid = uuid.uuid4()
    start = now_ns()
    rng = np.random.default_rng(3)

    async with maker() as db:
        db.add(
            DbSession(
                session_id=sid,
                vehicle_id="TIGOR-TEST",
                start_ts_ns=start,
                end_ts_ns=start + seconds_to_ns(n_frames),
                city="BLR",
                sensors={"cam_f": {"serial": "x", "calibration_hash": "y"}},
                ontology_version="labelox-in-0.1.0",
            )
        )
        await db.flush()
        for i in range(n_frames):
            ts = start + seconds_to_ns(i)
            img = rng.integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            key = f"frames/{sid}/cam_f/{ts}.jpg"
            uri = store.put_bytes(key, buf.tobytes(), "image/jpeg")
            db.add(
                Frame(
                    session_id=sid,
                    ts_ns=ts,
                    cam_id="cam_f",
                    img_uri=uri,
                    width=640,
                    height=480,
                    quality=0.8,
                )
            )
        await db.commit()
    return sid


@requires_gpu
async def test_stage1_runs_under_vram_ceiling():
    from services.autolabel.runner import process_session

    sid = await _seed_session(2)

    seen = {"frames": 0}

    async def on_frame(fd) -> None:
        seen["frames"] += 1
        # masks from Path B must be 2D boolean arrays sized to the frame
        for d in fd.dets_b:
            if d.mask is not None:
                assert d.mask.dtype == bool
                assert d.mask.shape == (fd.frame.height, fd.frame.width)

    summary = await process_session(sid, limit=2, on_frame=on_frame)

    assert seen["frames"] == 2
    assert summary["frames"] == 2
    assert 0 < summary["peak_vram_mb"] <= summary["vram_ceiling_mb"]
