"""SEC-M2: the static-camera ingestion adapter + the schema that lets a session have no ego vehicle.

The clip is generated procedurally with OpenCV (never downloaded). The schema round-trip proves migration
0068: a session can carry pack_id='sec' with a null vehicle_id, and an unspecified pack_id backfills to 'av'.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from packs.base import FrameDraft, IngestSource, SessionDraft
from packs.sec.ingest import StaticCameraIngestionAdapter

pytestmark = pytest.mark.db


def _write_clip(path: Path, n: int = 10, fps: int = 10, size: int = 32) -> bool:
    """Write a short MJPG/AVI clip: a grey scene with a moving square. Returns False if no writer codec."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not writer.isOpened():
        return False
    for i in range(n):
        f = np.full((size, size, 3), 90, dtype=np.uint8)
        x = 1 + (i % (size - 5))
        f[x:x + 4, x:x + 4] = 255
        writer.write(f)
    writer.release()
    return path.exists() and path.stat().st_size > 0


def test_adapter_can_handle_only_its_media_types():
    ad = StaticCameraIngestionAdapter()
    assert ad.can_handle(IngestSource("video", "x.mp4", "cam_1"))
    assert not ad.can_handle(IngestSource("mcap", "x.mcap", "cam_1"))


def test_adapter_reads_a_clip_into_null_ego_drafts(tmp_path):
    clip = tmp_path / "cctv.avi"
    if not _write_clip(clip):
        pytest.skip("no OpenCV VideoWriter codec available")

    ad = StaticCameraIngestionAdapter()
    session, frames = ad.read(IngestSource("video", str(clip), "cam_lobby", meta={"fps": 5, "city": "BLR"}))

    assert isinstance(session, SessionDraft)
    assert session.pack_id == "sec"
    assert session.vehicle_id is None          # the fork: no ego vehicle
    assert session.cam_id == "cam_lobby"
    assert session.city == "BLR"

    drafts = list(frames)
    assert len(drafts) >= 2
    assert all(isinstance(d, FrameDraft) for d in drafts)
    assert all(d.cam_id == "cam_lobby" for d in drafts)
    assert all(d.ego_speed is None for d in drafts)          # a fixed camera has no CAN speed
    ts = [d.ts_ns for d in drafts]
    assert ts == sorted(ts)


def test_adapter_rejects_unhandled_source():
    with pytest.raises(ValueError):
        StaticCameraIngestionAdapter().read(IngestSource("rtsp", "rtsp://x", "cam_1"))


async def test_static_camera_session_persists_with_null_vehicle():
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    ver = get_ontology("av").version
    sec_id, av_id = uuid4(), uuid4()
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sec_id, pack_id="sec", vehicle_id=None,
                         start_ts_ns=0, end_ts_ns=1, ontology_version=ver))
        # No pack_id: the server default must backfill it to 'av'.
        db.add(DbSession(session_id=av_id, vehicle_id="veh_1",
                         start_ts_ns=0, end_ts_ns=1, ontology_version=ver))
        await db.commit()

    async with get_sessionmaker()() as db:
        sec = await db.get(DbSession, sec_id)
        av = await db.get(DbSession, av_id)
        assert sec.pack_id == "sec" and sec.vehicle_id is None
        assert av.pack_id == "av" and av.vehicle_id == "veh_1"
