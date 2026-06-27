"""M1 integration tests. Require infra up (make up). Synthesize a real MP4 + sidecar and a real
protobuf MCAP, ingest both, and assert frames in MinIO, rows in Postgres, manifest written.
"""

from __future__ import annotations

import csv
import uuid

import cv2
import numpy as np
import pytest
from sqlalchemy import func, select

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns
from db.models import Frame
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.ingest.reader_mcap import read_mcap
from services.ingest.reader_video import read_video
from services.ingest.run import ingest

pytestmark = pytest.mark.asyncio


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (run: make up)")


def _make_video(path, n=30, fps=10, w=640, h=480) -> float:
    # Random-noise frames: high Laplacian variance so they pass the blur gate.
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "VideoWriter failed to open (codec unavailable)"
    for _ in range(n):
        frame = rng.integers(40, 220, size=(h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return float(fps)


def _make_sidecar(path, start_ns, fps, n) -> None:
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["ts_ns", "lat", "lon", "ego_speed"])
        for i in range(n):
            ts = start_ns + seconds_to_ns(i / fps)
            wr.writerow([ts, 12.97 + i * 1e-4, 77.59 + i * 1e-4, 8.5])


@requires_infra
async def test_video_ingest(tmp_path):
    video = tmp_path / "clip.mp4"
    sidecar = tmp_path / "side.csv"
    start = now_ns()
    fps = _make_video(video, n=30, fps=10)
    _make_sidecar(sidecar, start, fps, 30)

    frame_iter = read_video(video, "cam_f", start, target_fps=3.0, sidecar_path=sidecar)
    result = await ingest(
        frame_iter=frame_iter,
        vehicle="TIGOR-07",
        city="BLR",
        route="BLR-EAST",
        raw_uri="s3://test/raw/clip.mp4",
        mcap_uri=None,
        source_streams=["cam_f"],
    )

    assert result["n_frames"] > 0
    sid = uuid.UUID(result["session_id"])

    store = get_object_store()
    maker = get_sessionmaker()
    async with maker() as db:
        sess = await db.get(DbSession, sid)
        assert sess is not None
        assert sess.start_ts_ns <= sess.end_ts_ns
        assert sess.ontology_version == "labelox-in-0.1.0"
        assert sess.manifest_uri and store.exists(sess.manifest_uri)
        assert "cam_f" in sess.sensors

        count = (
            await db.execute(select(func.count()).select_from(Frame).where(Frame.session_id == sid))
        ).scalar_one()
        assert count == result["n_frames"]

        frame = (
            await db.execute(select(Frame).where(Frame.session_id == sid).limit(1))
        ).scalar_one()
        assert isinstance(frame.ts_ns, int)
        assert store.exists(frame.img_uri)
        assert frame.gnss is not None  # GNSS attached from sidecar
        assert frame.quality > 0.0


@requires_infra
async def test_mcap_ingest(tmp_path):
    foxglove = pytest.importorskip("foxglove_schemas_protobuf")
    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
    from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
    from mcap_protobuf.writer import Writer

    mcap_path = tmp_path / "session.mcap"
    rng = np.random.default_rng(11)
    start = now_ns()

    with open(mcap_path, "wb") as fh, Writer(fh) as writer:
        for i in range(20):
            ts = start + seconds_to_ns(i / 10.0)
            img = rng.integers(40, 220, size=(480, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            ci = CompressedImage()
            ci.timestamp.FromNanoseconds(ts)
            ci.frame_id = "cam_f"
            ci.format = "jpeg"
            ci.data = buf.tobytes()
            writer.write_message(topic="/camera/cam_f", message=ci, log_time=ts, publish_time=ts)

            fix = LocationFix()
            fix.latitude = 12.97 + i * 1e-4
            fix.longitude = 77.59 + i * 1e-4
            writer.write_message(topic="/gnss", message=fix, log_time=ts, publish_time=ts)

    frame_iter = read_mcap(mcap_path, target_fps=3.0)
    result = await ingest(
        frame_iter=frame_iter,
        vehicle="TIGOR-07",
        city="BLR",
        route=None,
        raw_uri=None,
        mcap_uri="s3://test/raw/session.mcap",
        source_streams=["mcap"],
    )

    assert result["n_frames"] > 0
    sid = uuid.UUID(result["session_id"])
    maker = get_sessionmaker()
    async with maker() as db:
        frame = (
            await db.execute(select(Frame).where(Frame.session_id == sid).limit(1))
        ).scalar_one()
        assert frame.cam_id == "cam_f"
        # Forward-filled GNSS: the very first frame may precede the first fix, so assert the
        # session as a whole carried GNSS through to the frames.
        with_gnss = (
            await db.execute(
                select(func.count()).select_from(Frame).where(Frame.session_id == sid, Frame.gnss.isnot(None))
            )
        ).scalar_one()
        assert with_gnss > 0
