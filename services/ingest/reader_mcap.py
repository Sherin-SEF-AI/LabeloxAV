"""MCAP reader. MCAP is the native container (Principle 07): sensor streams in one indexed,
timestamped file. Camera images plus GNSS/IMU/CAN side channels are demuxed and each frame is
mapped to UTC nanoseconds from the message log time.

Protobuf messages are decoded generically from the schema embedded in the file, so this works
with Foxglove well-known schemas (CompressedImage, LocationFix) without compile-time stubs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.ingest.types import RawFrame

log = get_logger(__name__)


def _decode_image(msg) -> np.ndarray | None:
    data = getattr(msg, "data", None)
    if not data:
        return None
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _is_image_schema(name: str) -> bool:
    return "CompressedImage" in name or name.endswith(".Image")


def _is_gnss_schema(name: str) -> bool:
    return "LocationFix" in name or "NavSatFix" in name


def read_mcap(
    mcap_path: str | Path,
    target_fps: float,
    default_cam_id: str = "cam_f",
) -> Iterator[RawFrame]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    mcap_path = Path(mcap_path)
    if not mcap_path.exists():
        raise FileNotFoundError(mcap_path)

    settings = get_settings()
    interval_ns = int(1e9 / target_fps) if target_fps > 0 else 0
    last_kept: dict[str, int] = {}

    # Forward-filled latest side-channel state, attached to camera frames as they arrive.
    cur_lat: float | None = None
    cur_lon: float | None = None
    cur_speed: float | None = None

    with open(mcap_path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto in reader.iter_decoded_messages():
            ts_ns = int(message.log_time)
            sname = schema.name if schema else ""

            if _is_gnss_schema(sname):
                cur_lat = getattr(proto, "latitude", cur_lat)
                cur_lon = getattr(proto, "longitude", cur_lon)
                continue

            # Vehicle speed: a float-bearing message on a speed topic.
            if "speed" in channel.topic.lower():
                for attr in ("ego_speed", "speed", "value"):
                    if hasattr(proto, attr):
                        cur_speed = float(getattr(proto, attr))
                        break
                continue

            if _is_image_schema(sname):
                cam_id = channel.topic.strip("/").split("/")[-1] or default_cam_id
                if interval_ns and (ts_ns - last_kept.get(cam_id, -interval_ns)) < interval_ns:
                    continue
                img = _decode_image(proto)
                if img is None:
                    log.warning("mcap.image_decode_failed", topic=channel.topic, ts_ns=ts_ns)
                    continue
                last_kept[cam_id] = ts_ns
                yield RawFrame(
                    ts_ns=ts_ns,
                    cam_id=cam_id,
                    image_bgr=img,
                    lat=cur_lat,
                    lon=cur_lon,
                    ego_speed=cur_speed,
                )
