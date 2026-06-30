"""A point cloud channel in MCAP. MCAP is the native container (Principle 07): clouds round-trip alongside
the existing camera, GNSS, IMU, and CAN channels in one indexed, timestamped file, on the same UTC-ns base.

Each cloud is one message on the /points topic; the message bytes are the canonical compressed Cloud
serialization and the log_time is the cloud ts_ns, so a reader needs no compile-time stubs and the clouds
stay time-aligned with the other sensor streams.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from services.lidar.ingest.normalize import Cloud

POINTS_TOPIC = "/points"
SCHEMA_NAME = "labelox.PointCloud"
MESSAGE_ENCODING = "labelox-npz"

_SCHEMA = json.dumps({
    "title": SCHEMA_NAME,
    "description": "Compressed npz holding xyz (N,3) float32, intensity (N,) float32, optional ring (N,) "
                   "int16, and a json metadata sidecar (ts_ns, source, frame, depth_model, calibration).",
}).encode("utf-8")


def write_pointclouds_mcap(clouds: Iterable[Cloud], path: str | Path) -> dict:
    """Write clouds to an MCAP file on the /points channel, ordered by ts_ns."""
    from mcap.writer import Writer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()
        schema_id = writer.register_schema(name=SCHEMA_NAME, encoding="jsonschema", data=_SCHEMA)
        channel_id = writer.register_channel(topic=POINTS_TOPIC, message_encoding=MESSAGE_ENCODING,
                                             schema_id=schema_id)
        for cloud in sorted(clouds, key=lambda c: c.ts_ns):
            writer.add_message(channel_id=channel_id, log_time=int(cloud.ts_ns),
                               publish_time=int(cloud.ts_ns), data=cloud.to_npz_bytes())
            n += 1
        writer.finish()
    return {"path": str(path), "clouds": n, "topic": POINTS_TOPIC}


def read_pointclouds_mcap(path: str | Path) -> Iterator[Cloud]:
    """Yield each cloud on the /points channel, with its ts_ns taken from the message log time."""
    from mcap.reader import make_reader

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(topics=[POINTS_TOPIC]):
            cloud = Cloud.from_npz_bytes(message.data)
            cloud.ts_ns = int(message.log_time)
            yield cloud
