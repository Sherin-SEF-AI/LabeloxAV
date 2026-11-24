"""Point cloud channels in MCAP. MCAP is the native container (Principle 07): clouds round-trip alongside
the existing camera, GNSS, IMU, and CAN channels in one indexed, timestamped file, on the same UTC-ns base.

Three encodings are read, and `read_pointclouds_mcap` tells them apart per channel:

  labelox-npz                   our own: one message per cloud on the /points topic, the bytes being the
                                canonical compressed Cloud serialization. This is what
                                `write_pointclouds_mcap` writes.
  sensor_msgs/msg/PointCloud2   ROS 2, CDR-encoded (message_encoding "cdr"), as rosbag2 records it.
  sensor_msgs/PointCloud2       ROS 1 (message_encoding "ros1"), as `mcap convert` writes a .bag.

The two ROS forms are deserialized by services.lidar.ingest.pointcloud2 from the message's fixed layout,
not through mcap-ros2-support's schema-driven decoder, so no decoder library is needed (there is none
installed for ROS 1) and a channel whose schema text is missing still reads.

A cloud's ts_ns is the message log time, as for every other channel this codebase reads from MCAP
(services.ingest.reader_mcap), so clouds stay time-aligned with the camera frames of the same file.
`time_source="header"` uses the PointCloud2 header stamp instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.pointcloud2 import (
    ROS1_SCHEMA,
    ROS2_SCHEMA,
    decode_pointcloud2,
    parse_pointcloud2_cdr,
    parse_pointcloud2_ros1,
)

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


def _channel_kind(schema, channel) -> str | None:
    """Which reader a channel's messages need: "npz", "cdr", "ros1", or None for a channel that does not
    carry point clouds. The schema name says it is a PointCloud2; the channel's message encoding says how
    it was serialized (a ROS 1 bag converted to MCAP keeps the ROS 1 type name, so the name alone is not
    enough, and neither is the encoding)."""
    encoding = (channel.message_encoding or "").lower()
    name = schema.name if schema is not None else ""
    if encoding == MESSAGE_ENCODING or name == SCHEMA_NAME:
        return "npz"
    if name not in (ROS2_SCHEMA, ROS1_SCHEMA):
        return None
    if encoding == "cdr":
        return "cdr"
    if encoding == "ros1":
        return "ros1"
    return None


def pointcloud_topics_mcap(path: str | Path) -> dict[str, str]:
    """{topic: kind} for every channel of the file that carries point clouds, from the summary section
    (falling back to nothing for a file written without one; `read_pointclouds_mcap` still scans those)."""
    from mcap.reader import make_reader

    with open(Path(path), "rb") as f:
        summary = make_reader(f).get_summary()
    out: dict[str, str] = {}
    if summary is None:
        return out
    for channel in summary.channels.values():
        kind = _channel_kind(summary.schemas.get(channel.schema_id), channel)
        if kind is not None:
            out[channel.topic] = kind
    return out


def iter_pointcloud_messages_mcap(path: str | Path, topics: Iterable[str] | None = None,
                                  time_source: str = "log") -> Iterator[tuple[str, Cloud]]:
    """Yield (topic, Cloud) for every point cloud message, in log-time order.

    `topics` restricts the read; by default every channel recognised as carrying clouds is read, which for
    a file from `write_pointclouds_mcap` is exactly its /points channel. Channels of any other kind are
    passed over. A PointCloud2 becomes a Cloud in its own header frame with non-finite points dropped.
    """
    from mcap.reader import make_reader

    if time_source not in ("log", "header"):
        raise ValueError(f"time_source must be 'log' or 'header', got {time_source!r}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    wanted = list(topics) if topics is not None else None
    with open(path, "rb") as f:
        reader = make_reader(f)
        if wanted is None:
            # Name the cloud topics up front when the file has a summary, so the reader can skip the chunks
            # that hold only camera or CAN data instead of handing every message here to be passed over.
            summary = reader.get_summary()
            if summary is not None and summary.channels:
                wanted = sorted({ch.topic for ch in summary.channels.values()
                                 if _channel_kind(summary.schemas.get(ch.schema_id), ch) is not None})
                if not wanted:
                    return
        for schema, channel, message in reader.iter_messages(topics=wanted):
            kind = _channel_kind(schema, channel)
            if kind is None:
                continue
            if kind == "npz":
                cloud = Cloud.from_npz_bytes(message.data)
                cloud.ts_ns = int(message.log_time)
            else:
                msg = (parse_pointcloud2_cdr if kind == "cdr" else parse_pointcloud2_ros1)(message.data)
                ts_ns = msg.stamp_ns if time_source == "header" else int(message.log_time)
                cloud = decode_pointcloud2(msg).to_cloud(ts_ns=ts_ns)
            yield channel.topic, cloud


def read_pointclouds_mcap(path: str | Path, topics: Iterable[str] | None = None,
                          time_source: str = "log") -> Iterator[Cloud]:
    """Yield each cloud in the file: our own /points channel, and ROS 1 / ROS 2 PointCloud2 topics alike.

    With several LiDARs recorded in one file this yields all of them interleaved; pass `topics` (see
    `pointcloud_topics_mcap`) to read one, or use `iter_pointcloud_messages_mcap` to keep the topic.
    """
    for _topic, cloud in iter_pointcloud_messages_mcap(path, topics, time_source):
        yield cloud
