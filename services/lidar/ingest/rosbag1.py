"""PointCloud2 topics out of a plain ROS 1 `.bag` (format 2.0), with no ROS install and no bag library.

Only as much of the format as reading messages needs (http://wiki.ros.org/Bags/Format/2.0):

    #ROSBAG V2.0\\n
    record*          record  := header_len:u32  header  data_len:u32  data        (all little-endian)
                     header  := (field_len:u32  name '=' value)*                  one of the fields is `op`

    op 0x03  bag header      skipped (it only locates the index)
    op 0x05  chunk           header: compression (none | bz2 | lz4), size; data: a run of records
    op 0x07  connection      header: conn, topic; data: a second header carrying type, md5sum, ...
    op 0x02  message data    header: conn, time (secs:u32, nsecs:u32); data: the serialized message
    op 0x04, 0x06            index data and chunk info, skipped

The file is read front to back and every chunk is decompressed, without using the index. That is slower on
a bag that is mostly camera data than an indexed read would be, and it is also what makes an unindexed bag
(one whose recorder was killed, `rosbag reindex` not yet run) read the same as a clean one.

bz2 is the standard library. lz4 chunks are an LZ4 frame, decompressed with `lz4.frame`, which is installed
here as a dependency of the mcap package; a bag with lz4 chunks on a machine without it raises a clear
error instead of skipping data.

VERIFICATION STATUS: this reader is tested against bags assembled byte by byte from the format
specification (tests/test_rosbag1_pointcloud2.py). It has not been run against a bag recorded by
`rosbag record`, because neither a ROS 1 bag nor a library that writes one was available when it was
written.
"""

from __future__ import annotations

import bz2
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.pointcloud2 import (
    ROS1_SCHEMA,
    decode_pointcloud2,
    parse_pointcloud2_ros1,
)

MAGIC = b"#ROSBAG V2.0\n"

_OP_MESSAGE = 0x02
_OP_CHUNK = 0x05
_OP_CONNECTION = 0x07


class RosbagError(ValueError):
    """The file is not a ROS 1 bag this reader can read. The message says why."""


def _parse_header(raw: bytes) -> dict[str, bytes]:
    """A record header (or a connection record's data, which has the same shape) as {name: value bytes}."""
    out: dict[str, bytes] = {}
    pos = 0
    while pos < len(raw):
        if pos + 4 > len(raw):
            raise RosbagError("truncated record header")
        (n,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        if pos + n > len(raw):
            raise RosbagError("record header field runs past the header")
        name, sep, value = raw[pos:pos + n].partition(b"=")
        if not sep:
            raise RosbagError("record header field has no '='")
        out[name.decode("ascii", errors="replace")] = value
        pos += n
    return out


def _records_from_bytes(buf: bytes) -> Iterator[tuple[dict[str, bytes], bytes]]:
    """The records inside a decompressed chunk."""
    pos, end = 0, len(buf)
    while pos < end:
        if pos + 4 > end:
            raise RosbagError("truncated record inside a chunk")
        (hlen,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        header = _parse_header(buf[pos:pos + hlen])
        pos += hlen
        if pos + 4 > end:
            raise RosbagError("truncated record inside a chunk")
        (dlen,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if pos + dlen > end:
            raise RosbagError("record data runs past its chunk")
        yield header, buf[pos:pos + dlen]
        pos += dlen


def _records_from_file(f: BinaryIO) -> Iterator[tuple[dict[str, bytes], bytes]]:
    """The top-level records of the file. A file that ends partway through a record (a recorder that was
    killed) simply ends: everything complete before that point has already been yielded."""
    while True:
        raw = f.read(4)
        if len(raw) < 4:
            return
        (hlen,) = struct.unpack("<I", raw)
        hraw = f.read(hlen)
        dl = f.read(4)
        if len(hraw) < hlen or len(dl) < 4:
            return
        (dlen,) = struct.unpack("<I", dl)
        header = _parse_header(hraw)
        op = header.get("op", b"\x00")[:1]
        if op and op[0] in (_OP_MESSAGE, _OP_CHUNK, _OP_CONNECTION):
            data = f.read(dlen)
            if len(data) < dlen:
                return
        else:
            f.seek(dlen, 1)                            # bag header padding, index data, chunk info
            data = b""
        yield header, data


def _decompress(compression: str, data: bytes, size: int | None) -> bytes:
    if compression == "none":
        out = data
    elif compression == "bz2":
        out = bz2.decompress(data)
    elif compression == "lz4":
        try:
            import lz4.frame
        except ImportError as exc:
            raise RosbagError("this bag has lz4-compressed chunks and the lz4 package is not installed") from exc
        out = lz4.frame.decompress(data)
    else:
        raise RosbagError(f"unknown chunk compression {compression!r}")
    if size is not None and len(out) != size:
        raise RosbagError(f"chunk decompressed to {len(out)} bytes, its header says {size}")
    return out


def _u32(header: dict[str, bytes], name: str) -> int:
    raw = header.get(name)
    if raw is None or len(raw) != 4:
        raise RosbagError(f"record is missing its {name!r} field")
    return struct.unpack("<I", raw)[0]


def iter_pointcloud_messages_rosbag(path: str | Path, topics: Iterable[str] | None = None,
                                    time_source: str = "log") -> Iterator[tuple[str, Cloud]]:
    """Yield (topic, Cloud) for every `sensor_msgs/PointCloud2` message of a ROS 1 bag, in file order.

    `topics` restricts the read to those topics. ts_ns is the time the message was recorded at
    (`time_source="log"`, matching the MCAP reader) or its header stamp (`"header"`). Connections of any
    other message type are passed over.
    """
    if time_source not in ("log", "header"):
        raise ValueError(f"time_source must be 'log' or 'header', got {time_source!r}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    wanted = set(topics) if topics is not None else None
    conn_topic: dict[int, str] = {}            # connection id -> topic, PointCloud2 connections only
    seen_conn: set[int] = set()

    def handle(header: dict[str, bytes], data: bytes) -> tuple[str, Cloud] | None:
        op = header.get("op", b"\x00")[:1]
        code = op[0] if op else 0
        if code == _OP_CONNECTION:
            conn = _u32(header, "conn")
            if conn not in seen_conn:
                seen_conn.add(conn)
                info = _parse_header(data)
                topic = header.get("topic", info.get("topic", b"")).decode("utf-8", errors="replace")
                if (info.get("type", b"").decode("ascii", errors="replace") == ROS1_SCHEMA
                        and (wanted is None or topic in wanted)):
                    conn_topic[conn] = topic
            return None
        if code == _OP_MESSAGE:
            cloud_topic = conn_topic.get(_u32(header, "conn"))
            if cloud_topic is None:
                return None
            msg = parse_pointcloud2_ros1(data)
            if time_source == "header":
                ts_ns = msg.stamp_ns
            else:
                t = header.get("time")
                if t is None or len(t) != 8:
                    raise RosbagError("message record is missing its 'time' field")
                secs, nsecs = struct.unpack("<II", t)
                ts_ns = secs * 1_000_000_000 + nsecs
            return cloud_topic, decode_pointcloud2(msg).to_cloud(ts_ns=ts_ns)
        return None

    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise RosbagError(f"{path.name} is not a ROS 1 bag in format 2.0 (it starts with {magic[:13]!r})")
        for header, data in _records_from_file(f):
            op = header.get("op", b"\x00")[:1]
            if op and op[0] == _OP_CHUNK:
                compression = header.get("compression", b"none").decode("ascii", errors="replace")
                size = _u32(header, "size") if "size" in header else None
                for inner_header, inner_data in _records_from_bytes(_decompress(compression, data, size)):
                    item = handle(inner_header, inner_data)
                    if item is not None:
                        yield item
            else:
                item = handle(header, data)
                if item is not None:
                    yield item


def read_pointclouds_rosbag(path: str | Path, topics: Iterable[str] | None = None,
                            time_source: str = "log") -> Iterator[Cloud]:
    """Yield each PointCloud2 of a ROS 1 bag as a Cloud. See `iter_pointcloud_messages_rosbag`."""
    for _topic, cloud in iter_pointcloud_messages_rosbag(path, topics, time_source):
        yield cloud
