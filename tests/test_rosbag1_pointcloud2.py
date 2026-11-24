"""PointCloud2 topics read out of a ROS 1 .bag (format 2.0).

Pure. The bags are assembled here, record by record, from the published format description
(http://wiki.ros.org/Bags/Format/2.0), with every record type a recorded bag contains: the padded bag
header, chunks (uncompressed, bz2, lz4), connection and message records inside them, the index data
records after each chunk, and the connection + chunk info records of the index section at the end.

What this does and does not establish: the reader follows the specification as this file states it. No bag
written by `rosbag record` was available, so agreement with real recordings is not tested here.
"""

from __future__ import annotations

import bz2
import struct

import numpy as np
import pytest

from services.lidar.ingest import iter_pointcloud_messages_rosbag, read_pointclouds_rosbag
from services.lidar.ingest.rosbag1 import RosbagError

FLOAT32, UINT16 = 7, 4
XYZI = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1), ("intensity", 12, FLOAT32, 1)]
XYZR = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1), ("ring", 12, UINT16, 1)]


def _pointcloud2(fields, point_step, data, width, frame_id, sec, nsec) -> bytes:
    """ROS 1 serialization of sensor_msgs/PointCloud2: little-endian, packed, strings as length + bytes."""
    b = struct.pack("<III", 7, sec, nsec)
    b += struct.pack("<I", len(frame_id)) + frame_id.encode()
    b += struct.pack("<II", 1, width)
    b += struct.pack("<I", len(fields))
    for name, offset, datatype, count in fields:
        b += struct.pack("<I", len(name)) + name.encode() + struct.pack("<IBI", offset, datatype, count)
    b += struct.pack("<BII", 0, point_step, width * point_step)
    b += struct.pack("<I", len(data)) + data
    b += struct.pack("<B", 1)
    return b


# --- bag records, per the format description --------------------------------------------------------------

def _header(fields: list[tuple[str, bytes]]) -> bytes:
    out = b""
    for name, value in fields:
        raw = name.encode() + b"=" + value
        out += struct.pack("<I", len(raw)) + raw
    return out


def _record(fields: list[tuple[str, bytes]], data: bytes) -> bytes:
    h = _header(fields)
    return struct.pack("<I", len(h)) + h + struct.pack("<I", len(data)) + data


def _connection(conn: int, topic: str, msg_type: str) -> bytes:
    data = _header([("topic", topic.encode()), ("type", msg_type.encode()),
                    ("md5sum", b"1158d486dd51d683ce2f1be655c3c181"),
                    ("message_definition", b"# not needed to read the message\n"),
                    ("callerid", b"/driver"), ("latching", b"0")])
    return _record([("conn", struct.pack("<I", conn)), ("topic", topic.encode()), ("op", b"\x07")], data)


def _message(conn: int, secs: int, nsecs: int, payload: bytes) -> bytes:
    return _record([("conn", struct.pack("<I", conn)), ("time", struct.pack("<II", secs, nsecs)),
                    ("op", b"\x02")], payload)


def _chunk(records: bytes, compression: str) -> bytes:
    if compression == "none":
        data = records
    elif compression == "bz2":
        data = bz2.compress(records)
    else:
        import lz4.frame

        # roslz4 writes an LZ4 frame with independent blocks and a content checksum.
        data = lz4.frame.compress(records, block_linked=False, content_checksum=True)
    return _record([("compression", compression.encode()), ("size", struct.pack("<I", len(records))),
                    ("op", b"\x05")], data)


def _index_data(conn: int, entries: list[tuple[int, int, int]]) -> bytes:
    data = b"".join(struct.pack("<III", s, ns, off) for s, ns, off in entries)
    return _record([("ver", struct.pack("<I", 1)), ("conn", struct.pack("<I", conn)),
                    ("count", struct.pack("<I", len(entries))), ("op", b"\x04")], data)


def _chunk_info(chunk_pos: int, conns: dict[int, int]) -> bytes:
    data = b"".join(struct.pack("<II", c, n) for c, n in conns.items())
    return _record([("ver", struct.pack("<I", 1)), ("chunk_pos", struct.pack("<Q", chunk_pos)),
                    ("start_time", struct.pack("<II", 100, 0)), ("end_time", struct.pack("<II", 103, 0)),
                    ("count", struct.pack("<I", len(conns))), ("op", b"\x06")], data)


def _bag_header(index_pos: int, conn_count: int, chunk_count: int) -> bytes:
    fields = [("index_pos", struct.pack("<Q", index_pos)), ("conn_count", struct.pack("<I", conn_count)),
              ("chunk_count", struct.pack("<I", chunk_count)), ("op", b"\x03")]
    pad = 4096 - 4 - len(_header(fields)) - 4       # the whole record is padded to 4096 bytes with spaces
    rec = _record(fields, b" " * pad)
    assert len(rec) == 4096
    return rec


def _blob_xyzi(points) -> bytes:
    return b"".join(struct.pack("<ffff", *p) for p in points)


CLOUD_A = [(1.5, -2.25, 0.5, 0.25), (10.0, 20.0, -1.75, 0.75)]
CLOUD_B = [(3.0, 4.0, 5.0, 0.5)]
CLOUD_C = [(7.0, 8.0, 9.0, 1.0), (float("nan"), 0.0, 0.0, 0.0), (-7.0, -8.0, -9.0, 0.0)]
RING_BLOB = b"".join(struct.pack("<fffH", float(i), 1.0, 2.0, 10 + i) + b"\x55\x55" for i in range(3))


def _build_bag(lz4_chunk: bool = True) -> tuple[bytes, list[int]]:
    """A three-chunk bag with two PointCloud2 connections and one camera connection. Returns the bytes and
    the file offset of each chunk record."""
    msg_a = _pointcloud2(XYZI, 16, _blob_xyzi(CLOUD_A), 2, "velodyne", 99, 500)
    msg_b = _pointcloud2(XYZI, 16, _blob_xyzi(CLOUD_B), 1, "velodyne", 100, 900_000_000)
    msg_c = _pointcloud2(XYZI, 16, _blob_xyzi(CLOUD_C), 3, "velodyne", 102, 0)
    msg_ring = _pointcloud2(XYZR, 16, RING_BLOB, 3, "os_sensor", 101, 250)
    image = b"\x89not a point cloud" * 20

    chunks = [
        ("none", _connection(0, "/velodyne_points", "sensor_msgs/PointCloud2")
         + _connection(1, "/camera/image_raw", "sensor_msgs/Image")
         + _message(0, 100, 1, msg_a) + _message(1, 100, 2, image)),
        ("bz2", _connection(2, "/ouster/points", "sensor_msgs/PointCloud2")
         + _message(0, 101, 0, msg_b) + _message(1, 101, 1, image) + _message(2, 101, 500, msg_ring)),
        ("lz4" if lz4_chunk else "none", _message(0, 102, 7, msg_c)),
    ]
    body = b""
    positions = []
    base = len(b"#ROSBAG V2.0\n") + 4096
    for compression, records in chunks:
        positions.append(base + len(body))
        body += _chunk(records, compression)
        body += _index_data(0, [(100, 0, 0)])           # index records follow every chunk
    index_pos = base + len(body)
    index = (_connection(0, "/velodyne_points", "sensor_msgs/PointCloud2")
             + _connection(1, "/camera/image_raw", "sensor_msgs/Image")
             + _connection(2, "/ouster/points", "sensor_msgs/PointCloud2")
             + b"".join(_chunk_info(p, {0: 1}) for p in positions))
    return b"#ROSBAG V2.0\n" + _bag_header(index_pos, 3, 3) + body + index, positions


@pytest.fixture
def bag(tmp_path):
    pytest.importorskip("lz4.frame")
    data, _ = _build_bag()
    path = tmp_path / "drive.bag"
    path.write_bytes(data)
    return path


def test_pointcloud2_messages_are_read_from_every_kind_of_chunk(bag):
    got = list(iter_pointcloud_messages_rosbag(bag))
    assert [(topic, c.ts_ns) for topic, c in got] == [
        ("/velodyne_points", 100 * 10**9 + 1),          # uncompressed chunk
        ("/velodyne_points", 101 * 10**9),              # bz2 chunk
        ("/ouster/points", 101 * 10**9 + 500),
        ("/velodyne_points", 102 * 10**9 + 7),          # lz4 chunk
    ]
    a, b, ring, c = (cloud for _t, cloud in got)
    assert a.xyz.tolist() == [list(p[:3]) for p in CLOUD_A] and a.intensity.tolist() == [0.25, 0.75]
    assert a.frame == "velodyne" and a.source == "lidar"
    assert b.xyz.tolist() == [[3.0, 4.0, 5.0]]
    assert ring.xyz[:, 0].tolist() == [0.0, 1.0, 2.0] and ring.ring.tolist() == [10, 11, 12]
    assert ring.frame == "os_sensor"
    assert c.xyz.tolist() == [[7.0, 8.0, 9.0], [-7.0, -8.0, -9.0]]         # the NaN return is dropped
    np.testing.assert_allclose(c.intensity, [1.0, 0.0])
    # The camera connection contributed nothing, and the connection records repeated in the index section
    # at the end of the file did not replay anything.
    assert len(got) == 4


def test_topics_and_time_source_narrow_the_read(bag):
    only = list(iter_pointcloud_messages_rosbag(bag, topics=["/ouster/points"]))
    assert [t for t, _c in only] == ["/ouster/points"] and only[0][1].ring.tolist() == [10, 11, 12]
    assert list(read_pointclouds_rosbag(bag, topics=["/camera/image_raw"])) == []   # not a cloud topic
    stamps = [c.ts_ns for c in read_pointclouds_rosbag(bag, topics=["/velodyne_points"], time_source="header")]
    assert stamps == [99 * 10**9 + 500, 100 * 10**9 + 900_000_000, 102 * 10**9]
    with pytest.raises(ValueError):
        list(read_pointclouds_rosbag(bag, time_source="wall"))


def test_a_bag_cut_short_gives_what_was_complete(tmp_path):
    data, positions = _build_bag(lz4_chunk=False)
    path = tmp_path / "killed.bag"
    path.write_bytes(data[:positions[2] + 40])             # the recorder died inside the third chunk
    got = list(iter_pointcloud_messages_rosbag(path))
    assert [(t, c.ts_ns) for t, c in got] == [
        ("/velodyne_points", 100 * 10**9 + 1), ("/velodyne_points", 101 * 10**9),
        ("/ouster/points", 101 * 10**9 + 500)]


def test_files_that_are_not_readable_bags_say_why(tmp_path):
    not_bag = tmp_path / "x.bag"
    not_bag.write_bytes(b"#ROSBAG V1.2\n" + b"\x00" * 64)
    with pytest.raises(RosbagError, match="not a ROS 1 bag in format 2.0"):
        list(read_pointclouds_rosbag(not_bag))
    with pytest.raises(FileNotFoundError):
        list(read_pointclouds_rosbag(tmp_path / "absent.bag"))

    msg = _pointcloud2(XYZI, 16, _blob_xyzi(CLOUD_B), 1, "velodyne", 1, 2)
    records = _connection(0, "/p", "sensor_msgs/PointCloud2") + _message(0, 1, 2, msg)
    head = b"#ROSBAG V2.0\n" + _bag_header(0, 1, 1)

    odd = tmp_path / "zstd.bag"
    odd.write_bytes(head + _record([("compression", b"zstd"), ("size", struct.pack("<I", len(records))),
                                    ("op", b"\x05")], records))
    with pytest.raises(RosbagError, match="unknown chunk compression 'zstd'"):
        list(read_pointclouds_rosbag(odd))

    lying = tmp_path / "size.bag"
    lying.write_bytes(head + _record([("compression", b"none"), ("size", struct.pack("<I", len(records) + 9)),
                                      ("op", b"\x05")], records))
    with pytest.raises(RosbagError, match="its header says"):
        list(read_pointclouds_rosbag(lying))

    # An unindexed bag (index_pos 0, nothing after the chunk) reads like any other.
    plain = tmp_path / "unindexed.bag"
    plain.write_bytes(head + _chunk(records, "bz2"))
    clouds = list(read_pointclouds_rosbag(plain))
    assert len(clouds) == 1 and clouds[0].xyz.tolist() == [[3.0, 4.0, 5.0]] and clouds[0].ts_ns == 10**9 + 2
