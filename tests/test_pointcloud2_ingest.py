"""sensor_msgs/PointCloud2 decodes into the internal Cloud, from ROS 2 (CDR) and ROS 1 messages and from
MCAP files holding either, without disturbing the private labelox-npz channel.

Pure: point blobs are packed with `struct` here (not with the numpy record dtype the decoder uses), CDR
and ROS 1 messages are laid out by hand, and the MCAP files are real ones written to tmp_path. One group of
tests also has mcap-ros2-support serialize the message from its .msg definition, as an encoder that shares
no code with the reader under test.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from mcap.writer import Writer

from services.lidar.ingest import (
    Cloud,
    iter_pointcloud_messages_mcap,
    pointcloud_topics_mcap,
    read_pointclouds_mcap,
    write_pointclouds_mcap,
)
from services.lidar.ingest.pointcloud2 import (
    PointCloud2,
    PointCloud2Error,
    PointField,
    decode_pointcloud2,
    parse_pointcloud2_cdr,
    parse_pointcloud2_ros1,
)

INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 = range(1, 9)


def _msg(fields, point_step, data, width=None, height=1, row_step=None, bigendian=False, dense=True,
         frame_id="lidar_top", stamp_ns=0) -> PointCloud2:
    width = len(data) // point_step if width is None else width
    return PointCloud2(frame_id=frame_id, stamp_ns=stamp_ns, height=height, width=width,
                       fields=[PointField(*f) for f in fields], is_bigendian=bigendian,
                       point_step=point_step, row_step=width * point_step if row_step is None else row_step,
                       data=data, is_dense=dense)


# --- the point blob ---------------------------------------------------------------------------------------

def test_fields_are_read_at_their_offsets_past_padding_and_unknown_fields():
    # x y z | 4 junk bytes | intensity | ring (u16) | 2 junk bytes | time | rgb : 32 bytes per point.
    # The junk is non-zero, so reading a field four bytes early or late cannot pass.
    rows = [(1.5, -2.25, 0.5, 0.25, 3, 0.001, 0x00FF8800),
            (10.0, 20.0, -1.75, 0.75, 17, 0.002, 0x00112233),
            (-33.5, 0.125, 2.0, 1.0, 31, 0.003, 0x00ABCDEF)]
    data = b"".join(struct.pack("<fffIfHHfI", x, y, z, 0xDEADBEEF, i, ring, 0xBEEF, t, rgb)
                    for x, y, z, i, ring, t, rgb in rows)
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1), ("intensity", 16, FLOAT32, 1),
              ("ring", 20, UINT16, 1), ("time", 24, FLOAT32, 1), ("rgb", 28, UINT32, 1)]
    pts = decode_pointcloud2(_msg(fields, 32, data))

    assert pts.xyz.tolist() == [[1.5, -2.25, 0.5], [10.0, 20.0, -1.75], [-33.5, 0.125, 2.0]]
    assert pts.intensity.tolist() == [0.25, 0.75, 1.0]            # a float intensity is passed through
    assert pts.ring.tolist() == [3, 17, 31]
    np.testing.assert_allclose(pts.time, [0.001, 0.002, 0.003], rtol=1e-6)
    assert pts.time_field == "time" and (pts.n_total, pts.n_dropped) == (3, 0)

    # The same points with the fields declared in another order: the declaration order is irrelevant.
    shuffled = decode_pointcloud2(_msg(list(reversed(fields)), 32, data))
    assert shuffled.xyz.tolist() == pts.xyz.tolist() and shuffled.ring.tolist() == [3, 17, 31]

    cloud = pts.to_cloud(ts_ns=77)
    assert isinstance(cloud, Cloud) and cloud.n == 3 and cloud.ts_ns == 77 and cloud.frame == "lidar_top"
    assert cloud.xyz.dtype == np.float32 and cloud.ring.dtype == np.int16
    assert cloud.xyz.tolist() == pts.xyz.tolist()


def test_big_endian_data_is_honoured():
    rows = [(1.5, -2.25, 0.5, 65535, 4), (100.0, 200.0, 300.0, 13107, 9)]
    data = b"".join(struct.pack(">fffHH", *r) for r in rows)
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1),
              ("intensity", 12, UINT16, 1), ("ring", 14, UINT16, 1)]
    pts = decode_pointcloud2(_msg(fields, 16, data, bigendian=True))
    assert pts.xyz.tolist() == [[1.5, -2.25, 0.5], [100.0, 200.0, 300.0]]
    np.testing.assert_allclose(pts.intensity, [1.0, 0.2], rtol=1e-6)      # uint16 is scaled by its range
    assert pts.ring.tolist() == [4, 9]
    # Read as little-endian the same bytes are different numbers: the flag is not decoration.
    wrong = decode_pointcloud2(_msg(fields, 16, data, bigendian=False))
    assert wrong.xyz.shape[0] < 2 or wrong.xyz.tolist() != pts.xyz.tolist()


def test_non_finite_points_are_dropped_and_the_other_fields_stay_aligned():
    nan, inf = float("nan"), float("inf")
    rows = [(1.0, 1.0, 1.0, 10, 0), (nan, 2.0, 2.0, 20, 1), (3.0, 3.0, 3.0, 30, 2), (4.0, -inf, 4.0, 40, 3),
            (5.0, 5.0, nan, 50, 4), (6.0, 6.0, 6.0, 60, 5)]
    data = b"".join(struct.pack("<fffBB", *r) for r in rows)
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1),
              ("intensity", 12, UINT8, 1), ("ring", 13, UINT8, 1)]
    for dense in (False, True):       # a driver that claims is_dense while sending NaNs is tolerated too
        pts = decode_pointcloud2(_msg(fields, 14, data, dense=dense))
        assert pts.xyz.tolist() == [[1.0] * 3, [3.0] * 3, [6.0] * 3]
        np.testing.assert_allclose(pts.intensity, [10 / 255, 30 / 255, 60 / 255], rtol=1e-6)
        assert pts.ring.tolist() == [0, 2, 5]
        assert (pts.n_total, pts.n_dropped) == (6, 3)
    assert np.isfinite(pts.to_cloud().xyz).all()


def test_float64_coordinates_at_unaligned_offsets():
    # A leading uint8 puts every double at an odd offset; the point is 26 bytes.
    rows = [(7, 1234567.125, -0.5, 2.0e-3, 200), (9, 2.5, 3.5, 4.5, 51)]
    data = b"".join(struct.pack("<BdddB", *r) for r in rows)
    fields = [("tag", 0, UINT8, 1), ("x", 1, FLOAT64, 1), ("y", 9, FLOAT64, 1), ("z", 17, FLOAT64, 1),
              ("intensity", 25, UINT8, 1)]
    pts = decode_pointcloud2(_msg(fields, 26, data))
    assert pts.xyz.dtype == np.float64
    assert pts.xyz.tolist() == [[1234567.125, -0.5, 2.0e-3], [2.5, 3.5, 4.5]]
    np.testing.assert_allclose(pts.intensity, [200 / 255, 51 / 255], rtol=1e-6)
    assert pts.ring is None and pts.time is None


def test_integer_coordinates_and_field_name_variants():
    # Millimetre int32 coordinates (some solid-state units), intensity called reflectivity, ring called
    # channel, time called t as a uint32.
    rows = [(1000, -2000, 300, 128, 6, 123456), (5, 6, 7, 255, 7, 123999)]
    data = b"".join(struct.pack("<iiiBBI", *r) for r in rows)
    fields = [("x", 0, INT32, 1), ("y", 4, INT32, 1), ("z", 8, INT32, 1), ("reflectivity", 12, UINT8, 1),
              ("channel", 13, UINT8, 1), ("t", 14, UINT32, 1)]
    pts = decode_pointcloud2(_msg(fields, 18, data))
    assert pts.xyz.tolist() == [[1000.0, -2000.0, 300.0], [5.0, 6.0, 7.0]]
    np.testing.assert_allclose(pts.intensity, [128 / 255, 1.0], rtol=1e-6)
    assert pts.ring.tolist() == [6, 7] and pts.time.tolist() == [123456.0, 123999.0] and pts.time_field == "t"


def _organised(height, width, pad, last_row_padded=True):
    """An organised xyz cloud whose point (r, c) is (r, c, 10r + c), with `pad` junk bytes after each row."""
    rows = []
    for r in range(height):
        row = b"".join(struct.pack("<fff", float(r), float(c), float(10 * r + c)) for c in range(width))
        rows.append(row + (b"\xa5" * pad if (last_row_padded or r < height - 1) else b""))
    return b"".join(rows)


def test_an_organised_cloud_with_padded_rows_reads_row_major():
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1)]
    expected = [[float(r), float(c), float(10 * r + c)] for r in range(3) for c in range(4)]
    for padded_last in (True, False):
        data = _organised(3, 4, pad=5, last_row_padded=padded_last)
        pts = decode_pointcloud2(_msg(fields, 12, data, width=4, height=3, row_step=4 * 12 + 5))
        assert pts.xyz.tolist() == expected and pts.n_total == 12
    # No padding: row_step is exactly width * point_step.
    pts = decode_pointcloud2(_msg(fields, 12, _organised(3, 4, pad=0), width=4, height=3, row_step=48))
    assert pts.xyz.tolist() == expected

    # An organised cloud keeps its invalid returns as NaN so the grid stays rectangular; they are dropped
    # on the way to a Cloud and the survivors keep their order.
    blob = bytearray(_organised(2, 3, pad=4))
    struct.pack_into("<fff", blob, 1 * 12, float("nan"), float("nan"), float("nan"))        # point (0, 1)
    struct.pack_into("<fff", blob, (3 * 12 + 4) + 2 * 12, float("nan"), 0.0, 0.0)           # point (1, 2)
    pts = decode_pointcloud2(_msg(fields, 12, bytes(blob), width=3, height=2, row_step=3 * 12 + 4, dense=False))
    assert pts.xyz.tolist() == [[0.0, 0.0, 0.0], [0.0, 2.0, 2.0], [1.0, 0.0, 10.0], [1.0, 1.0, 11.0]]


def test_a_wrong_row_step_does_not_lose_an_unorganised_cloud():
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1)]
    data = b"".join(struct.pack("<fff", float(i), 0.0, 1.0) for i in range(5))
    for row_step in (0, 12, 60, 4096):
        pts = decode_pointcloud2(_msg(fields, 12, data, width=5, height=1, row_step=row_step))
        assert pts.xyz[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    empty = decode_pointcloud2(_msg(fields, 12, b"", width=0, height=1))
    assert empty.xyz.shape == (0, 3) and empty.to_cloud().n == 0


def test_messages_that_cannot_be_decoded_say_why():
    xyz = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1)]
    data = struct.pack("<fff", 1.0, 2.0, 3.0) * 4
    with pytest.raises(PointCloud2Error, match="no z field"):
        decode_pointcloud2(_msg(xyz[:2], 12, data))
    with pytest.raises(PointCloud2Error, match="does not fit"):
        decode_pointcloud2(_msg(xyz[:2] + [("z", 10, FLOAT32, 1)], 12, data))
    with pytest.raises(PointCloud2Error, match="unknown datatype"):
        decode_pointcloud2(_msg(xyz[:2] + [("z", 8, 9, 1)], 12, data))
    with pytest.raises(PointCloud2Error, match="need 60"):
        decode_pointcloud2(_msg(xyz, 12, data, width=5))                     # one point short
    with pytest.raises(PointCloud2Error, match="rows of row_step"):
        decode_pointcloud2(_msg(xyz, 12, data + data[:12], width=2, height=2, row_step=40))   # needs 64
    with pytest.raises(PointCloud2Error, match="point_step"):
        decode_pointcloud2(_msg(xyz, 0, data, width=4))


# --- ROS 2: CDR -------------------------------------------------------------------------------------------

POINTS = [(1.5, -2.25, 0.5, 0.25), (10.0, 20.0, -1.75, 0.75), (-33.5, 0.125, 2.0, 1.0)]
BLOB = b"".join(struct.pack("<ffff", *p) for p in POINTS)
SEC, NSEC = 1_700_000_123, 456_789_012


def _explicit_cdr_message() -> bytes:
    """A little-endian CDR PointCloud2, byte by byte. Offsets are from the end of the 4-byte encapsulation
    header, which is what CDR alignment is measured from; every uint32 must start on a multiple of 4."""
    b = b""
    b += struct.pack("<iI", SEC, NSEC)                         # 0   stamp.sec, stamp.nanosec
    b += struct.pack("<I", 10) + b"lidar_top\x00"              # 8   frame_id: length counts the NUL -> 22
    b += b"\x00" * 2                                           # 22  pad to 24
    b += struct.pack("<II", 1, 3)                              # 24  height, width
    b += struct.pack("<I", 4)                                  # 32  fields: sequence length
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):        # 36, 56, 76: each 20 bytes
        b += struct.pack("<I", 2) + name.encode() + b"\x00"    #     name: 4 + 2 bytes
        b += b"\x00" * 2                                       #     pad to the uint32 offset
        b += struct.pack("<I", offset)
        b += struct.pack("<B", 7) + b"\x00" * 3                #     datatype, pad to the uint32 count
        b += struct.pack("<I", 1)
    assert len(b) == 96
    b += struct.pack("<I", 10) + b"intensity\x00"              # 96  name: 4 + 10 bytes -> 110
    b += b"\x00" * 2                                           # 110 pad to 112
    b += struct.pack("<I", 12)                                 # 112 offset
    b += struct.pack("<B", 7) + b"\x00" * 3                    # 116 datatype, pad to 120
    b += struct.pack("<I", 1)                                  # 120 count
    b += struct.pack("<B", 0) + b"\x00" * 3                    # 124 is_bigendian, pad to 128
    b += struct.pack("<II", 16, 48)                            # 128 point_step, row_step
    b += struct.pack("<I", len(BLOB)) + BLOB                   # 136 data: length, then the bytes
    b += struct.pack("<B", 1)                                  #     is_dense
    return b"\x00\x01\x00\x00" + b                             # CDR_LE encapsulation header first


class _CdrWriter:
    """A general CDR writer for the variants below, checked against the explicit message above. Padding is
    filled with 0xEE rather than zeros so that a reader which fails to skip it reads garbage."""

    def __init__(self, big: bool = False, fill: int = 0xEE) -> None:
        self.order, self.big, self.fill, self.b = (">" if big else "<"), big, fill, bytearray()

    def _align(self, n: int) -> None:
        self.b += bytes([self.fill]) * ((-len(self.b)) % n)

    def u32(self, v: int) -> None:
        self._align(4)
        self.b += struct.pack(self.order + "I", v)

    def i32(self, v: int) -> None:
        self._align(4)
        self.b += struct.pack(self.order + "i", v)

    def u8(self, v: int) -> None:
        self.b.append(v)

    def string(self, s: str) -> None:
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.b += raw

    def octets(self, data: bytes) -> None:
        self.u32(len(data))
        self.b += data

    def done(self) -> bytes:
        return (b"\x00\x00" if self.big else b"\x00\x01") + b"\x00\x00" + bytes(self.b)


def _cdr_message(fields, point_step, data, width, height=1, frame_id="lidar_top", big=False, fill=0xEE,
                 data_bigendian=False, dense=True, sec=SEC, nsec=NSEC, row_step=None) -> bytes:
    w = _CdrWriter(big, fill)
    w.i32(sec)
    w.u32(nsec)
    w.string(frame_id)
    w.u32(height)
    w.u32(width)
    w.u32(len(fields))
    for name, offset, datatype, count in fields:
        w.string(name)
        w.u32(offset)
        w.u8(datatype)
        w.u32(count)
    w.u8(1 if data_bigendian else 0)
    w.u32(point_step)
    w.u32(width * point_step if row_step is None else row_step)
    w.octets(data)
    w.u8(1 if dense else 0)
    return w.done()


XYZI = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1), ("intensity", 12, FLOAT32, 1)]


def test_a_hand_laid_out_cdr_message_parses():
    raw = _explicit_cdr_message()
    assert raw == _cdr_message(XYZI, 16, BLOB, width=3, fill=0x00)     # the helper writes the same bytes

    msg = parse_pointcloud2_cdr(raw)
    assert msg.frame_id == "lidar_top" and msg.stamp_ns == SEC * 1_000_000_000 + NSEC
    assert (msg.height, msg.width, msg.point_step, msg.row_step) == (1, 3, 16, 48)
    assert [(f.name, f.offset, f.datatype, f.count) for f in msg.fields] == XYZI
    assert msg.is_bigendian is False and msg.is_dense is True and bytes(msg.data) == BLOB

    pts = decode_pointcloud2(msg)
    assert pts.xyz.tolist() == [list(p[:3]) for p in POINTS]
    assert pts.intensity.tolist() == [p[3] for p in POINTS]


def test_cdr_alignment_holds_for_every_string_length_and_garbage_padding():
    # frame_id lengths 0..8 walk the frame_id padding through 3, 2, 1, 0; the field names (2, 3, 5, 10
    # bytes with their NUL) do the same inside the sequence. Padding bytes are 0xEE.
    fields = [("x", 0, FLOAT32, 1), ("yy", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1), ("ring", 12, UINT16, 1),
              ("intensity", 14, UINT16, 1)]
    data = b"".join(struct.pack("<fffHH", float(i), float(-i), 0.5 * i, i, 1000 * i) for i in range(4))
    for n in range(9):
        frame_id = "velodyne"[:n]
        msg = parse_pointcloud2_cdr(_cdr_message(fields, 16, data, width=4, frame_id=frame_id, dense=False))
        assert msg.frame_id == frame_id and msg.is_dense is False
        assert [(f.name, f.offset, f.datatype, f.count) for f in msg.fields] == fields
        assert (msg.point_step, msg.row_step, bytes(msg.data)) == (16, 64, data)
    # "yy" is not y: the cloud has no y field, and says so rather than guessing.
    with pytest.raises(PointCloud2Error, match="no y field"):
        decode_pointcloud2(msg)


def test_big_endian_cdr_and_big_endian_points_are_independent():
    # The encapsulation's byte order governs the message's own integers; is_bigendian governs the blob.
    le_blob = BLOB
    be_blob = b"".join(struct.pack(">ffff", *p) for p in POINTS)
    for cdr_big in (False, True):
        for data_big, blob in ((False, le_blob), (True, be_blob)):
            raw = _cdr_message(XYZI, 16, blob, width=3, big=cdr_big, data_bigendian=data_big)
            assert raw[:2] == (b"\x00\x00" if cdr_big else b"\x00\x01")
            msg = parse_pointcloud2_cdr(raw)
            assert msg.stamp_ns == SEC * 1_000_000_000 + NSEC and msg.is_bigendian is data_big
            assert decode_pointcloud2(msg).xyz.tolist() == [list(p[:3]) for p in POINTS]


def test_cdr_messages_that_are_not_what_they_claim_are_refused():
    raw = _explicit_cdr_message()
    with pytest.raises(PointCloud2Error, match="ends at byte"):
        parse_pointcloud2_cdr(raw[:-20])                                   # cut inside the data
    with pytest.raises(PointCloud2Error, match="encapsulation"):
        parse_pointcloud2_cdr(b"\x00\x07\x00\x00" + raw[4:])               # XCDR2, not what ROS 2 writes
    with pytest.raises(PointCloud2Error):
        parse_pointcloud2_cdr(b"\x00\x01")


# --- ROS 1 ------------------------------------------------------------------------------------------------

def _ros1_message(fields, point_step, data, width, height=1, frame_id="velodyne", seq=42, sec=SEC, nsec=NSEC,
                  data_bigendian=False, dense=True) -> bytes:
    """ROS 1 serialization: little-endian, packed, strings as length + bytes with no terminator, and a
    Header that begins with a sequence number."""
    b = struct.pack("<III", seq, sec, nsec)
    b += struct.pack("<I", len(frame_id)) + frame_id.encode()
    b += struct.pack("<II", height, width)
    b += struct.pack("<I", len(fields))
    for name, offset, datatype, count in fields:
        b += struct.pack("<I", len(name)) + name.encode() + struct.pack("<IBI", offset, datatype, count)
    b += struct.pack("<BII", 1 if data_bigendian else 0, point_step, width * point_step)
    b += struct.pack("<I", len(data)) + data
    b += struct.pack("<B", 1 if dense else 0)
    return b


def test_a_ros1_message_parses():
    msg = parse_pointcloud2_ros1(_ros1_message(XYZI, 16, BLOB, width=3))
    assert msg.frame_id == "velodyne" and msg.stamp_ns == SEC * 1_000_000_000 + NSEC
    assert (msg.height, msg.width, msg.point_step, msg.row_step) == (1, 3, 16, 48)
    assert [(f.name, f.offset, f.datatype, f.count) for f in msg.fields] == XYZI
    assert decode_pointcloud2(msg).xyz.tolist() == [list(p[:3]) for p in POINTS]
    with pytest.raises(PointCloud2Error):
        parse_pointcloud2_ros1(_ros1_message(XYZI, 16, BLOB, width=3)[:50])


# --- MCAP -------------------------------------------------------------------------------------------------

T0 = 1_700_000_200_000_000_000


def _write_mcap(path, channels):
    """channels: [(topic, schema_name, schema_encoding, message_encoding, [(log_time, bytes), ...])]."""
    with open(path, "wb") as f:
        w = Writer(f)
        w.start()
        pending = []
        for topic, schema_name, schema_encoding, message_encoding, messages in channels:
            sid = w.register_schema(name=schema_name, encoding=schema_encoding, data=b"")
            cid = w.register_channel(topic=topic, message_encoding=message_encoding, schema_id=sid)
            pending += [(t, cid, data) for t, data in messages]
        for t, cid, data in sorted(pending, key=lambda m: m[0]):
            w.add_message(channel_id=cid, log_time=t, publish_time=t, data=data)
        w.finish()
    return path


def test_the_private_encoding_reads_exactly_as_before(tmp_path):
    rng = np.random.default_rng(4)
    clouds = [Cloud(xyz=rng.uniform(-50, 50, (40, 3)), intensity=rng.uniform(0, 1, 40), ts_ns=T0 + i * 100,
                    ring=rng.integers(0, 32, 40), source="lidar", frame="ego") for i in (2, 0, 1)]
    info = write_pointclouds_mcap(clouds, tmp_path / "own.mcap")
    assert info == {"path": str(tmp_path / "own.mcap"), "clouds": 3, "topic": "/points"}

    got = list(read_pointclouds_mcap(tmp_path / "own.mcap"))
    assert [c.ts_ns for c in got] == [T0, T0 + 100, T0 + 200]
    by_ts = {c.ts_ns: c for c in clouds}
    for c in got:
        src = by_ts[c.ts_ns]
        assert np.array_equal(c.xyz, src.xyz) and np.array_equal(c.intensity, src.intensity)
        assert np.array_equal(c.ring, src.ring) and (c.source, c.frame) == ("lidar", "ego")
    assert pointcloud_topics_mcap(tmp_path / "own.mcap") == {"/points": "npz"}
    with pytest.raises(FileNotFoundError):
        list(read_pointclouds_mcap(tmp_path / "absent.mcap"))


def test_ros2_and_ros1_pointcloud2_topics_read_from_one_mcap(tmp_path):
    ring_fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1), ("ring", 12, UINT16, 1)]
    ring_blob = b"".join(struct.pack("<fffH", float(i), 1.0, 2.0, i) + b"\x77\x77" for i in range(5))
    nan_blob = struct.pack("<ffff", 1.0, 2.0, 3.0, 0.5) + struct.pack("<ffff", float("nan"), 0, 0, 0.5)
    own = Cloud(xyz=[[9.0, 8.0, 7.0]], intensity=[0.5], ts_ns=0, frame="ego")
    path = _write_mcap(tmp_path / "drive.mcap", [
        ("/lidar_top/points", "sensor_msgs/msg/PointCloud2", "ros2msg", "cdr", [
            (T0 + 10, _cdr_message(XYZI, 16, BLOB, width=3)),
            (T0 + 30, _cdr_message(XYZI, 16, nan_blob, width=2, dense=False, sec=5, nsec=6)),
        ]),
        ("/velodyne_points", "sensor_msgs/PointCloud2", "ros1msg", "ros1", [
            (T0 + 20, _ros1_message(ring_fields, 16, ring_blob, width=5)),
        ]),
        ("/points", "labelox.PointCloud", "jsonschema", "labelox-npz", [(T0 + 40, own.to_npz_bytes())]),
        # Channels that are not clouds: another ROS 2 type, and JSON on a topic named like a cloud topic.
        ("/imu", "sensor_msgs/msg/Imu", "ros2msg", "cdr", [(T0 + 15, b"\x00\x01\x00\x00" + b"\x01" * 64)]),
        ("/points_meta", "meta", "jsonschema", "json", [(T0 + 16, b"{}")]),
    ])

    assert pointcloud_topics_mcap(path) == {"/lidar_top/points": "cdr", "/velodyne_points": "ros1",
                                            "/points": "npz"}
    got = list(iter_pointcloud_messages_mcap(path))
    assert [(t, c.ts_ns) for t, c in got] == [
        ("/lidar_top/points", T0 + 10), ("/velodyne_points", T0 + 20), ("/lidar_top/points", T0 + 30),
        ("/points", T0 + 40)]
    ros2_a, ros1, ros2_b, private = (c for _t, c in got)
    assert ros2_a.xyz.tolist() == [list(p[:3]) for p in POINTS] and ros2_a.frame == "lidar_top"
    assert ros2_a.intensity.tolist() == [p[3] for p in POINTS] and ros2_a.ring is None
    assert ros1.xyz[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0] and ros1.ring.tolist() == [0, 1, 2, 3, 4]
    assert ros1.frame == "velodyne"
    assert ros2_b.xyz.tolist() == [[1.0, 2.0, 3.0]]               # its NaN point was dropped
    assert private.xyz.tolist() == [[9.0, 8.0, 7.0]] and private.frame == "ego"

    # The existing entry point yields the same clouds, and can be narrowed to one topic.
    assert [c.ts_ns for c in read_pointclouds_mcap(path)] == [T0 + 10, T0 + 20, T0 + 30, T0 + 40]
    assert [c.ts_ns for c in read_pointclouds_mcap(path, topics=["/velodyne_points"])] == [T0 + 20]
    assert list(read_pointclouds_mcap(path, topics=["/imu"])) == []
    # Header stamps instead of log times, for a recording whose log clock is not the sensor clock.
    stamps = [c.ts_ns for c in read_pointclouds_mcap(path, topics=["/lidar_top/points"], time_source="header")]
    assert stamps == [SEC * 1_000_000_000 + NSEC, 5 * 1_000_000_000 + 6]


def test_an_mcap_without_cloud_channels_yields_nothing(tmp_path):
    path = _write_mcap(tmp_path / "can.mcap", [("/can", "can", "jsonschema", "json", [(T0, b"{}")])])
    assert list(read_pointclouds_mcap(path)) == [] and pointcloud_topics_mcap(path) == {}


POINTCLOUD2_MSGDEF = """std_msgs/Header header
uint32 height
uint32 width
PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
================================================================================
MSG: sensor_msgs/PointField
uint8 INT8=1
uint8 UINT8=2
uint8 INT16=3
uint8 UINT16=4
uint8 INT32=5
uint8 UINT32=6
uint8 FLOAT32=7
uint8 FLOAT64=8
string name
uint32 offset
uint8 datatype
uint32 count
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def test_messages_serialized_by_mcap_ros2_support_read_back(tmp_path):
    """The encoder here is the library's, driven by the .msg definition; the decoder is ours."""
    ros2_writer = pytest.importorskip("mcap_ros2.writer")
    fields = [("x", 0, FLOAT64, 1), ("y", 8, FLOAT64, 1), ("z", 16, FLOAT64, 1), ("intensity", 24, UINT8, 1),
              ("ring", 25, UINT16, 1), ("timestamp", 28, FLOAT32, 1)]
    rows = [(float(i) + 0.125, -float(i), 100.0 + i, 5 * i, i, 0.01 * i) for i in range(1, 7)]
    data = b"".join(struct.pack("<dddBHxf", *r) for r in rows)             # 32 bytes, one pad byte at 27
    # Organised 2 x 3, and frame ids of several lengths so the library's own alignment is exercised.
    path = tmp_path / "ros2.mcap"
    with open(path, "wb") as f:
        w = ros2_writer.Writer(f)
        schema = w.register_msgdef("sensor_msgs/msg/PointCloud2", POINTCLOUD2_MSGDEF)
        for k, frame_id in enumerate(("", "os", "lidar", "os_sensor_top")):
            w.write_message("/ouster/points", schema, {
                "header": {"stamp": {"sec": 100 + k, "nanosec": 7}, "frame_id": frame_id},
                "height": 2, "width": 3,
                "fields": [{"name": n, "offset": o, "datatype": d, "count": c} for n, o, d, c in fields],
                "is_bigendian": False, "point_step": 32, "row_step": 96, "data": data, "is_dense": True,
            }, log_time=T0 + k, publish_time=T0 + k)
        w.finish()

    assert pointcloud_topics_mcap(path) == {"/ouster/points": "cdr"}
    clouds = list(read_pointclouds_mcap(path))
    assert [c.ts_ns for c in clouds] == [T0, T0 + 1, T0 + 2, T0 + 3]
    assert [c.frame for c in clouds] == ["sensor", "os", "lidar", "os_sensor_top"]   # empty id: "sensor"
    for c in clouds:
        np.testing.assert_allclose(c.xyz, [r[:3] for r in rows], rtol=1e-6)
        np.testing.assert_allclose(c.intensity, [r[3] / 255 for r in rows], rtol=1e-6)
        assert c.ring.tolist() == [r[4] for r in rows]

    # And the library's bytes are the bytes this file's hand writer produces for the same message.
    from mcap.reader import make_reader

    with open(path, "rb") as f:
        raw = [m.data for _s, _c, m in make_reader(f).iter_messages()]
    assert raw[2] == _cdr_message(fields, 32, data, width=3, height=2, frame_id="lidar", fill=0x00,
                                  sec=102, nsec=7, row_step=96)
    msg = parse_pointcloud2_cdr(raw[3])
    assert msg.frame_id == "os_sensor_top" and msg.stamp_ns == 103 * 1_000_000_000 + 7
    pts = decode_pointcloud2(msg)
    np.testing.assert_allclose(pts.time, [r[5] for r in rows], rtol=1e-6)
    assert pts.time_field == "timestamp"
