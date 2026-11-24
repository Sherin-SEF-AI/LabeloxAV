"""`sensor_msgs/PointCloud2`: the message off the wire, and its points into the internal Cloud.

Two layers, both pure Python + numpy with no ROS install and no generated message classes:

  deserialize   `parse_pointcloud2_cdr` (ROS 2: XCDR1 plain CDR, either byte order, behind the 4-byte
                encapsulation header) and `parse_pointcloud2_ros1` (ROS 1: little-endian, unaligned). Both
                read this one message type by its fixed, documented layout,

                    std_msgs/Header header        (ROS 1 has a leading uint32 seq, ROS 2 does not)
                    uint32 height, width
                    PointField[] fields           (string name, uint32 offset, uint8 datatype, uint32 count)
                    bool is_bigendian
                    uint32 point_step, row_step
                    uint8[] data
                    bool is_dense

                rather than through a schema-driven decoder, so a file whose schema record is empty or
                abbreviated still reads, and ROS 1 needs no decoding library at all.

  decode        `decode_pointcloud2` turns the data blob into arrays, honouring each field's offset and
                datatype (so padding and unknown fields between the ones we want are skipped), point_step,
                row_step (rows of an organised cloud may be padded), is_bigendian, and both organised
                (height > 1) and unorganised clouds. Points with a non-finite x, y or z are dropped; that
                is what is_dense = false announces, and it is applied when is_dense is true as well, since
                a driver that mislabels its cloud is cheaper to tolerate than a NaN in every later stage.

What is read: x, y, z (any numeric datatype), and when present intensity, ring and a per-point time. The
internal Cloud has no per-point time, so `DecodedPoints.to_cloud()` leaves it behind; it stays available on
DecodedPoints for a caller that wants to de-skew a sweep.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from services.lidar.ingest.normalize import Cloud

# sensor_msgs/PointField datatype constants -> numpy type code (byte order is applied per message).
_DATATYPES: dict[int, str] = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

# Field names in use across drivers for the same quantity; the first present wins.
_INTENSITY_NAMES = ("intensity", "intensities", "i", "reflectivity")
_RING_NAMES = ("ring", "channel", "laser_id", "line")
_TIME_NAMES = ("time", "t", "timestamp", "time_stamp", "offset_time", "time_offset")

ROS2_SCHEMA = "sensor_msgs/msg/PointCloud2"
ROS1_SCHEMA = "sensor_msgs/PointCloud2"


class PointCloud2Error(ValueError):
    """The bytes are not a PointCloud2 this reader can interpret. The message says which part."""


@dataclass
class PointField:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass
class PointCloud2:
    """One deserialized message. `data` is the raw point blob, untouched."""

    frame_id: str
    stamp_ns: int                    # header.stamp as nanoseconds
    height: int
    width: int
    fields: list[PointField]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes | memoryview
    is_dense: bool


@dataclass
class DecodedPoints:
    """The valid points of one message, row-major, every array the same length."""

    xyz: np.ndarray                          # (N, 3) float64 as decoded (float32 sources widen exactly)
    intensity: np.ndarray | None = None      # (N,) float32, see decode_pointcloud2 for the scaling
    ring: np.ndarray | None = None           # (N,) int16
    time: np.ndarray | None = None           # (N,) float64, the driver's own units, unconverted
    time_field: str | None = None            # which field `time` came from
    n_total: int = 0                         # points in the message, before dropping non-finite ones
    frame_id: str = ""
    stamp_ns: int = 0

    @property
    def n_dropped(self) -> int:
        return self.n_total - int(self.xyz.shape[0])

    def to_cloud(self, ts_ns: int | None = None, source: str = "lidar") -> Cloud:
        """The internal Cloud, in the message's own frame. ts_ns defaults to the header stamp."""
        intensity = (self.intensity if self.intensity is not None
                     else np.zeros(self.xyz.shape[0], dtype=np.float32))     # no field: zeros, as Cloud does
        return Cloud(xyz=self.xyz, intensity=intensity, ring=self.ring,
                     ts_ns=self.stamp_ns if ts_ns is None else int(ts_ns), source=source,
                     frame=self.frame_id or "sensor")


# --- decoding the point blob ------------------------------------------------------------------------------

def _pick(fields: dict[str, PointField], names: tuple[str, ...]) -> PointField | None:
    for n in names:
        if n in fields:
            return fields[n]
    return None


def decode_pointcloud2(msg: PointCloud2) -> DecodedPoints:
    """The points of a message as arrays. See the module docstring for what is honoured.

    Intensity scaling follows the other readers in this package (0..1 where the source range is known):
    a uint8 field is divided by 255 and a uint16 field by 65535, because the type bounds the range. A float
    field is passed through unchanged; its range is the driver's business (0..1, 0..255 and raw return
    energies all occur) and guessing it per message would make two sweeps of one drive disagree.
    """
    if msg.point_step <= 0:
        raise PointCloud2Error(f"point_step is {msg.point_step}")
    n_points = int(msg.height) * int(msg.width)
    by_name: dict[str, PointField] = {}
    for f in msg.fields:
        by_name.setdefault(f.name.strip().lower(), f)       # a repeated name: the first definition stands
    missing = [n for n in ("x", "y", "z") if n not in by_name]
    if missing:
        raise PointCloud2Error(f"PointCloud2 has no {'/'.join(missing)} field; fields are "
                               f"{[f.name for f in msg.fields]}")

    wanted: dict[str, PointField] = {n: by_name[n] for n in ("x", "y", "z")}
    intensity_f = _pick(by_name, _INTENSITY_NAMES)
    ring_f = _pick(by_name, _RING_NAMES)
    time_f = _pick(by_name, _TIME_NAMES)
    for key, optional in (("intensity", intensity_f), ("ring", ring_f), ("time", time_f)):
        if optional is not None:
            wanted[key] = optional

    order = ">" if msg.is_bigendian else "<"
    names, formats, offsets = [], [], []
    for key, f in wanted.items():
        code = _DATATYPES.get(int(f.datatype))
        if code is None:
            raise PointCloud2Error(f"field {f.name!r} has unknown datatype {f.datatype}")
        size = int(code[1:])
        if f.offset < 0 or f.offset + size > msg.point_step:
            raise PointCloud2Error(f"field {f.name!r} at offset {f.offset} ({size} bytes) does not fit "
                                   f"in point_step {msg.point_step}")
        names.append(key)
        formats.append(order + code)        # count > 1: only the first element is read
        offsets.append(int(f.offset))
    # A record dtype with explicit offsets and the full point_step as its size: every byte the message
    # puts between or after these fields (padding, other fields) is stepped over, not interpreted.
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(msg.point_step)})

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    packed = n_points * msg.point_step
    row_bytes = int(msg.width) * msg.point_step
    if n_points == 0:
        records = np.empty(0, dtype=dtype)
    elif msg.height > 1 and msg.row_step > row_bytes and raw.size != packed:
        # Organised cloud with padded rows. The last row may or may not carry its padding. (Data that is
        # exactly height * width points long has no room for row padding, whatever row_step claims.)
        need = (msg.height - 1) * msg.row_step + row_bytes
        if raw.size < need:
            raise PointCloud2Error(f"data is {raw.size} bytes, {msg.height} rows of row_step "
                                   f"{msg.row_step} need at least {need}")
        rows = np.lib.stride_tricks.as_strided(
            raw, shape=(int(msg.height), row_bytes), strides=(int(msg.row_step), 1), writeable=False)
        records = np.ascontiguousarray(rows).reshape(-1).view(dtype)
    else:
        # Contiguous points. row_step is not trusted beyond this: drivers that publish it as 0, or as the
        # size of the whole cloud, still put height * width points of point_step bytes in data.
        if raw.size < packed:
            raise PointCloud2Error(f"data is {raw.size} bytes, {msg.height} x {msg.width} points of "
                                   f"point_step {msg.point_step} need {packed}")
        records = raw[:packed].view(dtype)

    xyz = np.stack([records["x"].astype(np.float64), records["y"].astype(np.float64),
                    records["z"].astype(np.float64)], axis=1) if n_points else np.zeros((0, 3))
    valid = np.isfinite(xyz).all(axis=1)
    out = DecodedPoints(xyz=xyz[valid], n_total=n_points, frame_id=msg.frame_id, stamp_ns=msg.stamp_ns)

    if intensity_f is not None:
        inten = records["intensity"][valid].astype(np.float32)
        scale = {2: 255.0, 4: 65535.0}.get(int(intensity_f.datatype))
        out.intensity = inten / scale if scale else inten
    if ring_f is not None:
        out.ring = records["ring"][valid].astype(np.int16)
    if time_f is not None:
        out.time = records["time"][valid].astype(np.float64)
        out.time_field = time_f.name
    return out


# --- deserializing the message ----------------------------------------------------------------------------

class _Cursor:
    """Sequential reads over a serialized message. `align_base` is where CDR alignment is measured from
    (the first byte after the encapsulation header); with alignment off it reads ROS 1's packed layout."""

    def __init__(self, buf: bytes | memoryview, pos: int, order: str, aligned: bool, what: str) -> None:
        self.buf = memoryview(buf)
        self.pos = pos
        self.base = pos
        self.order = order
        self.aligned = aligned
        self.what = what

    def _take(self, n: int) -> memoryview:
        if n < 0 or self.pos + n > len(self.buf):
            raise PointCloud2Error(f"{self.what}: message ends at byte {len(self.buf)}, "
                                   f"needed {n} more at {self.pos}")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def _align(self, size: int) -> None:
        if self.aligned:
            self.pos += (-(self.pos - self.base)) % size

    def u8(self) -> int:
        return self._take(1)[0]

    def u32(self) -> int:
        self._align(4)
        return struct.unpack(self.order + "I", self._take(4))[0]

    def i32(self) -> int:
        self._align(4)
        return struct.unpack(self.order + "i", self._take(4))[0]

    def string(self) -> str:
        raw = bytes(self._take(self.u32()))
        # CDR counts and writes a terminating NUL; ROS 1 does not.
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    def octets(self) -> memoryview:
        return self._take(self.u32())


def _read_body(c: _Cursor, sec: int, nsec: int) -> PointCloud2:
    frame_id = c.string()
    height, width = c.u32(), c.u32()
    n_fields = c.u32()
    if n_fields > 4096:
        raise PointCloud2Error(f"{c.what}: {n_fields} point fields; this is not a PointCloud2")
    fields = []
    for _ in range(n_fields):
        name = c.string()
        offset = c.u32()
        datatype = c.u8()
        count = c.u32()
        fields.append(PointField(name=name, offset=offset, datatype=datatype, count=count))
    is_bigendian = bool(c.u8())
    point_step, row_step = c.u32(), c.u32()
    data = c.octets()
    is_dense = bool(c.u8())
    return PointCloud2(frame_id=frame_id, stamp_ns=int(sec) * 1_000_000_000 + int(nsec), height=height,
                       width=width, fields=fields, is_bigendian=is_bigendian, point_step=point_step,
                       row_step=row_step, data=data, is_dense=is_dense)


def parse_pointcloud2_cdr(buf: bytes | memoryview) -> PointCloud2:
    """A ROS 2 `sensor_msgs/msg/PointCloud2` as DDS puts it on the wire and rosbag2/MCAP store it.

    The first four bytes are the encapsulation header: a representation id (0x0000 big-endian CDR, 0x0001
    little-endian CDR) and two option bytes. Every primitive after it is aligned to its own size, measured
    from the end of that header. Other representations (parameter lists, XCDR2) are not what ROS 2 writes
    for a message and are refused rather than misread.
    """
    if len(buf) < 4:
        raise PointCloud2Error("CDR message shorter than its 4-byte encapsulation header")
    rep = bytes(buf[:2])
    if rep == b"\x00\x01":
        order = "<"
    elif rep == b"\x00\x00":
        order = ">"
    else:
        raise PointCloud2Error(f"unsupported CDR encapsulation {rep.hex()} (expected 0000 or 0001)")
    c = _Cursor(buf, 4, order, aligned=True, what="CDR PointCloud2")
    sec, nsec = c.i32(), c.u32()                       # builtin_interfaces/Time: int32 sec, uint32 nanosec
    return _read_body(c, sec, nsec)


def parse_pointcloud2_ros1(buf: bytes | memoryview) -> PointCloud2:
    """A ROS 1 `sensor_msgs/PointCloud2`: little-endian, no alignment, strings without a terminator, and a
    Header that starts with a uint32 sequence number."""
    c = _Cursor(buf, 0, "<", aligned=False, what="ROS 1 PointCloud2")
    c.u32()                                            # header.seq
    sec, nsec = c.u32(), c.u32()                       # time: uint32 secs, uint32 nsecs
    return _read_body(c, sec, nsec)
