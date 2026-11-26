"""Radar sweeps from the formats they are published in, into one in-memory RadarScan in the ego frame.

Three sources, each read by its documented layout with no vendor SDK:

  nuScenes RADAR_*   `.pcd` files (PCD v0.7, binary or ascii) with the Continental ARS 408 fields
                     x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms
                     invalid_state pdh0 vx_rms vy_rms. Positions and velocities are in the radar's own frame
                     (x forward); `vx_comp, vy_comp` are the velocities with the ego motion removed. The
                     sample_data / calibrated_sensor tables place each file (`list_nuscenes_radar_sweeps`).
                     By default the devkit's filter is applied (invalid_state 0, dyn_prop 0..6, ambig_state
                     3), the one nuScenes' own tooling uses; `filter_points=False` keeps every return.
  PointCloud2        a ROS 1 or ROS 2 `sensor_msgs/PointCloud2` with x, y, z and velocity fields: a vector
                     (vx, vy[, vz]) and/or a radial speed (doppler / radial_velocity / range_rate / v_r /
                     velocity), plus rcs or a power field when present. This is how most radar drivers that
                     do not have their own message publish to rosbag2.
  RadarScan          `radar_msgs/msg/RadarScan` (ROS 2 CDR): Header, then RadarReturn[] of float32 range,
                     azimuth, elevation, doppler_velocity, amplitude. Spherical, so x/y/z come from
                     range/azimuth/elevation and the only velocity is the radial one.

The radial (Doppler) speed is what a radar actually measures. Where a source gives a velocity vector, the
radial speed is its projection on the line of sight; where it gives only the radial speed, the velocity
vector is that speed along the line of sight, and `vector` in the stats says which the sweep had.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from services.lidar.ingest.pointcloud2 import (
    PointCloud2,
    PointCloud2Error,
    _Cursor,
    parse_pointcloud2_cdr,
    parse_pointcloud2_ros1,
)

RADAR_CHANNELS = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")
RADARSCAN_SCHEMAS = ("radar_msgs/msg/RadarScan", "radar_msgs/RadarScan")
POINTCLOUD2_SCHEMAS = ("sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2")

# Field names in use for the same quantity across drivers; the first present wins.
_VX = ("vx_comp", "vx", "velocity_x", "v_x")
_VY = ("vy_comp", "vy", "velocity_y", "v_y")
_VZ = ("vz_comp", "vz", "velocity_z", "v_z")
_RADIAL = ("doppler", "radial_velocity", "range_rate", "v_r", "vr", "doppler_velocity", "velocity", "radial_speed")
_RCS = ("rcs", "rcs0", "power", "snr", "amplitude", "intensity")


class RadarFormatError(ValueError):
    """The bytes are not a radar sweep this reader can interpret; the message says which part."""


@dataclass
class RadarScan:
    """One sweep, every array the same length N, positions and velocities in `frame`."""

    xyz: np.ndarray                         # (N, 3) float32, metres
    velocity: np.ndarray                    # (N, 3) float32, m/s; ego-motion compensated when `compensated`
    radial: np.ndarray                      # (N,) float32, m/s along the line of sight, positive = receding
    rcs: np.ndarray                         # (N,) float32, dBsm or the driver's power units
    ts_ns: int = 0
    sensor: str = "radar"
    frame: str = "sensor"
    compensated: bool = False
    vector: bool = True                     # False: only the radial speed was measured
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    n_total: int = 0                        # returns in the source before filtering

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    def transformed(self, T: np.ndarray, frame: str) -> RadarScan:
        """The same sweep in another frame: positions moved by T, velocities rotated (they are directions)."""
        T = np.asarray(T, dtype=np.float64)
        xyz = self.xyz.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        vel = self.velocity.astype(np.float64) @ T[:3, :3].T
        return RadarScan(xyz=xyz.astype(np.float32), velocity=vel.astype(np.float32), radial=self.radial,
                         rcs=self.rcs, ts_ns=self.ts_ns, sensor=self.sensor, frame=frame,
                         compensated=self.compensated, vector=self.vector, extra=dict(self.extra),
                         n_total=self.n_total)

    def to_npz_bytes(self, sensor_origin: np.ndarray | None = None) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, xyz=self.xyz.astype(np.float32), velocity=self.velocity.astype(np.float32),
                            radial=self.radial.astype(np.float32), rcs=self.rcs.astype(np.float32),
                            sensor_origin=(np.zeros(3, np.float32) if sensor_origin is None
                                           else np.asarray(sensor_origin, np.float32)),
                            **{f"extra_{k}": v for k, v in self.extra.items()})
        return buf.getvalue()

    @classmethod
    def from_npz_bytes(cls, data: bytes, **meta) -> RadarScan:
        z = np.load(io.BytesIO(data))
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
        return cls(xyz=z["xyz"], velocity=z["velocity"], radial=z["radial"], rcs=z["rcs"], extra=extra,
                   n_total=int(len(z["xyz"])), **meta)


def radial_speed(xyz: np.ndarray, velocity: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    """Projection of each velocity on the line of sight from `origin` (the radar), positive moving away."""
    rel = xyz.astype(np.float64) - (np.zeros(3) if origin is None else np.asarray(origin, np.float64))
    norm = np.linalg.norm(rel, axis=1)
    norm[norm < 1e-6] = 1.0
    return (np.sum(rel * velocity.astype(np.float64), axis=1) / norm).astype(np.float32)


# --- PCD (nuScenes radar) ---------------------------------------------------------------------------------

_PCD_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
              ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8"}


def parse_pcd_fields(data: bytes) -> dict[str, np.ndarray]:
    """Every field of a PCD v0.7 file as its own array. ascii and binary; binary_compressed is refused (it is
    LZF-compressed and nuScenes never writes it)."""
    header: dict[str, list[str]] = {}
    pos = 0
    while True:
        end = data.find(b"\n", pos)
        if end < 0:
            raise RadarFormatError("PCD header has no DATA line")
        line = data[pos:end].decode("ascii", errors="replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            break
    try:
        names = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]
        n = int(header.get("POINTS", [str(int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0]))])[0])
    except (KeyError, ValueError, IndexError) as exc:
        raise RadarFormatError(f"PCD header is incomplete: {exc}") from exc
    if not (len(names) == len(sizes) == len(types) == len(counts)):
        raise RadarFormatError("PCD FIELDS/SIZE/TYPE/COUNT disagree in length")
    mode = header["DATA"][0].lower()
    cols: list[tuple[str, str, int]] = []
    for name, size, typ, count in zip(names, sizes, types, counts, strict=True):
        code = _PCD_TYPES.get((typ.upper(), size))
        if code is None:
            raise RadarFormatError(f"PCD field {name!r} has unsupported TYPE {typ} SIZE {size}")
        cols.append((name, code, count))
    if mode == "binary":
        dtype = np.dtype([(nm if c == 1 else nm, "<" + code, (c,) if c > 1 else ()) for nm, code, c in cols])
        need = n * dtype.itemsize
        if len(data) - pos < need:
            raise RadarFormatError(f"PCD body is {len(data) - pos} bytes, {n} points need {need}")
        rec = np.frombuffer(data, dtype=dtype, count=n, offset=pos)
        return {nm: np.asarray(rec[nm]) for nm, _, _ in cols}
    if mode == "ascii":
        rows = [ln.split() for ln in data[pos:].decode("ascii", errors="replace").splitlines() if ln.strip()]
        width = sum(c for _, _, c in cols)
        arr = np.array([r[:width] for r in rows[:n]], dtype=np.float64).reshape(-1, width)
        out, k = {}, 0
        for nm, code, c in cols:
            out[nm] = arr[:, k:k + c].astype(code).reshape(-1) if c == 1 else arr[:, k:k + c].astype(code)
            k += c
        return out
    raise RadarFormatError(f"PCD DATA {mode!r} is not supported (binary and ascii are)")


def _first(fields: dict[str, np.ndarray], names: tuple[str, ...]) -> tuple[str, np.ndarray] | None:
    lower = {k.lower(): k for k in fields}
    for nm in names:
        if nm in lower:
            return lower[nm], fields[lower[nm]]
    return None


def scan_from_fields(fields: dict[str, np.ndarray], *, ts_ns: int = 0, sensor: str = "radar",
                     frame: str = "sensor") -> RadarScan:
    """A RadarScan from named per-point arrays (x, y, z plus whichever velocity and power fields exist)."""
    lower = {k.lower(): v for k, v in fields.items()}
    if not {"x", "y"} <= set(lower):
        raise RadarFormatError(f"radar points need x and y; fields are {sorted(fields)}")
    x = lower["x"].astype(np.float32)
    y = lower["y"].astype(np.float32)
    z = lower["z"].astype(np.float32) if "z" in lower else np.zeros_like(x)
    xyz = np.stack([x, y, z], axis=1)
    vx, vy, vz = _first(fields, _VX), _first(fields, _VY), _first(fields, _VZ)
    rad = _first(fields, _RADIAL)
    compensated = bool(vx and vx[0].lower().endswith("_comp"))
    if vx is not None and vy is not None:
        vel = np.stack([vx[1].astype(np.float32), vy[1].astype(np.float32),
                        vz[1].astype(np.float32) if vz is not None else np.zeros_like(x)], axis=1)
        radial = radial_speed(xyz, vel)
        vector = True
    elif rad is not None:
        radial = rad[1].astype(np.float32)
        rel = xyz.astype(np.float64)
        norm = np.linalg.norm(rel, axis=1)
        norm[norm < 1e-6] = 1.0
        vel = (rel / norm[:, None] * radial[:, None]).astype(np.float32)
        vector = False
    else:
        raise RadarFormatError(f"radar points have no velocity field; fields are {sorted(fields)}")
    rcs_f = _first(fields, _RCS)
    rcs = rcs_f[1].astype(np.float32) if rcs_f is not None else np.zeros(len(x), np.float32)
    used = {"x", "y", "z"} | {f[0].lower() for f in (vx, vy, vz, rad, rcs_f) if f is not None}
    extra = {k: v for k, v in fields.items() if k.lower() not in used and np.asarray(v).ndim == 1}
    ok = np.isfinite(xyz).all(axis=1) & np.isfinite(vel).all(axis=1)
    return RadarScan(xyz=xyz[ok], velocity=vel[ok], radial=radial[ok], rcs=rcs[ok], ts_ns=int(ts_ns),
                     sensor=sensor, frame=frame, compensated=compensated, vector=vector,
                     extra={k: np.asarray(v)[ok] for k, v in extra.items()}, n_total=int(len(x)))


def read_nuscenes_radar_pcd(data: bytes, *, ts_ns: int = 0, sensor: str = "RADAR_FRONT",
                            filter_points: bool = True) -> RadarScan:
    """One nuScenes radar .pcd, in the radar's frame. With `filter_points`, the devkit's default filter."""
    fields = parse_pcd_fields(data)
    keep = None
    if filter_points:
        f = {k.lower(): v for k, v in fields.items()}
        keep = np.ones(len(next(iter(fields.values()))), dtype=bool)
        if "invalid_state" in f:
            keep &= f["invalid_state"] == 0
        if "dyn_prop" in f:
            keep &= (f["dyn_prop"] >= 0) & (f["dyn_prop"] <= 6)
        if "ambig_state" in f:
            keep &= f["ambig_state"] == 3
    n_total = len(next(iter(fields.values())))
    if keep is not None:
        fields = {k: (v[keep] if np.asarray(v).ndim == 1 else v[keep]) for k, v in fields.items()}
    scan = scan_from_fields(fields, ts_ns=ts_ns, sensor=sensor, frame="sensor")
    scan.n_total = int(n_total)
    return scan


@dataclass
class NuScenesRadarSweep:
    path: Path
    ts_ns: int
    channel: str
    sensor_to_ego: np.ndarray
    ego_to_global: np.ndarray
    is_key_frame: bool
    sample_ts_ns: int | None
    token: str
    scene_token: str | None = None


def list_nuscenes_radar_sweeps(root: str | Path, channels: tuple[str, ...] = RADAR_CHANNELS,
                               keyframes_only: bool = False, scene_token: str | None = None,
                               tables=None) -> list[NuScenesRadarSweep]:
    """Every radar sample_data row of a nuScenes directory (optionally one scene), in time order."""
    from services.imports.nuscenes_tables import load_tables

    tables = tables or load_tables(Path(root))
    out: list[NuScenesRadarSweep] = []
    for sd in tables.sample_data:
        cs = tables.calibrated_sensor.get(sd.get("calibrated_sensor_token", ""))
        sensor = tables.sensor.get(cs.get("sensor_token", "")) if cs else None
        if sensor is None or sensor.get("modality") != "radar" or sensor.get("channel") not in channels:
            continue
        if sd.get("ego_pose_token") not in tables.ego_pose:
            continue
        key = bool(sd.get("is_key_frame"))
        if keyframes_only and not key:
            continue
        sample = tables.sample.get(sd.get("sample_token", ""), {})
        if scene_token and sample.get("scene_token") != scene_token:
            continue
        out.append(NuScenesRadarSweep(
            path=tables.file_path(sd), ts_ns=int(sd["timestamp"]) * 1000, channel=sensor["channel"],
            sensor_to_ego=tables.sensor_to_ego(sd), ego_to_global=tables.ego_to_global(sd), is_key_frame=key,
            sample_ts_ns=int(sample["timestamp"]) * 1000 if "timestamp" in sample else None,
            token=sd.get("token", ""), scene_token=sample.get("scene_token")))
    out.sort(key=lambda s: (s.ts_ns, s.channel))
    return out


def read_nuscenes_radar_sweep(sweep: NuScenesRadarSweep, *, filter_points: bool = True) -> RadarScan:
    """One sweep in the ego frame. A keyframe sweep takes its sample's timestamp, the ts_ns the sample's camera
    frames were imported at, as the LiDAR keyframes do (services/lidar/ingest/nuscenes_sweeps.py)."""
    ts = sweep.sample_ts_ns if (sweep.is_key_frame and sweep.sample_ts_ns is not None) else sweep.ts_ns
    scan = read_nuscenes_radar_pcd(Path(sweep.path).read_bytes(), ts_ns=ts, sensor=sweep.channel,
                                   filter_points=filter_points)
    return scan.transformed(sweep.sensor_to_ego, "ego")


# --- PointCloud2 with velocity fields ---------------------------------------------------------------------

_PC2_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pointcloud2_fields(msg: PointCloud2) -> dict[str, np.ndarray]:
    """Every scalar field of a PointCloud2 as its own array (unorganised or organised, either byte order)."""
    order = ">" if msg.is_bigendian else "<"
    names, formats, offsets = [], [], []
    seen = set()
    for f in msg.fields:
        code = _PC2_TYPES.get(int(f.datatype))
        if code is None or f.name in seen:
            continue
        if f.offset + int(code[1:]) > msg.point_step:
            raise PointCloud2Error(f"field {f.name!r} does not fit in point_step {msg.point_step}")
        seen.add(f.name)
        names.append(f.name)
        formats.append(order + code)
        offsets.append(int(f.offset))
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(msg.point_step)})
    n = int(msg.height) * int(msg.width)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.height > 1 and msg.row_step > msg.width * msg.point_step and raw.size != n * msg.point_step:
        rows = np.lib.stride_tricks.as_strided(raw, shape=(int(msg.height), int(msg.width) * msg.point_step),
                                               strides=(int(msg.row_step), 1), writeable=False)
        rec = np.ascontiguousarray(rows).reshape(-1).view(dtype)
    else:
        if raw.size < n * msg.point_step:
            raise PointCloud2Error(f"data is {raw.size} bytes, {n} points need {n * msg.point_step}")
        rec = raw[:n * msg.point_step].view(dtype)
    return {nm: np.asarray(rec[nm]) for nm in names}


def scan_from_pointcloud2(buf: bytes | memoryview, encoding: str, *, sensor: str, ts_ns: int | None = None
                          ) -> RadarScan:
    msg = (parse_pointcloud2_cdr if encoding == "cdr" else parse_pointcloud2_ros1)(buf)
    return scan_from_fields(pointcloud2_fields(msg), ts_ns=msg.stamp_ns if ts_ns is None else ts_ns,
                            sensor=sensor, frame=msg.frame_id or "sensor")


# --- radar_msgs/msg/RadarScan -----------------------------------------------------------------------------

def parse_radarscan_cdr(buf: bytes | memoryview, *, sensor: str = "radar") -> RadarScan:
    """`radar_msgs/msg/RadarScan` as rosbag2 stores it: CDR, Header then sequence<RadarReturn>."""
    if len(buf) < 4:
        raise RadarFormatError("CDR message shorter than its 4-byte encapsulation header")
    rep = bytes(buf[:2])
    if rep not in (b"\x00\x00", b"\x00\x01"):
        raise RadarFormatError(f"unsupported CDR encapsulation {rep.hex()}")
    order = "<" if rep == b"\x00\x01" else ">"
    c = _Cursor(buf, 4, order, aligned=True, what="CDR RadarScan")
    try:
        sec, nsec = c.i32(), c.u32()
        frame_id = c.string()
        n = c.u32()
        if n > 10_000_000:
            raise RadarFormatError(f"{n} returns; this is not a RadarScan")
        c._align(4)
        body = c._take(n * 20)
    except PointCloud2Error as exc:
        raise RadarFormatError(str(exc)) from exc
    arr = np.frombuffer(bytes(body), dtype=np.dtype(order + "f4")).reshape(n, 5) if n else np.zeros((0, 5), np.float32)
    rng, az, el, dop, amp = (arr[:, i].astype(np.float64) for i in range(5))
    xyz = np.stack([rng * np.cos(el) * np.cos(az), rng * np.cos(el) * np.sin(az), rng * np.sin(el)], axis=1)
    return scan_from_fields({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "doppler": dop, "amplitude": amp},
                            ts_ns=int(sec) * 1_000_000_000 + int(nsec), sensor=sensor, frame=frame_id or "sensor")


def encode_radarscan_cdr(returns: np.ndarray, *, stamp_ns: int, frame_id: str = "radar") -> bytes:
    """Serialize (range, azimuth, elevation, doppler_velocity, amplitude) rows as a CDR RadarScan. The writer
    side of `parse_radarscan_cdr`, for recording fixtures and round-trip tests."""
    out = bytearray(b"\x00\x01\x00\x00")
    body = bytearray()
    body += struct.pack("<iI", int(stamp_ns // 1_000_000_000), int(stamp_ns % 1_000_000_000))
    name = frame_id.encode() + b"\x00"
    body += struct.pack("<I", len(name)) + name
    body += b"\x00" * ((-len(body)) % 4)
    rows = np.asarray(returns, dtype="<f4").reshape(-1, 5)
    body += struct.pack("<I", len(rows)) + rows.tobytes()
    return bytes(out + body)


def iter_radar_mcap(path: str | Path, topics: list[str] | None = None, time_source: str = "log"):
    """Yield (topic, RadarScan, kind) for every radar message in an MCAP: RadarScan channels, and PointCloud2
    channels whose points carry a velocity field (a LiDAR PointCloud2 has none and is passed over)."""
    from mcap.reader import make_reader

    with open(Path(path), "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=topics):
            name = schema.name if schema is not None else ""
            enc = (channel.message_encoding or "").lower()
            ts = None if time_source == "header" else int(message.log_time)
            if name in RADARSCAN_SCHEMAS and enc == "cdr":
                scan = parse_radarscan_cdr(message.data, sensor=channel.topic)
                if ts is not None:
                    scan.ts_ns = ts
                yield channel.topic, scan, "mcap_radarscan"
            elif name in POINTCLOUD2_SCHEMAS and enc in ("cdr", "ros1"):
                try:
                    scan = scan_from_pointcloud2(message.data, enc, sensor=channel.topic, ts_ns=ts)
                except RadarFormatError:
                    continue                                    # no velocity field: a LiDAR, not a radar
                yield channel.topic, scan, "mcap_pc2"


def radar_topics_mcap(path: str | Path) -> dict[str, str]:
    """{topic: schema} for the channels that could carry radar (RadarScan, or PointCloud2 to be checked)."""
    from mcap.reader import make_reader

    with open(Path(path), "rb") as f:
        summary = make_reader(f).get_summary()
    out: dict[str, str] = {}
    if summary is None:
        return out
    for ch in summary.channels.values():
        sc = summary.schemas.get(ch.schema_id)
        if sc is not None and sc.name in RADARSCAN_SCHEMAS + POINTCLOUD2_SCHEMAS:
            out[ch.topic] = sc.name
    return out
