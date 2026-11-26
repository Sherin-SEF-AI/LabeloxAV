"""Radar (#10): the published formats parsed on fixtures written in those formats, then the product paths.

There is no radar data on this machine. Every fixture here is generated in the format it imitates: a nuScenes
radar .pcd with the ARS 408 field layout and the tables that place it, a ROS 2 `radar_msgs/RadarScan` CDR
message and a `sensor_msgs/PointCloud2` with velocity fields recorded in an MCAP, and the database rows a
session with a 3D track would have. Each test knows its answer in advance (a return at a known place moving at a
known speed), so it checks the geometry and the arithmetic, not only that something came back.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import uuid
from pathlib import Path

import numpy as np
import pytest

from services.lidar.ingest.radar import (
    RadarFormatError,
    encode_radarscan_cdr,
    list_nuscenes_radar_sweeps,
    parse_pcd_fields,
    parse_radarscan_cdr,
    read_nuscenes_radar_pcd,
    read_nuscenes_radar_sweep,
    scan_from_pointcloud2,
)

NUSC_FIELDS = ["x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy", "vx_comp", "vy_comp", "is_quality_valid",
               "ambig_state", "x_rms", "y_rms", "invalid_state", "pdh0", "vx_rms", "vy_rms"]
NUSC_SIZE = [4, 4, 4, 1, 2, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1]
NUSC_TYPE = ["F", "F", "F", "I", "I", "F", "F", "F", "F", "F", "I", "I", "I", "I", "I", "I", "I", "I"]
_FMT = {("F", 4): "f", ("I", 1): "b", ("I", 2): "h"}


def nuscenes_radar_pcd(points: list[dict]) -> bytes:
    """A radar .pcd exactly as nuScenes ships it: PCD v0.7 header, then packed little-endian binary rows."""
    head = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
            f"FIELDS {' '.join(NUSC_FIELDS)}\nSIZE {' '.join(map(str, NUSC_SIZE))}\n"
            f"TYPE {' '.join(NUSC_TYPE)}\nCOUNT {' '.join(['1'] * len(NUSC_FIELDS))}\n"
            f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n")
    fmt = "<" + "".join(_FMT[(t, s)] for t, s in zip(NUSC_TYPE, NUSC_SIZE, strict=True))
    body = b"".join(struct.pack(fmt, *[p.get(f, 0) for f in NUSC_FIELDS]) for p in points)
    return head.encode() + body


def _pt(x, y, vx_comp=0.0, vy_comp=0.0, **kw):
    base = {"x": x, "y": y, "z": 0.0, "dyn_prop": 0, "id": 1, "rcs": 5.0, "vx": vx_comp, "vy": vy_comp,
            "vx_comp": vx_comp, "vy_comp": vy_comp, "ambig_state": 3, "invalid_state": 0}
    return {**base, **kw}


# ---- formats -----------------------------------------------------------------------------------------------

def test_a_nuscenes_radar_pcd_reads_every_field_and_applies_the_devkit_filter():
    pts = [_pt(20.0, 0.0, 5.0, 0.0), _pt(10.0, 10.0, -3.0, 0.0), _pt(30.0, 1.0, invalid_state=1),
           _pt(15.0, -2.0, ambig_state=1), _pt(8.0, 0.0, dyn_prop=7)]
    raw = nuscenes_radar_pcd(pts)
    fields = parse_pcd_fields(raw)
    assert list(fields) == NUSC_FIELDS and len(fields["x"]) == 5
    assert fields["id"].dtype == np.int16 and fields["dyn_prop"].dtype == np.int8

    scan = read_nuscenes_radar_pcd(raw, ts_ns=123, sensor="RADAR_FRONT")
    assert scan.n_total == 5 and scan.n == 2              # invalid, ambiguous and dyn_prop 7 are dropped
    assert scan.compensated and scan.vector
    # radial = v . line of sight: straight ahead at 5 m/s away, and (-3, 0) seen at 45 degrees
    assert scan.radial.tolist() == pytest.approx([5.0, -3.0 / math.sqrt(2)], abs=1e-5)
    assert read_nuscenes_radar_pcd(raw, filter_points=False).n == 5


def test_ascii_pcd_and_bad_headers():
    ascii_pcd = (b"VERSION 0.7\nFIELDS x y z vx vy\nSIZE 4 4 4 4 4\nTYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH 2\n"
                 b"HEIGHT 1\nPOINTS 2\nDATA ascii\n1 0 0 2 0\n0 4 0 0 -1\n")
    f = parse_pcd_fields(ascii_pcd)
    assert f["x"].tolist() == [1.0, 0.0] and f["vy"].tolist() == [0.0, -1.0]
    with pytest.raises(RadarFormatError, match="DATA"):
        parse_pcd_fields(b"VERSION 0.7\nFIELDS x\n")
    with pytest.raises(RadarFormatError, match="binary_compressed"):
        parse_pcd_fields(ascii_pcd.replace(b"DATA ascii", b"DATA binary_compressed"))


def test_radarscan_cdr_round_trip_places_returns_by_range_and_angles():
    rows = np.array([[10.0, 0.0, 0.0, -2.5, 30.0], [20.0, math.pi / 2, 0.0, 1.0, 12.0],
                     [5.0, 0.0, math.radians(30), 0.0, 3.0]], dtype=np.float32)
    msg = encode_radarscan_cdr(rows, stamp_ns=1_700_000_000_123_456_789, frame_id="radar_front")
    scan = parse_radarscan_cdr(msg, sensor="/radar/front")
    assert scan.ts_ns == 1_700_000_000_123_456_789 and scan.frame == "radar_front" and scan.n == 3
    assert scan.xyz[0].tolist() == pytest.approx([10, 0, 0], abs=1e-4)
    assert scan.xyz[1].tolist() == pytest.approx([0, 20, 0], abs=1e-4)
    assert scan.xyz[2].tolist() == pytest.approx([5 * math.cos(math.radians(30)), 0, 2.5], abs=1e-4)
    assert scan.radial.tolist() == pytest.approx([-2.5, 1.0, 0.0])
    assert not scan.vector and scan.rcs.tolist() == pytest.approx([30, 12, 3])
    # the velocity vector of a radial-only reading lies along the line of sight
    assert scan.velocity[1].tolist() == pytest.approx([0, 1.0, 0], abs=1e-5)
    with pytest.raises(RadarFormatError):
        parse_radarscan_cdr(b"\x02\x00\x00\x00junk")


class _Cdr:
    def __init__(self):
        self.b = bytearray()

    def _al(self, n):
        self.b += b"\x00" * ((-len(self.b)) % n)

    def u8(self, v):
        self.b += struct.pack("<B", v)

    def u32(self, v):
        self._al(4)
        self.b += struct.pack("<I", v)

    def i32(self, v):
        self._al(4)
        self.b += struct.pack("<i", v)

    def s(self, v):
        raw = v.encode() + b"\x00"
        self.u32(len(raw))
        self.b += raw

    def done(self):
        return b"\x00\x01\x00\x00" + bytes(self.b)


def pointcloud2_cdr(fields: list[tuple[str, int, int]], rows: list[tuple], fmt: str, stamp_ns: int,
                    frame_id: str = "radar") -> bytes:
    """A ROS 2 sensor_msgs/PointCloud2 in CDR, every primitive aligned as CDR requires."""
    step = struct.calcsize("<" + fmt)
    data = b"".join(struct.pack("<" + fmt, *r) for r in rows)
    w = _Cdr()
    w.i32(stamp_ns // 1_000_000_000)
    w.u32(stamp_ns % 1_000_000_000)
    w.s(frame_id)
    w.u32(1)
    w.u32(len(rows))
    w.u32(len(fields))
    for name, off, dt in fields:
        w.s(name)
        w.u32(off)
        w.u8(dt)
        w.u32(1)
    w.u8(0)
    w.u32(step)
    w.u32(step * len(rows))
    w.u32(len(data))
    w.b += data
    w.u8(1)
    return w.done()


F32 = 7
RADAR_PC2 = [("x", 0, F32), ("y", 4, F32), ("z", 8, F32), ("vx", 12, F32), ("vy", 16, F32), ("rcs", 20, F32)]
LIDAR_PC2 = [("x", 0, F32), ("y", 4, F32), ("z", 8, F32), ("intensity", 12, F32)]


def test_a_pointcloud2_with_velocity_fields_is_a_radar_and_one_without_is_not():
    msg = pointcloud2_cdr(RADAR_PC2, [(10, 0, 0, 4, 0, 7.5), (0, 10, 0, 0, -2, 1.0)], "ffffff", 5_000_000_000)
    scan = scan_from_pointcloud2(msg, "cdr", sensor="/radar")
    assert scan.n == 2 and scan.vector and not scan.compensated
    assert scan.radial.tolist() == pytest.approx([4.0, -2.0]) and scan.rcs.tolist() == [7.5, 1.0]
    lidar = pointcloud2_cdr(LIDAR_PC2, [(1, 2, 3, 0.5)], "ffff", 5_000_000_000)
    with pytest.raises(RadarFormatError, match="no velocity"):
        scan_from_pointcloud2(lidar, "cdr", sensor="/lidar")


def _write_radar_mcap(path: Path, t0: int, frames: int = 3) -> None:
    from mcap.writer import Writer

    with open(path, "wb") as f:
        w = Writer(f)
        w.start()
        s_scan = w.register_schema(name="radar_msgs/msg/RadarScan", encoding="ros2msg", data=b"")
        s_pc2 = w.register_schema(name="sensor_msgs/msg/PointCloud2", encoding="ros2msg", data=b"")
        c_scan = w.register_channel(topic="/radar/front/scan", message_encoding="cdr", schema_id=s_scan)
        c_pc2 = w.register_channel(topic="/radar/left/points", message_encoding="cdr", schema_id=s_pc2)
        c_lid = w.register_channel(topic="/lidar/points", message_encoding="cdr", schema_id=s_pc2)
        for i in range(frames):
            ts = t0 + i * 50_000_000
            w.add_message(c_scan, log_time=ts, publish_time=ts, data=encode_radarscan_cdr(
                np.array([[12.0, 0.0, 0.0, 3.0, 20.0]], np.float32), stamp_ns=ts))
            w.add_message(c_pc2, log_time=ts, publish_time=ts, data=pointcloud2_cdr(
                RADAR_PC2, [(8, 1, 0, 1, 0, 2.0)], "ffffff", ts))
            w.add_message(c_lid, log_time=ts, publish_time=ts, data=pointcloud2_cdr(
                LIDAR_PC2, [(1, 2, 3, 0.5)], "ffff", ts))
        w.finish()


def test_mcap_radar_channels_are_found_and_a_lidar_is_passed_over(tmp_path):
    from services.lidar.ingest.radar import iter_radar_mcap, radar_topics_mcap

    p = tmp_path / "radar.mcap"
    _write_radar_mcap(p, 1_000_000_000_000)
    assert set(radar_topics_mcap(p)) == {"/radar/front/scan", "/radar/left/points", "/lidar/points"}
    got = list(iter_radar_mcap(p))
    assert {(t, k) for t, _s, k in got} == {("/radar/front/scan", "mcap_radarscan"),
                                            ("/radar/left/points", "mcap_pc2")}
    assert len(got) == 6


def _nuscenes_dir(root: Path, scene_t0_us: int) -> Path:
    """A minimal nuScenes v1.0 directory: one scene, two samples, a front radar sweep per sample (rotated 90
    degrees on the car, 2 m ahead) and one non-keyframe sweep between them."""
    tdir = root / "v1.0-mini"
    tdir.mkdir(parents=True)
    (root / "sweeps" / "RADAR_FRONT").mkdir(parents=True)
    (root / "samples" / "RADAR_FRONT").mkdir(parents=True)
    yaw = math.pi / 2
    q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    tables = {
        "scene": [{"token": "sc1", "name": "scene-0001", "first_sample_token": "s1", "log_token": "l1"}],
        "log": [{"token": "l1", "location": "test"}],
        "sample": [{"token": "s1", "timestamp": scene_t0_us, "scene_token": "sc1"},
                   {"token": "s2", "timestamp": scene_t0_us + 500_000, "scene_token": "sc1"}],
        "sensor": [{"token": "rf", "channel": "RADAR_FRONT", "modality": "radar"},
                   {"token": "lt", "channel": "LIDAR_TOP", "modality": "lidar"}],
        "calibrated_sensor": [{"token": "cs_rf", "sensor_token": "rf", "translation": [2.0, 0.0, 0.5],
                               "rotation": q, "camera_intrinsic": []}],
        "ego_pose": [{"token": f"ep{i}", "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
                      "timestamp": scene_t0_us + i * 250_000} for i in range(3)],
        "sample_annotation": [], "instance": [], "category": [], "attribute": [], "visibility": [],
    }
    sd = []
    for i, (key, sample) in enumerate(((True, "s1"), (False, "s2"), (True, "s2"))):
        ts = scene_t0_us + i * 250_000
        rel = f"{'samples' if key else 'sweeps'}/RADAR_FRONT/r{i}.pcd"
        # In the radar's own frame: 10 m along its x (the car's left), moving away at 4 m/s.
        (root / rel).write_bytes(nuscenes_radar_pcd([_pt(10.0, 0.0, 4.0, 0.0)]))
        sd.append({"token": f"sd{i}", "sample_token": sample, "ego_pose_token": f"ep{i}",
                   "calibrated_sensor_token": "cs_rf", "timestamp": ts, "fileformat": "pcd", "is_key_frame": key,
                   "filename": rel})
    tables["sample_data"] = sd
    for name, rows in tables.items():
        (tdir / f"{name}.json").write_text(json.dumps(rows))
    return root


def test_nuscenes_radar_sweeps_are_listed_and_placed_in_the_ego_frame(tmp_path):
    root = _nuscenes_dir(tmp_path / "nusc", 1_600_000_000_000_000)
    sweeps = list_nuscenes_radar_sweeps(root)
    assert [s.channel for s in sweeps] == ["RADAR_FRONT"] * 3
    assert [s.is_key_frame for s in sweeps] == [True, False, True]
    assert len(list_nuscenes_radar_sweeps(root, keyframes_only=True)) == 2
    scan = read_nuscenes_radar_sweep(sweeps[0])
    # radar x is the car's +y (yawed 90 degrees), mounted 2 m ahead and 0.5 m up
    assert scan.frame == "ego" and scan.xyz[0].tolist() == pytest.approx([2.0, 10.0, 0.5], abs=1e-5)
    assert scan.velocity[0].tolist() == pytest.approx([0.0, 4.0, 0.0], abs=1e-5)
    assert scan.radial[0] == pytest.approx(4.0)
    assert scan.ts_ns == 1_600_000_000_000_000 * 1000          # keyframe: the sample's time


# ---- fusion arithmetic -------------------------------------------------------------------------------------

def test_fuse_radial_replaces_only_the_radial_component_and_gates_disagreement():
    from services.oraclyx.radar import fuse_radial, fuse_track3d

    # Track says 10 m/s along +x; seen straight ahead, radar says 12: radial replaced, speed 12.
    r = fuse_radial([10.0, 0.0], [20.0, 0.0], 12.0, tol_mps=3.0)
    assert r["source"] == "radar" and r["speed_mps"] == pytest.approx(12.0)
    # Seen side-on the radar measures nothing of a crossing object's speed: radial 0, the track's 10 stands.
    r = fuse_radial([10.0, 0.0], [0.0, 15.0], 0.4, tol_mps=3.0)
    assert r["source"] == "radar" and r["velocity"] == pytest.approx([10.0, 0.4])
    # A reading far from the track is a different object, not a better measurement.
    r = fuse_radial([10.0, 0.0], [20.0, 0.0], -5.0, tol_mps=3.0)
    assert r["source"] == "track_rejected_radar" and r["speed_mps"] == pytest.approx(10.0)
    assert fuse_radial([1.0, 1.0], [1.0, 0.0], None)["source"] == "track"
    s = fuse_track3d([{"ts_ns": 1, "velocity": [5, 0], "los": [1, 0], "radial": 5.5},
                      {"ts_ns": 2, "velocity": [5, 0], "los": [1, 0], "radial": None}])
    assert (s["fused"], s["no_radar"], s["source"]) == (1, 1, "radar")


def test_returns_in_box_respects_yaw_and_margin():
    from services.sensors.radar import returns_in_box, track_velocities

    pts = np.array([[10.0, 0.0, 0.0], [10.0, 2.6, 0.0], [10.0, 3.2, 0.0]])
    assert returns_in_box(pts, [10, 0, 0], [4, 2, 1.5], 0.0, margin=1.0).tolist() == [True, False, False]
    # turned 90 degrees, the 4 m length lies along y
    assert returns_in_box(pts, [10, 0, 0], [4, 2, 1.5], math.pi / 2, margin=1.0).tolist() == [True, True, False]
    v = track_velocities(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]]), np.array([0, 10**8, 2 * 10**8]))
    assert v[:, 0].tolist() == pytest.approx([10, 10, 10])


# ---- database ----------------------------------------------------------------------------------------------

def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _seed_session(t0: int, frames: int = 3) -> dict:
    """A session with a forward camera (nominal calibration) and a frame every 50 ms."""
    import cv2

    from core.storage import get_object_store
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()

    async def go():
        sid = uuid.uuid4()
        out = {"session_id": sid, "frames": []}
        ok, jpg = cv2.imencode(".jpg", np.full((720, 1280, 3), 60, np.uint8))
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="RADAR-T", start_ts_ns=t0, end_ts_ns=t0 + 10**9,
                             ontology_version=get_ontology().version, sensors={}))
            await db.flush()
            for i in range(frames):
                ts = t0 + i * 50_000_000
                uri = store.put_bytes(f"frames/{sid}/front/{ts}.jpg", jpg.tobytes(), "image/jpeg")
                f = Frame(session_id=sid, ts_ns=ts, cam_id="front", img_uri=uri, width=1280, height=720, quality=1)
                db.add(f)
                await db.flush()
                out["frames"].append(f.frame_id)
            await db.commit()
        return out

    return _run(go())


@pytest.mark.db
def test_mcap_import_overlay_and_bev(tmp_path):
    from core.storage import get_object_store
    from services.sensors.radar import bev, frame_overlay, import_radar

    t0 = 1_700_000_000_000_000_000
    s = _seed_session(t0)
    p = tmp_path / "radar.mcap"
    _write_radar_mcap(p, t0)
    uri = get_object_store().put_bytes(f"uploads/test/{uuid.uuid4()}.mcap", p.read_bytes())
    # the front radar sits 3.5 m ahead of the ego origin, 0.6 m up
    ext = np.eye(4)
    ext[:3, 3] = [3.5, 0.0, 0.6]
    res = _run(import_radar({"session_id": str(s["session_id"]), "source_uri": uri, "format": "mcap",
                             "extrinsics": {"/radar/front/scan": ext.tolist()}}))
    assert res["imported"] == 6 and res["by_sensor"] == {"/radar/front/scan": 3, "/radar/left/points": 3}
    # re-importing replaces, never duplicates
    assert _run(import_radar({"session_id": str(s["session_id"]), "source_uri": uri, "format": "mcap",
                              "extrinsics": {"/radar/front/scan": ext.tolist()}}))["imported"] == 6

    async def check():
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            ov = await frame_overlay(db, s["frames"][1])
        async with get_sessionmaker()() as db:
            b = await bev(db, s["session_id"], t0 + 50_000_000)
        return ov, b

    ov, b = _run(check())
    front = next(x for x in ov["sensors"] if x["sensor"] == "/radar/front/scan")
    assert front["dt_ms"] == 0.0 and front["in_image"] == 1
    pt = front["points"][0]
    # 12 m straight ahead of a radar 3.5 m forward: 15.5 m from the ego, dead ahead, receding at 3 m/s
    assert pt["radial_mps"] == pytest.approx(3.0) and pt["range_m"] == pytest.approx(12.0, abs=1e-3)
    assert pt["u"] == pytest.approx(640, abs=1.0)
    assert pt["end"] is not None
    fb = next(x for x in b["sensors"] if x["sensor"] == "/radar/front/scan")
    assert fb["points"][0]["x"] == pytest.approx(15.5, abs=1e-3) and fb["origin"] == pytest.approx([3.5, 0, 0.6])


@pytest.mark.db
def test_import_refuses_sweeps_outside_the_session(tmp_path):
    from core.storage import get_object_store
    from services.sensors.radar import import_radar

    s = _seed_session(1_700_000_000_000_000_000)
    p = tmp_path / "late.mcap"
    _write_radar_mcap(p, 1_700_000_100_000_000_000)          # 100 s after the session
    uri = get_object_store().put_bytes(f"uploads/test/{uuid.uuid4()}.mcap", p.read_bytes())
    res = _run(import_radar({"session_id": str(s["session_id"]), "source_uri": uri, "format": "mcap"}))
    assert res["imported"] == 0 and res["outside_session"] == 6


@pytest.mark.db
def test_nuscenes_import_through_a_zip_upload(tmp_path):
    import shutil

    from core.storage import get_object_store
    from services.sensors.radar import import_radar

    t0_us = 1_600_000_000_000_000
    s = _seed_session(t0_us * 1000)
    root = _nuscenes_dir(tmp_path / "nusc", t0_us)
    z = shutil.make_archive(str(tmp_path / "nusc_radar"), "zip", root)
    uri = get_object_store().put_bytes(f"uploads/test/{uuid.uuid4()}.zip", Path(z).read_bytes())
    res = _run(import_radar({"session_id": str(s["session_id"]), "source_uri": uri, "format": "nuscenes"}))
    assert res["imported"] == 3 and res["by_sensor"] == {"RADAR_FRONT": 3} and res["points"] == 3


@pytest.mark.db
def test_fusion_writes_radar_speed_onto_a_moving_track():
    """A car driving away along +x at 8 m/s, boxed on five scans 100 ms apart; a compensated radar return
    inside each box reads 8.4 m/s radially. Fusion adopts the Doppler (within the gate) on every box."""
    from sqlalchemy import select

    from db.models import Object3D, PointCloud, RadarSweep, Track3D
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.ingest.radar import RadarScan
    from services.sensors.radar import fuse_session_tracks, store_scan

    t0 = 1_700_000_000_000_000_000
    s = _seed_session(t0, frames=1)
    sedan = get_ontology().by_name("sedan").id

    async def seed():
        async with get_sessionmaker()() as db:
            tr = Track3D(session_id=s["session_id"], class_id=sedan, first_ts_ns=t0, last_ts_ns=t0 + 4 * 10**8,
                         trajectory={"method": "test"})
            db.add(tr)
            await db.flush()
            for i in range(5):
                ts = t0 + i * 100_000_000
                pc = PointCloud(session_id=s["session_id"], ts_ns=ts, source="lidar", cloud_uri="s3://x/none.npz",
                                point_count=0)
                db.add(pc)
                await db.flush()
                cx = 20.0 + 0.8 * i
                db.add(Object3D(cloud_id=pc.cloud_id, track_3d_id=tr.track_3d_id, class_id=sedan,
                                center=[cx, 0.0, 0.0], dims=[4.4, 1.8, 1.5], yaw=0.0, conf=0.9,
                                box_source="native", source="human", state="review"))
                scan = RadarScan(xyz=np.array([[cx - 2.0, 0.2, 0.0]], np.float32),
                                 velocity=np.array([[8.4, 0.0, 0.0]], np.float32),
                                 radial=np.array([8.4], np.float32), rcs=np.array([10.0], np.float32),
                                 ts_ns=ts + 20_000_000, sensor="RADAR_FRONT", frame="ego", compensated=True)
                await store_scan(db, scan, s["session_id"], source="test", sensor_to_ego=np.eye(4))
            await db.commit()
            return tr.track_3d_id

    tid = _run(seed())
    dry = _run(fuse_session_tracks(s["session_id"], apply=False))
    assert dry["fused"] == 5 and dry["tracks"] == 1 and not dry["applied"]
    res = _run(fuse_session_tracks(s["session_id"], tol_mps=3.0))
    assert res["fused"] == 5 and res["rejected"] == 0
    assert res["per_track"][0]["median_speed_mps"] == pytest.approx(8.4, abs=1e-3)

    async def rows():
        async with get_sessionmaker()() as db:
            tr = await db.get(Track3D, tid)
            objs = (await db.execute(select(Object3D).where(Object3D.track_3d_id == tid))).scalars().all()
            n = len((await db.execute(select(RadarSweep).where(RadarSweep.session_id == s["session_id"]))).all())
            return tr, objs, n

    tr, objs, n = _run(rows())
    assert n == 5 and tr.trajectory["method"] == "test"          # the track's own keys are kept
    assert tr.trajectory["radar_speed"]["fused"] == 5
    assert all(o.attrs["radar"]["source"] == "radar" and o.attrs["radar"]["n_returns"] == 1 for o in objs)
    # A tight gate rejects the reading rather than averaging it in.
    strict = _run(fuse_session_tracks(s["session_id"], tol_mps=0.1, apply=False))
    assert strict["rejected"] == 5 and strict["fused"] == 0


@pytest.mark.db
def test_radar_routes_and_roles(tmp_path):
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from core.storage import get_object_store
    from services.api.main import app
    from services.tools.registry import run_job

    t0 = 1_700_000_000_000_000_000
    s = _seed_session(t0)
    p = tmp_path / "radar.mcap"
    _write_radar_mcap(p, t0)
    uri = get_object_store().put_bytes(f"uploads/test/{uuid.uuid4()}.mcap", p.read_bytes())
    sid = str(s["session_id"])
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        body = {"session_id": sid, "source_uri": uri, "format": "mcap"}
        assert c.post("/api/tools/sensors/radar/import", headers=ann, json=body).status_code == 403
        assert c.post("/api/tools/sensors/radar/import", headers=rev,
                      json={**body, "source_uri": "/etc/passwd"}).status_code == 422
        r = c.post("/api/tools/sensors/radar/import", headers=rev, json=body)
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
    _run(run_job(uuid.UUID(job_id)))
    with TestClient(app) as c:
        ann = auth_headers("annotator")
        j = c.get(f"/api/tools/jobs/{job_id}", headers=ann).json()
        assert j["status"] == "done" and j["result"]["imported"] == 6
        sw = c.get(f"/api/tools/sensors/radar/sessions/{sid}/sweeps", headers=ann).json()
        assert len(sw["sweeps"]) == 6 and sw["frames"][0]["radar_dt_ms"] == 0.0
        ov = c.get(f"/api/tools/sensors/radar/frames/{s['frames'][0]}/overlay", headers=ann).json()
        assert {x["sensor"] for x in ov["sensors"]} == {"/radar/front/scan", "/radar/left/points"}
        ovw = c.get("/api/tools/sensors/overview", headers=ann).json()
        assert any(x["session_id"] == sid and x["sweeps"] == 6 for x in ovw["radar"])
        assert c.post(f"/api/tools/sensors/radar/sessions/{sid}/fuse", headers=ann, json={}).status_code == 403
