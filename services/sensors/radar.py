"""Radar in the product: import sweeps, show them on the camera image and from above, fuse them into Track3D.

Formats are read by services/lidar/ingest/radar.py. This module owns what happens around them:

- import: every sweep becomes a `radar_sweep` row and an npz in the object store (`radar/<session>/<sensor>/
  <ts>.npz`), in the session's ego frame. A sweep outside the session's time span is not the session's and is
  skipped, which is also what picks one nuScenes scene out of a directory holding several.
- overlay: the sweeps nearest a camera frame, projected with that session's resolved calibration (the same
  `project_to_camera` the LiDAR overlay uses), with each return's velocity drawn as where it will be in 0.5 s.
- fusion: the Doppler of the returns inside each Object3D's box (grown by a margin) replaces the radial part of
  the track's velocity, under a consistency gate (services/oraclyx/radar.py `fuse_track3d`).

There is no radar data on this machine. Every path here is exercised by tests on fixtures written in the
published formats (nuScenes .pcd and tables, MCAP with ROS 2 RadarScan and PointCloud2), and nothing else.
"""

from __future__ import annotations

import asyncio
import math
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from services.lidar.ingest.radar import RadarScan

log = get_logger("sensors.radar")

ProgressFn = Callable[[float, str], Awaitable[None]]
# A sweep this far from a camera frame or a LiDAR scan still belongs to that instant (radars run at 13-20 Hz).
MATCH_NS = 75_000_000
SPAN_SLACK_NS = 1_000_000_000
ARROW_S = 0.5
BOX_MARGIN_M = 1.0


def _key(session_id, sensor: str, ts_ns: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sensor.strip("/")) or "radar"
    return f"radar/{session_id}/{safe}/{int(ts_ns)}.npz"


def scan_stats(scan: RadarScan) -> dict:
    sp = np.linalg.norm(scan.velocity[:, :2], axis=1) if scan.n else np.zeros(0)
    rng = np.linalg.norm(scan.xyz[:, :2], axis=1) if scan.n else np.zeros(0)
    return {"n_total": int(scan.n_total), "compensated": bool(scan.compensated), "vector": bool(scan.vector),
            "moving": int((sp > 0.5).sum()), "max_range_m": round(float(rng.max()), 2) if scan.n else None,
            "max_speed_mps": round(float(sp.max()), 2) if scan.n else None}


async def store_scan(db: AsyncSession, scan: RadarScan, session_id: uuid.UUID, *, source: str,
                     sensor_to_ego: np.ndarray | None, sensor_origin: np.ndarray | None = None) -> uuid.UUID:
    """Write one sweep (already in the ego frame) and its row. Re-importing the same sensor and instant replaces
    the row rather than duplicating it."""
    from db.models import RadarSweep

    store = get_object_store()
    origin = sensor_origin if sensor_origin is not None else (
        np.asarray(sensor_to_ego)[:3, 3] if sensor_to_ego is not None else np.zeros(3))
    uri = await asyncio.to_thread(store.put_bytes, _key(session_id, scan.sensor, scan.ts_ns),
                                  scan.to_npz_bytes(sensor_origin=origin))
    await db.execute(delete(RadarSweep).where(RadarSweep.session_id == session_id,
                                              RadarSweep.sensor == scan.sensor, RadarSweep.ts_ns == scan.ts_ns))
    row = RadarSweep(session_id=session_id, ts_ns=int(scan.ts_ns), sensor=scan.sensor[:48], source=source,
                     uri=uri, n_points=scan.n, has_velocity=True,
                     sensor_to_ego=np.asarray(sensor_to_ego).tolist() if sensor_to_ego is not None else None,
                     stats=scan_stats(scan))
    db.add(row)
    await db.flush()
    return row.sweep_id


def _matrix(value, default: np.ndarray | None = None) -> np.ndarray | None:
    if value is None:
        return default
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all() or abs(m[3, 3] - 1.0) > 1e-6:
        raise ValueError("an extrinsic must be a 4x4 homogeneous matrix (sensor -> ego)")
    return m


async def import_radar(params: dict, *, progress: ProgressFn | None = None) -> dict:
    """Import radar for one session. params: session_id, source_uri (s3:// upload, a .zip, or an allowed local
    path), format (nuscenes | mcap | pcd), plus per format: channels / keyframes_only / filter_points
    (nuscenes), topics / extrinsics {topic: 4x4} / time_source (mcap), sensor / ts_ns / sensor_to_ego (pcd)."""
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.imports.run import _acquire_source
    from services.lidar.ingest import radar as rf

    session_id = uuid.UUID(str(params["session_id"]))
    fmt = str(params.get("format", "nuscenes"))
    maker = get_sessionmaker()
    async with maker() as db:
        sess = await db.get(DbSession, session_id)
        if sess is None:
            raise LookupError("session not found")
        lo, hi = int(sess.start_ts_ns) - SPAN_SLACK_NS, int(sess.end_ts_ns) + SPAN_SLACK_NS
    workdir = Path(tempfile.mkdtemp(prefix="radar_import_"))
    counts: dict[str, Any] = {"imported": 0, "outside_session": 0, "unreadable": 0, "points": 0, "by_sensor": {}}
    try:
        local = await asyncio.to_thread(_acquire_source, str(params["source_uri"]), workdir)
        scans: list[tuple[RadarScan, np.ndarray | None, str]] = []
        if fmt == "nuscenes":
            sweeps = await asyncio.to_thread(
                rf.list_nuscenes_radar_sweeps, local, tuple(params.get("channels") or rf.RADAR_CHANNELS),
                bool(params.get("keyframes_only", False)))
            if not sweeps:
                raise ValueError("no radar sample_data rows in that nuScenes directory")
            for i, sw in enumerate(sweeps):
                ts = sw.sample_ts_ns if (sw.is_key_frame and sw.sample_ts_ns is not None) else sw.ts_ns
                if not lo <= ts <= hi:
                    counts["outside_session"] += 1
                    continue
                try:
                    scan = await asyncio.to_thread(rf.read_nuscenes_radar_sweep, sw,
                                                   filter_points=bool(params.get("filter_points", True)))
                except (OSError, rf.RadarFormatError) as exc:
                    log.warning("radar.unreadable", path=str(sw.path), error=str(exc))
                    counts["unreadable"] += 1
                    continue
                scans.append((scan, sw.sensor_to_ego, "nuscenes"))
                if progress is not None and i % 50 == 0:
                    await progress(0.1 + 0.5 * (i + 1) / len(sweeps), f"reading sweep {i + 1} of {len(sweeps)}")
        elif fmt == "mcap":
            path: Path | None = local if local.is_file() else next(iter(sorted(local.rglob("*.mcap"))), None)
            if path is None:
                raise ValueError("no .mcap file in the upload")
            ext = {k: _matrix(v) for k, v in (params.get("extrinsics") or {}).items()}
            default = _matrix(params.get("sensor_to_ego"), np.eye(4))
            topics = params.get("topics") or None
            msgs = await asyncio.to_thread(lambda: list(rf.iter_radar_mcap(
                path, topics, str(params.get("time_source", "log")))))
            if not msgs:
                raise ValueError("the MCAP has no RadarScan channel and no PointCloud2 channel with velocity fields")
            for topic, scan, kind in msgs:
                if not lo <= scan.ts_ns <= hi:
                    counts["outside_session"] += 1
                    continue
                T = ext.get(topic) if ext.get(topic) is not None else default
                T = T if T is not None else np.eye(4)
                scans.append((scan.transformed(T, "ego"), T, kind))
        elif fmt == "pcd":
            path = local if local.is_file() else next(iter(sorted(local.rglob("*.pcd"))), None)  # type: ignore[arg-type]
            if path is None:
                raise ValueError("no .pcd file in the upload")
            T = _matrix(params.get("sensor_to_ego"), np.eye(4))
            T = T if T is not None else np.eye(4)
            ts = int(params.get("ts_ns") or 0)
            scan = rf.read_nuscenes_radar_pcd(path.read_bytes(), ts_ns=ts, sensor=str(params.get("sensor") or
                                                                                  "RADAR_FRONT"),
                                              filter_points=bool(params.get("filter_points", True)))
            if not lo <= ts <= hi:
                raise ValueError("ts_ns is outside the session's time span")
            scans.append((scan.transformed(T, "ego"), T, "pcd"))
        else:
            raise ValueError(f"unknown radar format {fmt!r} (nuscenes, mcap, pcd)")

        async with maker() as db:
            for i, (scan, T, src) in enumerate(scans):
                await store_scan(db, scan, session_id, source=src, sensor_to_ego=T)
                counts["imported"] += 1
                counts["points"] += scan.n
                counts["by_sensor"][scan.sensor] = counts["by_sensor"].get(scan.sensor, 0) + 1
                if i % 100 == 99:
                    await db.commit()
                    if progress is not None:
                        await progress(0.6 + 0.4 * (i + 1) / len(scans), f"stored {i + 1} sweeps")
            await db.commit()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    log.info("radar.imported", session=str(session_id), **{k: v for k, v in counts.items() if k != "by_sensor"})
    return {"session_id": str(session_id), "format": fmt, **counts}


def load_scan(uri: str) -> tuple[RadarScan, np.ndarray]:
    """A stored sweep and the radar's position in the ego frame."""
    import io

    data = get_object_store().get_bytes(uri)
    z = np.load(io.BytesIO(data))
    origin = z["sensor_origin"] if "sensor_origin" in z.files else np.zeros(3, np.float32)
    return RadarScan.from_npz_bytes(data, frame="ego"), origin


async def sweeps_near(db: AsyncSession, session_id: uuid.UUID, ts_ns: int, tol_ns: int = MATCH_NS) -> list:
    """The sweep of each sensor nearest `ts_ns`, within the tolerance."""
    from db.models import RadarSweep

    rows = (await db.execute(select(RadarSweep).where(
        RadarSweep.session_id == session_id, RadarSweep.ts_ns.between(int(ts_ns) - tol_ns, int(ts_ns) + tol_ns))
    )).scalars().all()
    best: dict[str, object] = {}
    for r in rows:
        cur = best.get(r.sensor)
        if cur is None or abs(r.ts_ns - ts_ns) < abs(cur.ts_ns - ts_ns):
            best[r.sensor] = r
    return sorted(best.values(), key=lambda r: r.sensor)


def overlay_points(scan: RadarScan, calib, width: int, height: int, origin: np.ndarray) -> list[dict]:
    """Project one ego-frame sweep into an image. Each return: pixel, depth, radial speed, ground speed, RCS and
    the pixel its velocity reaches in ARROW_S seconds (null when that end leaves the camera's front)."""
    from services.lidar.project import project_to_camera

    if scan.n == 0:
        return []
    p0 = project_to_camera(scan.xyz.astype(np.float32), calib.cam_id, width, height, calib=calib)
    p1 = project_to_camera((scan.xyz + scan.velocity * ARROW_S).astype(np.float32), calib.cam_id, width, height,
                           calib=calib)
    rel = scan.xyz.astype(np.float64) - np.asarray(origin, np.float64)
    out = []
    for i in np.nonzero(p0["in_image"])[0]:
        end = p1["uv"][i].tolist() if p1["in_front"][i] else None
        out.append({"u": round(float(p0["uv"][i, 0]), 1), "v": round(float(p0["uv"][i, 1]), 1),
                    "depth_m": round(float(p0["depth"][i]), 2), "range_m": round(float(np.linalg.norm(rel[i])), 2),
                    "radial_mps": round(float(scan.radial[i]), 2),
                    "speed_mps": round(float(np.linalg.norm(scan.velocity[i, :2])), 2),
                    "rcs": round(float(scan.rcs[i]), 1),
                    "end": [round(float(end[0]), 1), round(float(end[1]), 1)] if end else None})
    return out


async def frame_overlay(db: AsyncSession, frame_id: uuid.UUID) -> dict:
    """The radar returns at a camera frame's instant, projected into it."""
    from db.models import Frame
    from services.calibration.resolve import resolve_calibration

    fr = await db.get(Frame, frame_id)
    if fr is None:
        raise LookupError("frame not found")
    rows = await sweeps_near(db, fr.session_id, fr.ts_ns)
    sid, cam, w, h, ts = fr.session_id, fr.cam_id, int(fr.width), int(fr.height), int(fr.ts_ns)
    await db.commit()
    cal = await resolve_calibration(sid, cam, w, h)
    sensors = []
    for r in rows:
        scan, origin = await asyncio.to_thread(load_scan, r.uri)
        pts = overlay_points(scan, cal, w, h, origin)
        sensors.append({"sweep_id": str(r.sweep_id), "sensor": r.sensor, "ts_ns": r.ts_ns, "dt_ms":
                        round((r.ts_ns - ts) / 1e6, 1), "n_points": r.n_points, "in_image": len(pts),
                        "points": pts})
    return {"frame_id": str(frame_id), "cam_id": cam, "width": w, "height": h, "calibration": cal.source,
            "arrow_s": ARROW_S, "sensors": sensors}


async def bev(db: AsyncSession, session_id: uuid.UUID, ts_ns: int) -> dict:
    """Every sensor's sweep near `ts_ns` in the ego frame, plus the Object3D boxes of the nearest scan."""
    from db.models import Object3D, PointCloud

    rows = await sweeps_near(db, session_id, ts_ns)
    sensors = []
    for r in rows:
        scan, origin = await asyncio.to_thread(load_scan, r.uri)
        sensors.append({"sweep_id": str(r.sweep_id), "sensor": r.sensor, "ts_ns": r.ts_ns,
                        "origin": [round(float(v), 3) for v in origin],
                        "points": [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "z": round(float(p[2]), 2),
                                    "vx": round(float(v[0]), 2), "vy": round(float(v[1]), 2),
                                    "radial_mps": round(float(rr), 2), "rcs": round(float(c), 1)}
                                   for p, v, rr, c in zip(scan.xyz, scan.velocity, scan.radial, scan.rcs,
                                                          strict=True)]})
    cloud = (await db.execute(select(PointCloud.cloud_id, PointCloud.ts_ns).where(
        PointCloud.session_id == session_id, PointCloud.ts_ns.between(ts_ns - MATCH_NS, ts_ns + MATCH_NS))
        .order_by(PointCloud.ts_ns))).first()
    boxes = []
    if cloud is not None:
        objs = (await db.execute(select(Object3D).where(Object3D.cloud_id == cloud.cloud_id))).scalars().all()
        boxes = [{"object_3d_id": str(o.object_3d_id), "center": list(o.center), "dims": list(o.dims),
                  "yaw": o.yaw, "class_id": o.class_id, "state": o.state,
                  "radar": (o.attrs or {}).get("radar")} for o in objs]
    return {"session_id": str(session_id), "ts_ns": int(ts_ns), "sensors": sensors, "boxes": boxes}


def returns_in_box(xyz: np.ndarray, center, dims, yaw: float, margin: float = BOX_MARGIN_M) -> np.ndarray:
    """Mask of ego-frame points inside an oriented box (dims = L, W, H) grown by `margin` on every side."""
    c = np.asarray(center, dtype=np.float64)
    d = np.asarray(xyz, dtype=np.float64)[:, :3] - c
    cy, sy = math.cos(yaw), math.sin(yaw)
    lx = d[:, 0] * cy + d[:, 1] * sy
    ly = -d[:, 0] * sy + d[:, 1] * cy
    L, W, H = (float(v) for v in dims)
    return (np.abs(lx) <= L / 2 + margin) & (np.abs(ly) <= W / 2 + margin) & (np.abs(d[:, 2]) <= H / 2 + margin)


def track_velocities(centers_world: np.ndarray, ts_ns: np.ndarray) -> np.ndarray:
    """Velocity over ground at each point of a track by central differences (one-sided at the ends)."""
    n = len(centers_world)
    v = np.zeros((n, 3))
    if n < 2:
        return v
    t = ts_ns.astype(np.float64) / 1e9
    for i in range(n):
        a, b = max(0, i - 1), min(n - 1, i + 1)
        dt = t[b] - t[a]
        if dt > 1e-6:
            v[i] = (centers_world[b] - centers_world[a]) / dt
    return v


async def fuse_session_tracks(session_id: uuid.UUID, *, tol_mps: float = 3.0, apply: bool = True,
                              progress: ProgressFn | None = None) -> dict:
    """Doppler onto every Track3D of the session. Writes Object3D.attrs['radar'] per box and
    Track3D.trajectory['radar_speed'] per track when `apply`; with apply=False it only reports."""
    from db.models import EgoPose, Object3D, PointCloud, RadarSweep, Track3D
    from db.session import get_sessionmaker
    from services.lidar.aggregate.scene_frame import session_scene_poses
    from services.oraclyx.radar import fuse_track3d

    maker = get_sessionmaker()
    async with maker() as db:
        n_sweeps = len((await db.execute(select(RadarSweep.sweep_id).where(
            RadarSweep.session_id == session_id))).all())
        if not n_sweeps:
            return {"session_id": str(session_id), "tracks": 0, "reason": "this session has no radar sweeps"}
        tracks = (await db.execute(select(Track3D).where(Track3D.session_id == session_id))).scalars().all()
        poses, _ = await session_scene_poses(db, session_id)
        world = {sp.cloud_id: sp.T_world_ego for sp in poses}
        ego = (await db.execute(select(EgoPose.ts_ns, EgoPose.speed_mps).where(EgoPose.session_id == session_id)
                                .order_by(EgoPose.ts_ns))).all()
    ego_ts = np.array([e[0] for e in ego], dtype=np.int64)
    ego_sp = [e[1] for e in ego]
    cache: dict[str, tuple[RadarScan, np.ndarray]] = {}
    summary = {"session_id": str(session_id), "tracks": 0, "boxes": 0, "fused": 0, "rejected": 0, "no_radar": 0,
               "sweeps": n_sweeps, "applied": apply, "per_track": []}
    for k, tr in enumerate(tracks):
        async with maker() as db:
            rows = (await db.execute(select(Object3D, PointCloud.ts_ns).join(
                PointCloud, PointCloud.cloud_id == Object3D.cloud_id).where(
                Object3D.track_3d_id == tr.track_3d_id).order_by(PointCloud.ts_ns))).all()
            if not rows:
                continue
            objs = [r[0] for r in rows]
            ts = np.array([r[1] for r in rows], dtype=np.int64)
            Ts = [world.get(o.cloud_id) for o in objs]
            if any(T is None for T in Ts):
                # Without a pose for every scan the track has no world motion to compare against.
                Ts = [np.eye(4)] * len(objs)
            cw = np.array([T[:3, :3] @ np.asarray(o.center, np.float64) + T[:3, 3]
                           for o, T in zip(objs, Ts, strict=True)])
            vw = track_velocities(cw, ts)
            pts = []
            for o, T, t, v_world in zip(objs, Ts, ts, vw, strict=True):
                v_ego = T[:3, :3].T @ v_world
                near = await sweeps_near(db, session_id, int(t))
                radials, n_ret, los_sum = [], 0, np.zeros(3)
                for sw in near:
                    if sw.uri not in cache:
                        cache[sw.uri] = await asyncio.to_thread(load_scan, sw.uri)
                    scan, origin = cache[sw.uri]
                    m = returns_in_box(scan.xyz, o.center, o.dims, o.yaw)
                    if not m.any():
                        continue
                    rad = scan.radial[m].astype(np.float64)
                    if not scan.compensated:
                        # A raw Doppler is relative to the moving radar: add the ego's own speed along each line
                        # of sight to get the object's speed over ground (ego velocity is +x in the ego frame).
                        j = int(np.argmin(np.abs(ego_ts - int(t)))) if len(ego_ts) else -1
                        ego_v = float(ego_sp[j] or 0.0) if j >= 0 else 0.0
                        los = scan.xyz[m].astype(np.float64) - origin
                        los /= np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-6)
                        rad = rad + ego_v * los[:, 0]
                    radials.extend(rad.tolist())
                    n_ret += int(m.sum())
                    los_sum += np.asarray(o.center, np.float64) - origin
                pts.append({"ts_ns": int(t), "velocity": v_ego[:2].tolist(),
                            "los": (los_sum[:2] if n_ret else np.asarray(o.center, np.float64)[:2]).tolist(),
                            "radial": float(np.median(radials)) if radials else None, "n_returns": n_ret})
            res = fuse_track3d(pts, tol_mps=tol_mps)
            if apply:
                for o, rec in zip(objs, res["points"], strict=True):
                    o.attrs = {**(o.attrs or {}), "radar": {k: rec[k] for k in (
                        "source", "speed_mps", "residual_mps", "n_returns") if k in rec}}
                tr.trajectory = {**(tr.trajectory or {}), "radar_speed": {
                    "median_speed_mps": res["median_speed_mps"], "fused": res["fused"], "rejected": res["rejected"],
                    "no_radar": res["no_radar"], "tol_mps": tol_mps, "source": res["source"]}}
                db.add(tr)
                await db.commit()
        summary["tracks"] += 1
        summary["boxes"] += len(objs)
        for key in ("fused", "rejected", "no_radar"):
            summary[key] += res[key]
        summary["per_track"].append({"track_3d_id": str(tr.track_3d_id), "boxes": len(objs),
                                     "median_speed_mps": res["median_speed_mps"], "fused": res["fused"],
                                     "rejected": res["rejected"]})
        if progress is not None and k % 10 == 0:
            await progress((k + 1) / max(1, len(tracks)), f"track {k + 1} of {len(tracks)}")
    summary["per_track"] = summary["per_track"][:200]
    return summary
