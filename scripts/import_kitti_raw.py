"""Import a KITTI raw drive: synced camera frames, Velodyne scans, real calibration, real GPS.

This corpus has GNSS on 3 frames of 41,752 and no measured ego pose anywhere, which is why WP4 had to
recover a trajectory from pixels and WP7's occupancy grids could not be placed. KITTI raw carries the
three things that are missing together: a calibrated camera, a 64-beam Velodyne synchronised to it, and
an OXTS GPS/IMU solution per frame. Importing one drive gives every 3D surface in the product a session
where its inputs are real rather than inferred.

Licensing: KITTI raw is published by KIT and Toyota Technological Institute under CC BY-NC-SA 3.0. It is
usable for research and demonstration, which is what this is, and it is attributed on the session.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from pathlib import Path

import click

# Defaults, overridable per run. Kept as constants so the original single-drive invocation still works
# unchanged, while a second drive is one flag rather than an edit.
DRIVE = "2011_09_26_drive_0005_sync"
DATE = "2011_09_26"
# KITTI's colour left camera, the one its own benchmarks use.
CAM = "image_02"
CAM_ID = "cam_front"


# KITTI's Velodyne sits 1.73 m above the road (the cameras at 1.65 m). Clouds, cuboids and point labels of a
# KITTI session are in the Velodyne frame, so its road is at z = -1.73, not z = 0 as on the camera fleet.
VELO_HEIGHT_M = 1.73


def kitti_sensor_frame(imu_to_velo: dict | None) -> dict:
    """What a session's `sensors` must say for 3D consumers to place KITTI data: the ego frame is the
    Velodyne's, how high it is above the road, and the OXTS -> Velodyne lever arm for world placement."""
    out: dict = {"ego_frame": "velodyne", "lidar_height_m": VELO_HEIGHT_M}
    if imu_to_velo is not None:
        out["imu_to_velo"] = imu_to_velo
    return out


def _oxts_rows(drive_dir: Path) -> list[dict]:
    """The OXTS solution per frame: latitude, longitude, altitude, roll/pitch/yaw, and speed."""
    rows = []
    for f in sorted((drive_dir / "oxts" / "data").glob("*.txt")):
        v = [float(x) for x in f.read_text().split()]
        rows.append({"lat": v[0], "lon": v[1], "alt": v[2], "roll": v[3], "pitch": v[4], "yaw": v[5],
                     # vf/vl are forward and leftward velocity in the vehicle frame.
                     "speed": math.hypot(v[8], v[9])})
    return rows


def _timestamps(drive_dir: Path, sub: str) -> list[int]:
    """KITTI's per-frame capture times as UTC nanoseconds."""
    from datetime import datetime

    out = []
    for line in (drive_dir / sub / "timestamps.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # "2011-09-26 13:02:25.964389445" - nanosecond precision, which datetime cannot parse directly.
        head, _, frac = line.partition(".")
        base = datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
        out.append(int(base.timestamp()) * 1_000_000_000 + int((frac + "0" * 9)[:9]))
    return out


async def run(root: Path, vehicle: str, limit: int | None,
              drive: str = DRIVE, date: str = DATE) -> dict:
    from geoalchemy2 import WKTElement

    from core.origin import REAL
    from core.storage import get_object_store
    from db.models import CameraCalibration, EgoPose, Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.calib.kitti import kitti_camera_calibration, kitti_imu_to_velo
    from services.lidar.ingest.readers import read_kitti_bin
    from services.lidar.ingest.store import store_cloud

    drive_dir = root / date / drive
    cam_dir = drive_dir / CAM / "data"
    velo_dir = drive_dir / "velodyne_points" / "data"
    images = sorted(cam_dir.glob("*.png"))
    scans = sorted(velo_dir.glob("*.bin"))
    if limit:
        images, scans = images[:limit], scans[:limit]

    cam_ts = _timestamps(drive_dir, CAM)
    oxts = _oxts_rows(drive_dir)

    # Real intrinsics AND extrinsics from the drive's own calibration: the rectified projection P_rect_02 and
    # the Velodyne -> camera transform (calib_velo_to_cam, through R_rect_00), expressed in the Velodyne frame
    # the clouds are stored in. Projecting a cloud point through this row reproduces KITTI's own
    # P_rect_02 @ R_rect_00 @ Tr_velo_to_cam to well under a pixel (services/lidar/calib/kitti.py).
    calib = kitti_camera_calibration(root / date)
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    width, height = calib["ref_width"], calib["height"]
    imu_to_velo = kitti_imu_to_velo(root / date)

    import cv2

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    sid = uuid.uuid4()
    maker = get_sessionmaker()

    async with maker() as db:
        db.add(DbSession(
            session_id=sid, vehicle_id=vehicle, start_ts_ns=cam_ts[0], end_ts_ns=cam_ts[len(images) - 1],
            city="Karlsruhe", route=f"KITTI {drive}", ontology_version=onto.version, origin=REAL,
            sensors={"source": "KITTI raw", "drive": drive, "lidar": "Velodyne HDL-64E",
                     "camera": "PointGrey Flea2 (image_02, rectified)",
                     "license": "CC BY-NC-SA 3.0, KIT and Toyota Technological Institute",
                     "attribution": "Geiger et al., Vision meets Robotics: The KITTI Dataset",
                     **kitti_sensor_frame(imu_to_velo)}))
        await db.flush()
        # The drive's measured intrinsics, so every projection in the product uses real numbers here
        # rather than the nominal rig defaults.
        db.add(CameraCalibration(
            session_id=sid, cam_id=CAM_ID, model="pinhole", fx=fx, fy=fy, cx=cx, cy=cy, dist=[],
            ref_width=width, rpy_deg=calib["rpy_deg"], xyz_m=calib["xyz_m"],
            source="dataset", quality=0.95))
        await db.commit()

    n_frames = n_clouds = n_poses = 0
    lat0 = lon0 = None
    # Poses are collected and written after the frames are committed. SQLAlchemy batches inserts by
    # table within a flush, so adding an EgoPose alongside its Frame sends the pose first and the
    # foreign key fails on a frame that exists only in the same uncommitted unit of work.
    pending_poses: list[dict] = []
    async with maker() as db:
        for i, img_path in enumerate(images):
            ts = cam_ts[i]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            uri = store.put_bytes(f"frames/{sid}/{CAM_ID}/{ts}.jpg", enc.tobytes(), "image/jpeg")
            o = oxts[i] if i < len(oxts) else None
            fid = uuid.uuid4()
            db.add(Frame(
                frame_id=fid, session_id=sid, ts_ns=ts, cam_id=CAM_ID,
                width=img.shape[1], height=img.shape[0], img_uri=uri, origin=REAL, selected=True,
                gnss=(WKTElement(f"POINT({o['lon']} {o['lat']})", srid=4326) if o else None),
                ego_speed=(o["speed"] if o else None)))
            n_frames += 1

            if o is not None:
                if lat0 is None:
                    lat0, lon0 = o["lat"], o["lon"]
                # A session-local ENU frame, the same convention services/intelligence/ego_pose uses.
                east = math.radians(o["lon"] - lon0) * 6_371_000.0 * math.cos(math.radians(lat0))
                north = math.radians(o["lat"] - lat0) * 6_371_000.0
                half = o["yaw"] / 2.0
                pending_poses.append({
                    "session_id": sid, "ts_ns": ts, "frame_id": fid, "x": east, "y": north, "z": 0.0,
                    "qw": math.cos(half), "qx": 0.0, "qy": 0.0, "qz": math.sin(half),
                    "speed_mps": o["speed"], "yaw_rate": None,
                    # The one thing this corpus has never had: a pose an instrument observed.
                    "source": "gnss_imu", "quality": 0.95, "measured": True})
            if (i + 1) % 25 == 0:
                await db.commit()
        await db.commit()

    async with maker() as db:
        for row in pending_poses:
            db.add(EgoPose(**row))
            n_poses += 1
        await db.commit()

    for i, scan in enumerate(scans):
        cloud = read_kitti_bin(scan.read_bytes(), ts_ns=cam_ts[i])
        cloud.source = "lidar"
        await store_cloud(cloud, sid, source="lidar", calibration_version=f"kitti-{date}")
        n_clouds += 1

    return {"session_id": str(sid), "frames": n_frames, "clouds": n_clouds, "poses": n_poses,
            "calibration": {"fx": round(fx, 2), "fy": round(fy, 2), "cx": round(cx, 2),
                            "cy": round(cy, 2), "width": width, "height": height,
                            "rpy_deg": calib["rpy_deg"], "xyz_m": calib["xyz_m"]}}


async def fix_calibration_rows(root: Path, apply: bool = True) -> list[dict]:
    """Correct the camera_calibration rows of KITTI sessions imported before the extrinsics were read.

    Idempotent: each KITTI raw session (route "KITTI <drive>") whose date folder holds the calibration files
    gets the exact rpy/xyz/intrinsics; a row already equal is reported unchanged. The session's `sensors` gains
    the Velodyne frame note and the IMU lever arm. Sessions whose calibration is not on disk are reported and
    left alone. Nothing else is touched: frames, clouds, poses and labels stay as they are.
    """
    from sqlalchemy import select

    from db.models import CameraCalibration
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.lidar.calib.kitti import kitti_camera_calibration, kitti_imu_to_velo

    out: list[dict] = []
    async with get_sessionmaker()() as db:
        sessions = (await db.execute(select(DbSession).where(DbSession.route.like("KITTI 20%")))).scalars().all()
        for sess in sessions:
            drive = (sess.sensors or {}).get("drive") or sess.route.split(" ", 1)[1]
            date = drive[:10]
            date_dir = root / date
            rec = {"session_id": str(sess.session_id), "drive": drive}
            if not (date_dir / "calib_velo_to_cam.txt").exists() or not (date_dir / "calib_cam_to_cam.txt").exists():
                rec["status"] = f"calibration not on disk under {date_dir}"
                out.append(rec)
                continue
            calib = kitti_camera_calibration(date_dir)
            row = (await db.execute(select(CameraCalibration).where(
                CameraCalibration.session_id == sess.session_id,
                CameraCalibration.cam_id == CAM_ID))).scalar_one_or_none()
            want = {"fx": calib["fx"], "fy": calib["fy"], "cx": calib["cx"], "cy": calib["cy"],
                    "ref_width": calib["ref_width"], "rpy_deg": calib["rpy_deg"], "xyz_m": calib["xyz_m"]}
            sensors = dict(sess.sensors or {})
            frame_note = kitti_sensor_frame(kitti_imu_to_velo(date_dir))
            if row is None:
                rec["status"] = "created"
                if apply:
                    db.add(CameraCalibration(session_id=sess.session_id, cam_id=CAM_ID, model="pinhole", dist=[],
                                             source="dataset", quality=0.95, **want))
            else:
                before = {"rpy_deg": list(row.rpy_deg or []), "xyz_m": list(row.xyz_m or [])}
                same = all(
                    (getattr(row, k) == v) if not isinstance(v, list)
                    else [round(float(x), 6) for x in (getattr(row, k) or [])] == v
                    for k, v in want.items())
                rec["before"] = before
                rec["status"] = "unchanged" if same else "corrected"
                if apply and not same:
                    for k, v in want.items():
                        setattr(row, k, v)
                    row.source, row.quality = "dataset", 0.95
            if any(sensors.get(k) != v for k, v in frame_note.items()):
                rec["sensors"] = "updated"
                if apply:
                    sess.sensors = {**sensors, **frame_note}
            rec["after"] = {"rpy_deg": want["rpy_deg"], "xyz_m": want["xyz_m"]}
            out.append(rec)
        if apply:
            await db.commit()
    return out


@click.command()
@click.option("--root", default=".scratch/demo/kitti", type=click.Path(exists=True))
@click.option("--vehicle", default="KITTI-0005")
@click.option("--limit", type=int, default=None)
@click.option("--drive", default=DRIVE, help="drive folder name, e.g. 2011_09_26_drive_0027_sync")
@click.option("--date", "date_", default=DATE, help="capture date folder holding the drive and its calibration")
@click.option("--fix-calibration", is_flag=True,
              help="import nothing: correct the camera calibration of already-imported KITTI sessions")
@click.option("--dry-run", is_flag=True, help="with --fix-calibration, report without writing")
def main(root: str, vehicle: str, limit: int | None, drive: str, date_: str, fix_calibration: bool,
         dry_run: bool) -> None:
    from core.logging import setup_logging

    setup_logging("INFO")
    if fix_calibration:
        for rec in asyncio.run(fix_calibration_rows(Path(root), apply=not dry_run)):
            print(f"  {rec}")
        return
    res = asyncio.run(run(Path(root), vehicle, limit, drive=drive, date=date_))
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
