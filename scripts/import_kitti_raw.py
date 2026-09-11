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
import numpy as np

DRIVE = "2011_09_26_drive_0005_sync"
DATE = "2011_09_26"
# KITTI's colour left camera, the one its own benchmarks use.
CAM = "image_02"
CAM_ID = "cam_front"


def _read_calib(path: Path) -> dict:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = np.array([float(x) for x in v.split()])
        except ValueError:
            continue
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


async def run(root: Path, vehicle: str, limit: int | None) -> dict:
    from geoalchemy2 import WKTElement

    from core.origin import REAL
    from core.storage import get_object_store
    from db.models import CameraCalibration, EgoPose, Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.ingest.readers import read_kitti_bin
    from services.lidar.ingest.store import store_cloud

    drive_dir = root / DATE / DRIVE
    cam_dir = drive_dir / CAM / "data"
    velo_dir = drive_dir / "velodyne_points" / "data"
    images = sorted(cam_dir.glob("*.png"))
    scans = sorted(velo_dir.glob("*.bin"))
    if limit:
        images, scans = images[:limit], scans[:limit]

    cam_ts = _timestamps(drive_dir, CAM)
    oxts = _oxts_rows(drive_dir)

    # Real intrinsics from the drive's own calibration, and the rectified projection KITTI publishes.
    c2c = _read_calib(root / DATE / "calib_cam_to_cam.txt")
    P2 = c2c["P_rect_02"].reshape(3, 4)
    fx, fy, cx, cy = float(P2[0, 0]), float(P2[1, 1]), float(P2[0, 2]), float(P2[1, 2])
    size = c2c.get("S_rect_02")
    width, height = (int(size[0]), int(size[1])) if size is not None else (1242, 375)

    import cv2

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    sid = uuid.uuid4()
    maker = get_sessionmaker()

    async with maker() as db:
        db.add(DbSession(
            session_id=sid, vehicle_id=vehicle, start_ts_ns=cam_ts[0], end_ts_ns=cam_ts[len(images) - 1],
            city="Karlsruhe", route=f"KITTI {DRIVE}", ontology_version=onto.version, origin=REAL,
            sensors={"source": "KITTI raw", "drive": DRIVE, "lidar": "Velodyne HDL-64E",
                     "camera": "PointGrey Flea2 (image_02, rectified)",
                     "license": "CC BY-NC-SA 3.0, KIT and Toyota Technological Institute",
                     "attribution": "Geiger et al., Vision meets Robotics: The KITTI Dataset"}))
        await db.flush()
        # The drive's measured intrinsics, so every projection in the product uses real numbers here
        # rather than the nominal rig defaults.
        db.add(CameraCalibration(
            session_id=sid, cam_id=CAM_ID, model="pinhole", fx=fx, fy=fy, cx=cx, cy=cy, dist=[],
            ref_width=width, rpy_deg=[0.0, 0.0, 0.0], xyz_m=[1.03, 0.0, 1.65],
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
        await store_cloud(cloud, sid, source="lidar", calibration_version=f"kitti-{DATE}")
        n_clouds += 1

    return {"session_id": str(sid), "frames": n_frames, "clouds": n_clouds, "poses": n_poses,
            "calibration": {"fx": round(fx, 2), "fy": round(fy, 2), "cx": round(cx, 2),
                            "cy": round(cy, 2), "width": width, "height": height}}


@click.command()
@click.option("--root", default=".scratch/demo/kitti", type=click.Path(exists=True))
@click.option("--vehicle", default="KITTI-0005")
@click.option("--limit", type=int, default=None)
def main(root: str, vehicle: str, limit: int | None) -> None:
    from core.logging import setup_logging

    setup_logging("INFO")
    res = asyncio.run(run(Path(root), vehicle, limit))
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
