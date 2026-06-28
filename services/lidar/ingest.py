"""Ingest a LiDAR sweep as a BEV frame: store the raw point cloud, rasterize a bird's-eye image, and
create a Session/Frame the editor renders. Oriented boxes drawn on the image lift to 3D cuboids using the
stored BEV params and the point cloud (services.lidar.cuboids)."""

from __future__ import annotations

import uuid

import cv2
import numpy as np

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.lidar.bev import default_bev_params, rasterize_bev

log = get_logger("lidar_ingest")


def read_kitti_bin(data: bytes) -> np.ndarray:
    """KITTI Velodyne .bin -> N x 4 float32 (x, y, z, intensity)."""
    return np.frombuffer(data, dtype=np.float32).reshape(-1, 4)


async def ingest_lidar_sweep(points: np.ndarray, raw_bytes: bytes, name: str,
                             vehicle: str = "LIDAR-KITTI", city: str = "BLR",
                             session_id: uuid.UUID | None = None, ts_ns: int = 0,
                             bev_params: dict | None = None) -> dict:
    """Store one sweep (raw cloud + BEV image) and create its Frame. Reuses an existing session if given."""
    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    p = bev_params or default_bev_params()
    ok, buf = cv2.imencode(".png", rasterize_bev(points, p))
    if not ok:
        raise RuntimeError("failed to encode BEV image")

    maker = get_sessionmaker()
    new_session = session_id is None
    sid = session_id or uuid.uuid4()
    async with maker() as db:
        if new_session:
            db.add(DbSession(session_id=sid, vehicle_id=vehicle, start_ts_ns=0, end_ts_ns=1, city=city,
                             sensors={"lidar": "velodyne"}, ontology_version=onto.version))
            await db.flush()
        pcd_uri = store.put_bytes(f"lidar/{sid}/{name}.bin", raw_bytes, "application/octet-stream")
        img_uri = store.put_bytes(f"frames/{sid}/lidar_bev/{name}.png", buf.tobytes(), "image/png")
        f = Frame(session_id=sid, ts_ns=ts_ns, cam_id="lidar_bev", img_uri=img_uri,
                  width=p["width"], height=p["height"], quality=1.0,
                  lidar={"pcd_uri": pcd_uri, "frame": "kitti_velo", "n_points": int(points.shape[0]), "bev": p})
        db.add(f)
        await db.flush()
        fid = f.frame_id
        await db.commit()
    log.info("lidar.ingested", session=str(sid), frame=str(fid), points=int(points.shape[0]))
    return {"session_id": str(sid), "frame_id": str(fid), "n_points": int(points.shape[0]), "bev": p}
