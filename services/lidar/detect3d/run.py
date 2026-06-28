"""Orchestrate 3D detection on a frame or cloud: load the synchronized cloud and ground plane, lift each 2D
object to a ground-snapped cuboid (the primary path) or run native detection on real LiDAR, gate every
proposal through the same governed gate, and persist object_3d linked to the 2D object by object_id (the
unifying identity). Idempotent: machine 3D objects are cleared and rewritten on re-run; human 3D objects
survive. Raw clouds and frames are never mutated.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select

from core.logging import get_logger
from db.models import Frame, Object, Object3D, PointCloud
from db.session import get_sessionmaker
from services.lidar.clean.ground import segment_ground
from services.lidar.detect3d.fuse3d import gate_cuboid
from services.lidar.detect3d.lift import lift_box
from services.lidar.detect3d.native import detect_native
from services.lidar.ingest.store import load_cloud

log = get_logger("lidar_detect3d")
LIFT_VERSION = "lift-3d-0.1"
CALIB_VERSION = "labelox-calib-0.1"   # the calibration that placed the cloud (Phase 1), recorded in provenance


async def _auto_accept_enabled(db) -> bool:
    try:
        from services.govern.killswitch import get_state
        return (await get_state(db)).auto_accept_enabled
    except Exception:
        return True


async def _cloud_for_frame(db, frame: Frame) -> PointCloud | None:
    return (await db.execute(
        select(PointCloud).where(PointCloud.session_id == frame.session_id, PointCloud.ts_ns == frame.ts_ns)
        .order_by(PointCloud.created_at.desc()).limit(1))).scalar_one_or_none()


def _row(cloud_id, frame_id, object_id, g: dict, cuboid: dict) -> Object3D:
    return Object3D(
        cloud_id=cloud_id, frame_id=frame_id, object_id=object_id, class_id=g["class_id"],
        center=cuboid["center"], dims=cuboid["dims"], yaw=cuboid["yaw"], pitch=cuboid["pitch"],
        roll=cuboid["roll"], conf=g["conf"], box_source=g["box_source"], source="fused", state=g["state"],
        attrs={"fill": cuboid.get("fill"), "n_points": cuboid.get("n_points"),
               "ground_z": cuboid.get("ground_z"), "is_rare": g["is_rare"], "is_fallback": g["is_fallback"]},
        provenance=g["provenance"])


async def lift_frame(frame_id: uuid.UUID) -> dict:
    """Lift every 2D object on a frame into a ground-snapped 3D cuboid, gate it, and persist object_3d linked
    to the 2D object. The primary path for the pseudo-LiDAR fleet."""
    async with get_sessionmaker()() as db:
        frame = await db.get(Frame, frame_id)
        if frame is None:
            return {"error": "frame not found"}
        pc = await _cloud_for_frame(db, frame)
        if pc is None:
            return {"frame_id": str(frame_id), "cuboids": 0, "reason": "no synchronized cloud at this ts_ns"}
        objs = (await db.execute(select(Object).where(Object.frame_id == frame_id))).scalars().all()
        aae = await _auto_accept_enabled(db)
        cloud_id, cloud_uri = pc.cloud_id, pc.cloud_uri
        w, h, cam = frame.width or 1280, frame.height or 960, frame.cam_id

    cloud = load_cloud(cloud_uri)
    _, plane, _ = segment_ground(cloud)
    proposals = []
    for o in objs:
        cuboid = lift_box(cloud.xyz, list(o.bbox), cam, w, h, ground_plane=plane)
        if cuboid is None:
            continue
        prov = o.provenance or {}
        agreement = bool(prov.get("agreement", o.source in ("auto_accept", "human")))
        g = gate_cuboid(cuboid, class_id=o.class_id, conf_2d=float(o.conf), frame_id=frame_id,
                        box_source="lifted", bbox_2d=list(o.bbox), agreement_2d=agreement,
                        track_id=o.track_id, model_version=LIFT_VERSION, calibration_version=CALIB_VERSION,
                        auto_accept_enabled=aae)
        proposals.append((o.object_id, cuboid, g))

    async with get_sessionmaker()() as db:
        # scope the idempotent clear to THIS frame's lifted objects: a fused cloud is shared by every camera
        # at this ts_ns, so deleting by cloud_id would wipe another camera's lifts. Human boxes survive.
        await db.execute(delete(Object3D).where(Object3D.frame_id == frame_id,
                         Object3D.box_source == "lifted", Object3D.source != "human"))
        rows = [_row(cloud_id, frame_id, oid, g, cuboid) for oid, cuboid, g in proposals]
        for r in rows:
            db.add(r)
        await db.flush()
        out = [{"object_3d_id": str(r.object_3d_id), "object_id": str(r.object_id) if r.object_id else None,
                "class_id": r.class_id, "center": r.center, "dims": r.dims, "yaw": r.yaw, "conf": r.conf,
                "state": r.state} for r in rows]
        await db.commit()
    log.info("lidar.lift_frame", frame=str(frame_id), cuboids=len(out))
    return {"frame_id": str(frame_id), "cloud_id": str(cloud_id), "cuboids": len(out),
            "box_source": "lifted", "objects": out}


async def detect_native_cloud(cloud_id: uuid.UUID) -> dict:
    """Native 3D detection on a real-LiDAR cloud, gated and persisted. Requires OpenPCDet on the burst node;
    raises the seam locally so the lidar_perception job queues it for the A100."""
    async with get_sessionmaker()() as db:
        pc = await db.get(PointCloud, cloud_id)
        if pc is None:
            return {"error": "cloud not found"}
        frame = (await db.execute(
            select(Frame).where(Frame.session_id == pc.session_id, Frame.ts_ns == pc.ts_ns)
            .order_by(Frame.cam_id).limit(1))).scalar_one_or_none()
        aae = await _auto_accept_enabled(db)
        frame_id = frame.frame_id if frame else None

    cloud = load_cloud(pc.cloud_uri)
    dets = detect_native(cloud)            # raises NativeDetectionUnavailable when OpenPCDet is absent
    proposals = []
    for d in dets:
        g = gate_cuboid(d, class_id=d["class_id"], conf_2d=d["conf"], frame_id=frame_id,
                        box_source="native", model_version=d.get("native_class", "native-3d"),
                        calibration_version=CALIB_VERSION, auto_accept_enabled=aae)
        proposals.append((None, d, g))

    async with get_sessionmaker()() as db:
        # native detection is per-cloud (one set per scan); scope the clear to native machine objects so it
        # never deletes the per-frame lifted boxes on the same cloud.
        await db.execute(delete(Object3D).where(Object3D.cloud_id == cloud_id,
                         Object3D.box_source == "native", Object3D.source != "human"))
        rows = [_row(cloud_id, frame_id, oid, g, cuboid) for oid, cuboid, g in proposals]
        for r in rows:
            db.add(r)
        await db.commit()
    log.info("lidar.native_cloud", cloud=str(cloud_id), cuboids=len(proposals))
    return {"cloud_id": str(cloud_id), "cuboids": len(proposals), "box_source": "native"}
