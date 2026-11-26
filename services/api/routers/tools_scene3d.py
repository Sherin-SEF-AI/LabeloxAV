"""API for the scene3d annotation tools, agent C: 3D (#3 label once in 3D, #5 point segmentation, #6 occupancy).

Mounted under /api; every path here starts with /tools/scene3d/.

#3 Label once in 3D (/lidar/annotate, "Label once" tab)
- GET  sessions/{id}/scene/points     every k-th scan placed by its ego pose, as one binary cloud (scene frame)
- GET  sessions/{id}/scene/poses      the placed scans' positions, for the trajectory overlay
- POST sessions/{id}/label-once       queue `scene3d_label_once`: one cuboid -> every scan -> every camera
- GET  sessions/{id}/label-once       the tracks this tool made in the session
- GET  tracks3d/{id}/review           per scan: the 3D box, its LiDAR support and the 2D box in each camera
- POST tracks3d/{id}/review           accept or reject some or all of them, 3D and 2D together (reviewer)

#5 Point labels (/lidar/segment)
- GET  clouds/{id}/points             Float32 xyzi for every `stride`-th point (X-Stride says which)
- GET  clouds/{id}/segmentations      the label versions on a cloud (auto, model, human)
- GET  clouds/{id}/labels             Int16 class id per subset point for one version
- PUT  clouds/{id}/labels             save the browser's labels as a new human version (octet-stream Int16)
- POST clouds/{id}/assist/ground|cluster|from-2d|from-cuboids    one-click selections and suggestions
- POST clouds/{id}/assist/ai          queue PTv3 (`scene3d_ptv3`); GET ptv3/status says whether it can run

#6 Occupancy (/lidar/occupancy)
- GET  sessions/{id}/occupancy        the generated grids, each with whether a person edited it
- GET  occupancy/{grid_id}            base + human delta + merged voxels
- POST occupancy/{grid_id}/edit       paint or box-fill voxels occupied/free/reset, into the human layer
- POST occupancy/{grid_id}/propagate  carry the edits to the next grids through the ego poses
- POST sessions/{id}/occupancy/build  queue grids from LiDAR or, without LiDAR, from camera depth (reviewer)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import OrderedDict
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import (
    EgoPose,
    Frame,
    Object,
    Object3D,
    OccupancyGrid,
    PointCloud,
    PointSegmentation,
    Track3D,
)
from services.api.deps import db_session, require_role, require_user
from services.tools.registry import enqueue

log = get_logger("tools_scene3d")

router = APIRouter()

_BIN = "application/octet-stream"


def _job_dict(job) -> dict:
    return {"job_id": str(job.job_id), "kind": job.kind, "status": job.status,
            "session_id": str(job.session_id) if job.session_id else None}


def _bin(arr: np.ndarray, headers: dict) -> Response:
    headers = {k: str(v) for k, v in headers.items()}
    headers["Access-Control-Expose-Headers"] = ",".join(headers)
    return Response(content=np.ascontiguousarray(arr).tobytes(), media_type=_BIN, headers=headers)


async def _cloud(db: AsyncSession, cloud_id: uuid.UUID) -> PointCloud:
    pc = await db.get(PointCloud, cloud_id)
    if pc is None:
        raise HTTPException(404, "cloud not found")
    return pc


# ---- session overview ---------------------------------------------------------------------------------------

@router.get("/tools/scene3d/sessions/{session_id}/overview", dependencies=[Depends(require_user)])
async def overview(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """What 3D data a session has, so each page can say which of its tools apply."""
    from db.models import CameraCalibration
    from db.models import Session as DbSession

    sess = await db.get(DbSession, session_id)
    if sess is None:
        raise HTTPException(404, "session not found")
    clouds = dict((await db.execute(select(PointCloud.source, func.count()).where(
        PointCloud.session_id == session_id).group_by(PointCloud.source))).all())
    poses = (await db.execute(select(func.count()).select_from(EgoPose).where(
        EgoPose.session_id == session_id))).scalar() or 0
    cams = (await db.execute(select(Frame.cam_id, func.count()).where(Frame.session_id == session_id)
                             .group_by(Frame.cam_id))).all()
    cal = (await db.execute(select(CameraCalibration.cam_id, CameraCalibration.source).where(
        CameraCalibration.session_id == session_id))).all()
    grids = dict((await db.execute(select(OccupancyGrid.source, func.count()).where(
        OccupancyGrid.session_id == session_id).group_by(OccupancyGrid.source))).all())
    return {"session_id": str(session_id), "vehicle_id": sess.vehicle_id, "route": sess.route,
            "clouds": {k: int(v) for k, v in clouds.items()}, "lidar": int(clouds.get("lidar", 0)) > 0,
            "poses": int(poses), "cameras": {c: int(n) for c, n in cams},
            "calibration": {c: s for c, s in cal},
            "ego_frame": (sess.sensors or {}).get("ego_frame", "ego"),
            "lidar_height_m": (sess.sensors or {}).get("lidar_height_m"),
            "occupancy": {k: int(v) for k, v in grids.items()}}


# ---- #3 label once in 3D ------------------------------------------------------------------------------------

# Accumulated scenes are a few seconds of object-store reads; the last few are kept so switching tabs or
# re-opening the label view does not pay that again. Bytes, so the cache is bounded (4 x ~10 MB).
_SCENE_CACHE: OrderedDict[tuple, tuple[bytes, dict]] = OrderedDict()
_SCENE_CACHE_MAX = 4


async def _scene_poses(db: AsyncSession, session_id: uuid.UUID):
    from services.lidar.aggregate.scene_frame import session_scene_poses

    poses, summary = await session_scene_poses(db, session_id)
    if not poses:
        raise HTTPException(409, summary.get("reason") or summary.get("error") or "no scan could be placed")
    anchor = np.round(poses[0].T_world_ego[:3, 3], 2)
    return poses, summary, anchor


@router.get("/tools/scene3d/sessions/{session_id}/scene/points", dependencies=[Depends(require_user)])
async def scene_points(session_id: uuid.UUID, every: int = Query(10, ge=1, le=100),
                       max_points: int = Query(600_000, alias="max", ge=10_000, le=2_000_000),
                       range_m: float = Query(40.0, ge=5, le=120),
                       db: AsyncSession = Depends(db_session)):
    """Every `every`-th scan of the session placed in one frame by its ego pose: Float32 [x, y, z, intensity]
    in scene coordinates (world minus X-Anchor, so the browser's float32 keeps centimetres)."""
    from services.lidar.aggregate.scene_frame import accumulate_scene

    poses, summary, anchor = await _scene_poses(db, session_id)
    key = (str(session_id), every, max_points, range_m, len(poses))
    hit = _SCENE_CACHE.get(key)
    if hit is None:
        acc = await asyncio.to_thread(accumulate_scene, poses, every=every, max_range_m=range_m,
                                      max_points=max_points)
        packed = np.empty((len(acc["xyz"]), 4), dtype=np.float32)
        packed[:, :3] = acc["xyz"] - anchor.astype(np.float32)
        packed[:, 3] = acc["intensity"]
        hit = (packed.tobytes(), {"X-Point-Count": len(packed), "X-Scans": acc["scans"],
                                  "X-Anchor": ",".join(f"{v:.2f}" for v in anchor),
                                  "X-Placed": summary["placed"], "X-Lever-Arm": summary["lever_arm"]})
        _SCENE_CACHE[key] = hit
        while len(_SCENE_CACHE) > _SCENE_CACHE_MAX:
            _SCENE_CACHE.popitem(last=False)
    else:
        _SCENE_CACHE.move_to_end(key)
    headers = {k: str(v) for k, v in hit[1].items()}
    headers["Access-Control-Expose-Headers"] = ",".join(headers)
    return Response(content=hit[0], media_type=_BIN, headers=headers)


@router.get("/tools/scene3d/sessions/{session_id}/scene/poses", dependencies=[Depends(require_user)])
async def scene_poses(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.lidar.aggregate.scene_frame import yaw_of

    poses, summary, anchor = await _scene_poses(db, session_id)
    return {"anchor": anchor.tolist(), "summary": summary,
            "poses": [{"cloud_id": str(p.cloud_id), "ts_ns": p.ts_ns,
                       "x": round(float(p.T_world_ego[0, 3] - anchor[0]), 3),
                       "y": round(float(p.T_world_ego[1, 3] - anchor[1]), 3),
                       "z": round(float(p.T_world_ego[2, 3] - anchor[2]), 3),
                       "yaw": round(yaw_of(p.T_world_ego), 5)} for p in poses]}


class LabelOnceIn(BaseModel):
    center: list[float] = Field(min_length=3, max_length=3)   # scene coordinates (world minus the anchor)
    dims: list[float] = Field(min_length=3, max_length=3)     # [L, W, H] metres
    yaw: float = 0.0                                          # radians about up, in the scene frame
    class_id: int
    ground_snap: bool = True
    max_range_m: float = Field(60.0, ge=5, le=150)
    project: bool = True


@router.post("/tools/scene3d/sessions/{session_id}/label-once")
async def label_once(session_id: uuid.UUID, body: LabelOnceIn, db: AsyncSession = Depends(db_session),
                     user=Depends(require_role("annotator"))):
    """Queue the label-once job. The cuboid comes in scene coordinates and is stored in world coordinates."""
    from services.autolabel.ontology import get_ontology

    try:
        get_ontology().by_id(body.class_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"class {body.class_id} is not in the ontology") from exc
    if min(body.dims) <= 0.05 or max(body.dims) > 30:
        raise HTTPException(422, "dims must be between 5 cm and 30 m")
    _, _, anchor = await _scene_poses(db, session_id)
    center_world = [float(c + a) for c, a in zip(body.center, anchor.tolist(), strict=True)]
    job = await enqueue(db, "scene3d_label_once",
                        {"session_id": str(session_id), "center": center_world, "dims": body.dims,
                         "yaw": body.yaw, "class_id": body.class_id, "ground_snap": body.ground_snap,
                         "max_range_m": body.max_range_m, "project": body.project,
                         "scene_center": body.center},
                        session_id=session_id, user_id=user.user_id)
    return _job_dict(job)


@router.get("/tools/scene3d/sessions/{session_id}/label-once", dependencies=[Depends(require_user)])
async def label_once_tracks(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    rows = (await db.execute(select(Track3D).where(
        Track3D.session_id == session_id,
        Track3D.trajectory["method"].astext == "scene_label_once").order_by(Track3D.first_ts_ns))).scalars().all()
    out = []
    for t in rows:
        states = dict((await db.execute(select(Object3D.state, func.count()).where(
            Object3D.track_3d_id == t.track_3d_id).group_by(Object3D.state))).all())
        try:
            cname = onto.by_id(t.class_id).name
        except Exception:  # noqa: BLE001
            cname = str(t.class_id)
        out.append({"track_3d_id": str(t.track_3d_id), "class_id": t.class_id, "class_name": cname,
                    "first_ts_ns": t.first_ts_ns, "last_ts_ns": t.last_ts_ns,
                    "scene_label": (t.trajectory or {}).get("scene_label"),
                    "states": {k: int(v) for k, v in states.items()}, "scans": int(sum(states.values()))})
    return {"tracks": out}


@router.get("/tools/scene3d/tracks3d/{track_3d_id}/review", dependencies=[Depends(require_user)])
async def track_review(track_3d_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Every scan of a label-once track: the 3D box, how many LiDAR points support it, and its 2D boxes."""
    tr = await db.get(Track3D, track_3d_id)
    if tr is None:
        raise HTTPException(404, "track not found")
    o3s = (await db.execute(select(Object3D, PointCloud.ts_ns).join(PointCloud, PointCloud.cloud_id == Object3D.cloud_id)
                            .where(Object3D.track_3d_id == track_3d_id).order_by(PointCloud.ts_ns))).all()
    ids = [str(o.object_3d_id) for o, _ in o3s]
    o2s = (await db.execute(select(Object, Frame.cam_id, Frame.width, Frame.height, Frame.ts_ns)
                            .join(Frame, Frame.frame_id == Object.frame_id)
                            .where(Object.provenance["track_3d_id"].astext == str(track_3d_id)))).all()
    by3: dict[str, list] = {}
    for o, cam, w, h, _ts in o2s:
        by3.setdefault(str((o.provenance or {}).get("object_3d_id")), []).append(
            {"object_id": str(o.object_id), "frame_id": str(o.frame_id), "cam_id": cam, "bbox": list(o.bbox),
             "state": o.state, "width": w, "height": h, "version": o.version})
    rows = []
    for o, ts in o3s:
        rows.append({"object_3d_id": str(o.object_3d_id), "cloud_id": str(o.cloud_id), "ts_ns": int(ts),
                     "frame_id": str(o.frame_id) if o.frame_id else None, "center": list(o.center),
                     "dims": list(o.dims), "yaw": o.yaw, "state": o.state, "version": o.version,
                     "n_points": (o.attrs or {}).get("n_points"), "range_m": (o.attrs or {}).get("range_m"),
                     "ground_snapped": (o.attrs or {}).get("ground_snapped"),
                     "boxes": by3.get(str(o.object_3d_id), [])})
    return {"track_3d_id": str(track_3d_id), "class_id": tr.class_id, "session_id": str(tr.session_id),
            "track_id": str(tr.track_id) if tr.track_id else None,
            "scene_label": (tr.trajectory or {}).get("scene_label"), "scans": rows, "n": len(ids)}


class TrackReviewIn(BaseModel):
    action: Literal["accept", "reject"]
    object_3d_ids: list[uuid.UUID] | None = None     # None: every box of the track
    include_2d: bool = True


@router.post("/tools/scene3d/tracks3d/{track_3d_id}/review")
async def track_review_apply(track_3d_id: uuid.UUID, body: TrackReviewIn, db: AsyncSession = Depends(db_session),
                             user=Depends(require_role("reviewer"))):
    """Accept or reject label-once boxes: the 3D cuboid and, with include_2d, the 2D boxes projected from it.
    The 2D side goes through the shared review path (role clamp, version, one revertible batch)."""
    from services.autolabel.ontology import get_ontology
    from services.review_apply import apply_review_batch
    from services.review_batch import record_batch

    tr = await db.get(Track3D, track_3d_id)
    if tr is None:
        raise HTTPException(404, "track not found")
    q = select(Object3D).where(Object3D.track_3d_id == track_3d_id)
    if body.object_3d_ids:
        q = q.where(Object3D.object_3d_id.in_(body.object_3d_ids))
    o3s = (await db.execute(q)).scalars().all()
    state = "accepted" if body.action == "accept" else "rejected"
    for o in o3s:
        o.state = state
        o.version = (o.version or 1) + 1
    n2, run_id = 0, None
    if body.include_2d and o3s:
        keys = [str(o.object_3d_id) for o in o3s]
        objs = (await db.execute(select(Object).where(
            Object.provenance["object_3d_id"].astext.in_(keys)))).scalars().all()
        if objs:
            res = await apply_review_batch(db, list(objs), action=body.action, onto=get_ontology(),
                                           role=getattr(user, "role", None), source="human",
                                           reviewer=user.name, uid=user.user_id,
                                           provenance_extra={"via": "scene3d_label_once_review"})
            n2 = res.n
            if res.changes:
                run_id = await record_batch(db, res.changes, created_by=user.name, commit=False,
                                            policy={"action": body.action, "tool": "scene3d_label_once"})
    await db.commit()
    return {"track_3d_id": str(track_3d_id), "state": state, "updated_3d": len(o3s), "updated_2d": n2,
            "run_id": run_id}


# ---- #5 point labels ----------------------------------------------------------------------------------------

def _subset(n: int, max_points: int) -> int:
    from services.lidar.segment3d.human import stride_for

    return stride_for(n, max_points)


async def _load(uri: str):
    from services.lidar.ingest.store import load_cloud

    return await asyncio.to_thread(load_cloud, uri)


@router.get("/tools/scene3d/clouds/{cloud_id}/points", dependencies=[Depends(require_user)])
async def seg_points(cloud_id: uuid.UUID, max_points: int = Query(400_000, alias="max", ge=1000, le=2_000_000),
                     db: AsyncSession = Depends(db_session)):
    """Float32 [x, y, z, intensity] for points 0, s, 2s, ... (s = X-Stride), the subset labels are edited on."""
    pc = await _cloud(db, cloud_id)
    cloud = await _load(pc.cloud_uri)
    s = _subset(cloud.n, max_points)
    packed = np.empty((len(range(0, cloud.n, s)), 4), dtype=np.float32)
    packed[:, :3] = cloud.xyz[::s]
    packed[:, 3] = cloud.intensity[::s] if cloud.intensity is not None else 0.0
    imin = float(packed[:, 3].min()) if len(packed) else 0.0
    imax = float(packed[:, 3].max()) if len(packed) else 1.0
    return _bin(packed, {"X-Point-Count": len(packed), "X-Total": cloud.n, "X-Stride": s, "X-Source": pc.source,
                         "X-Session": pc.session_id, "X-Ts": pc.ts_ns, "X-Intensity-Min": f"{imin:.5f}",
                         "X-Intensity-Max": f"{imax:.5f}"})


def _seg_dict(r: PointSegmentation) -> dict:
    return {"seg_id": str(r.seg_id), "source": r.source, "version": r.version, "method": r.method,
            "model_version": r.model_version, "kind": r.kind, "n_points": r.n_points,
            "low_conf_frac": r.low_conf_frac, "base_seg_id": str(r.base_seg_id) if r.base_seg_id else None,
            "created_by": str(r.created_by) if r.created_by else None,
            "created_at": r.created_at.isoformat() if r.created_at else None}


@router.get("/tools/scene3d/clouds/{cloud_id}/segmentations", dependencies=[Depends(require_user)])
async def seg_versions(cloud_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    await _cloud(db, cloud_id)
    rows = (await db.execute(select(PointSegmentation).where(PointSegmentation.cloud_id == cloud_id)
                             .order_by(PointSegmentation.created_at.desc()))).scalars().all()
    return {"cloud_id": str(cloud_id), "versions": [_seg_dict(r) for r in rows]}


@router.get("/tools/scene3d/clouds/{cloud_id}/labels", dependencies=[Depends(require_user)])
async def seg_labels(cloud_id: uuid.UUID, stride: int = Query(1, ge=1, le=10_000), seg_id: uuid.UUID | None = None,
                     db: AsyncSession = Depends(db_session)):
    """Int16 class id (-1 unlabelled) for points 0, stride, 2*stride, ... of one label version: the one named,
    else the latest human version, else the latest of any source."""
    from services.lidar.segment3d.human import full_labels, latest_segmentation

    pc = await _cloud(db, cloud_id)
    if seg_id is not None:
        row = await db.get(PointSegmentation, seg_id)
        if row is None or row.cloud_id != cloud_id:
            raise HTTPException(404, "segmentation not found on this cloud")
    else:
        row = await latest_segmentation(db, cloud_id, "human") or await latest_segmentation(db, cloud_id)
    n = pc.point_count
    labels = await asyncio.to_thread(full_labels, row, n) if row is not None else np.full(n, -1, np.int32)
    sub = labels[::stride].astype(np.int16)
    return _bin(sub, {"X-Point-Count": len(sub), "X-Seg-Id": row.seg_id if row else "",
                      "X-Seg-Source": row.source if row else "", "X-Seg-Version": row.version if row else 0})


@router.put("/tools/scene3d/clouds/{cloud_id}/labels")
async def seg_save(cloud_id: uuid.UUID, request: Request, stride: int = Query(..., ge=1, le=10_000),
                   base_seg_id: uuid.UUID | None = None, db: AsyncSession = Depends(db_session),
                   user=Depends(require_role("annotator"))):
    """Save the browser's Int16 labels (one per subset point) as a new human version merged over the base."""
    from services.lidar.segment3d.human import save_human_labels

    pc = await _cloud(db, cloud_id)
    body = await request.body()
    if len(body) % 2:
        raise HTTPException(422, "body must be Int16 labels")
    labels = np.frombuffer(body, dtype=np.int16).astype(np.int32)
    expected = len(range(0, pc.point_count, stride))
    if len(labels) != expected:
        raise HTTPException(422, f"{len(labels)} labels sent; this cloud at stride {stride} has {expected} points")
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    for cid in np.unique(labels[labels >= 0]).tolist():
        try:
            onto.by_id(int(cid))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"class {cid} is not in the ontology") from exc
    try:
        return await save_human_labels(db, cloud_id, labels, stride, base_seg_id=base_seg_id, user_id=user.user_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/tools/scene3d/clouds/{cloud_id}/assist/ground", dependencies=[Depends(require_role("annotator"))])
async def assist_ground(cloud_id: uuid.UUID, stride: int = Query(1, ge=1, le=10_000),
                        db: AsyncSession = Depends(db_session)):
    """Uint8 1/0 per subset point: on the fitted ground plane."""
    from services.lidar.segment3d.human import ground_mask

    pc = await _cloud(db, cloud_id)
    cloud = await _load(pc.cloud_uri)
    mask, plane, method = await asyncio.to_thread(ground_mask, cloud)
    sub = mask[::stride].astype(np.uint8)
    return _bin(sub, {"X-Point-Count": len(sub), "X-Selected": int(sub.sum()), "X-Method": method,
                      "X-Plane": ",".join(f"{v:.5f}" for v in plane)})


class ClusterIn(BaseModel):
    point: list[float] = Field(min_length=3, max_length=3)
    eps: float = Field(0.5, gt=0.05, le=3.0)
    min_points: int = Field(8, ge=1, le=200)


@router.post("/tools/scene3d/clouds/{cloud_id}/assist/cluster", dependencies=[Depends(require_role("annotator"))])
async def assist_cluster(cloud_id: uuid.UUID, body: ClusterIn, stride: int = Query(1, ge=1, le=10_000),
                         db: AsyncSession = Depends(db_session)):
    """Uint8 per subset point: the DBSCAN cluster of non-ground points under the clicked point."""
    from services.lidar.segment3d.human import cluster_at

    pc = await _cloud(db, cloud_id)
    cloud = await _load(pc.cloud_uri)
    mask, info = await asyncio.to_thread(cluster_at, cloud, body.point, eps=body.eps, min_points=body.min_points)
    sub = mask[::stride].astype(np.uint8)
    return _bin(sub, {"X-Point-Count": len(sub), "X-Selected": int(sub.sum()), "X-Info": json.dumps(info)})


@router.post("/tools/scene3d/clouds/{cloud_id}/assist/from-2d", dependencies=[Depends(require_role("annotator"))])
async def assist_from_2d(cloud_id: uuid.UUID, stride: int = Query(1, ge=1, le=10_000),
                         db: AsyncSession = Depends(db_session)):
    """Int16 per subset point: the class of the 2D mask it projects into (-1 where none does)."""
    from services.lidar.segment3d.human import labels_from_2d

    pc = await _cloud(db, cloud_id)
    cloud = await _load(pc.cloud_uri)
    labels, report = await labels_from_2d(db, cloud_id, cloud)
    sub = labels[::stride].astype(np.int16)
    return _bin(sub, {"X-Point-Count": len(sub), "X-Selected": int((sub >= 0).sum()), "X-Info": json.dumps(report)})


@router.post("/tools/scene3d/clouds/{cloud_id}/assist/from-cuboids",
             dependencies=[Depends(require_role("annotator"))])
async def assist_from_cuboids(cloud_id: uuid.UUID, stride: int = Query(1, ge=1, le=10_000),
                              db: AsyncSession = Depends(db_session)):
    """Int16 per subset point from the cloud's cuboids and ground plane (the projected segmentation)."""
    from services.lidar.segment3d.human import labels_from_cuboids

    pc = await _cloud(db, cloud_id)
    cloud = await _load(pc.cloud_uri)
    labels, report = await labels_from_cuboids(db, cloud_id, cloud)
    sub = labels[::stride].astype(np.int16)
    return _bin(sub, {"X-Point-Count": len(sub), "X-Selected": int((sub >= 0).sum()), "X-Info": json.dumps(report)})


@router.get("/tools/scene3d/ptv3/status", dependencies=[Depends(require_user)])
async def ptv3_status():
    """Whether the PTv3 point segmenter can run on this machine, and if not, why."""
    try:
        from services.lidar.segment3d import ptv3_infer
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"not installed on this machine ({type(exc).__name__})"}
    ok, reason = ptv3_infer.available()
    return {"available": bool(ok), "reason": reason, "classes": list(getattr(ptv3_infer, "CLASS_NAMES", []))}


@router.post("/tools/scene3d/clouds/{cloud_id}/assist/ai")
async def assist_ai(cloud_id: uuid.UUID, db: AsyncSession = Depends(db_session),
                    user=Depends(require_role("annotator"))):
    """Queue PTv3 on this cloud. Its labels land as a 'model' version the page offers as suggestions."""
    status = await ptv3_status()
    if not status["available"]:
        raise HTTPException(409, f"PTv3 cannot run here: {status['reason']}")
    pc = await _cloud(db, cloud_id)
    job = await enqueue(db, "scene3d_ptv3", {"cloud_ids": [str(cloud_id)]}, session_id=pc.session_id,
                        user_id=user.user_id)
    return _job_dict(job)


# ---- #6 occupancy -------------------------------------------------------------------------------------------

@router.get("/tools/scene3d/sessions/{session_id}/occupancy", dependencies=[Depends(require_user)])
async def occ_list(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    base = (await db.execute(select(OccupancyGrid).where(
        OccupancyGrid.session_id == session_id, OccupancyGrid.source != "human").order_by(OccupancyGrid.ts_ns))).scalars().all()
    layers = {r.base_grid_id: r for r in (await db.execute(select(OccupancyGrid).where(
        OccupancyGrid.session_id == session_id, OccupancyGrid.source == "human"))).scalars().all()}
    out = []
    for g in base:
        lay = layers.get(g.grid_id)
        out.append({"grid_id": str(g.grid_id), "ts_ns": g.ts_ns, "source": g.source, "occupied": g.occupied,
                    "flow_voxels": g.flow_voxels, "placed": g.ego_pose_ts is not None, "voxel_m": g.voxel_m,
                    "dims": list(g.dims), "frame_id": str(g.frame_id) if g.frame_id else None,
                    "edited": lay is not None, "occupied_after": lay.occupied if lay is not None else g.occupied})
    return {"session_id": str(session_id), "grids": out}


# A browser draws an instanced cube per voxel; above this the page would stall, so the reply says it is cut.
MAX_VOXELS_OUT = 120_000


@router.get("/tools/scene3d/occupancy/{grid_id}", dependencies=[Depends(require_user)])
async def occ_get(grid_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.lidar.occupancy_edit import FREE, OCCUPIED, read_grid

    try:
        g = await read_grid(db, grid_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    base = g["base"]
    vox, flow, edited = g["voxels"], g["flow"], g["edited"]
    total = len(vox)
    if total > MAX_VOXELS_OUT:
        keep = np.argsort(~edited, kind="stable")[:MAX_VOXELS_OUT]
        vox, flow, edited = vox[keep], flow[keep], edited[keep]
    dv, dval = g["delta_voxels"], g["delta_values"]
    return {"grid_id": str(grid_id), "session_id": str(base.session_id), "ts_ns": base.ts_ns, "source": base.source,
            "origin": list(base.origin), "voxel_m": base.voxel_m, "dims": list(g["dims"]),
            "placed": base.ego_pose_ts is not None, "frame_id": str(base.frame_id) if base.frame_id else None,
            "voxels": vox.tolist(), "flow": np.round(flow, 2).tolist(), "edited": edited.astype(int).tolist(),
            "total": total, "truncated": total > len(vox), "base_occupied": g["base_occupied"],
            "set_occupied": dv[dval == OCCUPIED].tolist(), "set_free": dv[dval == FREE].tolist(),
            "edited_by": str(g["layer"].edited_by) if g["layer"] is not None and g["layer"].edited_by else None}


class OccOp(BaseModel):
    kind: Literal["paint", "box"]
    value: Literal["occupied", "free", "reset"]
    voxels: list[list[int]] | None = None
    lo: list[int] | None = Field(None, min_length=3, max_length=3)
    hi: list[int] | None = Field(None, min_length=3, max_length=3)


class OccEditIn(BaseModel):
    ops: list[OccOp] = Field(min_length=1, max_length=200)


@router.post("/tools/scene3d/occupancy/{grid_id}/edit")
async def occ_edit(grid_id: uuid.UUID, body: OccEditIn, db: AsyncSession = Depends(db_session),
                   user=Depends(require_role("annotator"))):
    from services.lidar.occupancy_edit import edit_grid

    ops = [o.model_dump() for o in body.ops]
    for o in ops:
        if o["kind"] == "box" and (o["lo"] is None or o["hi"] is None):
            raise HTTPException(422, "a box op needs lo and hi")
    try:
        return await edit_grid(db, grid_id, ops, user.user_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409 if "another grid" in str(exc) else 422, str(exc)) from exc


class OccPropagateIn(BaseModel):
    count: int = Field(10, ge=1, le=200)
    ops: list[OccOp] | None = None       # carry only these ops; None carries the grid's whole edit layer


@router.post("/tools/scene3d/occupancy/{grid_id}/propagate")
async def occ_propagate(grid_id: uuid.UUID, body: OccPropagateIn, db: AsyncSession = Depends(db_session),
                        user=Depends(require_role("annotator"))):
    from services.lidar.occupancy_edit import propagate_edits

    try:
        return await propagate_edits(db, grid_id, body.count, user.user_id,
                                     only_ops=[o.model_dump() for o in body.ops] if body.ops else None)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


class OccBuildIn(BaseModel):
    mode: Literal["auto", "lidar", "camera"] = "auto"
    every: int = Field(5, ge=1, le=100)      # camera mode: lift every k-th instant
    limit: int = Field(40, ge=1, le=500)


@router.post("/tools/scene3d/sessions/{session_id}/occupancy/build")
async def occ_build(session_id: uuid.UUID, body: OccBuildIn, db: AsyncSession = Depends(db_session),
                    user=Depends(require_role("reviewer"))):
    """Queue occupancy generation: from the session's clouds, or (no LiDAR, or mode=camera) from camera depth."""
    n_lidar = (await db.execute(select(func.count()).select_from(PointCloud).where(
        PointCloud.session_id == session_id, PointCloud.source == "lidar"))).scalar() or 0
    mode = body.mode if body.mode != "auto" else ("lidar" if n_lidar else "camera")
    if mode == "lidar":
        n_any = (await db.execute(select(func.count()).select_from(PointCloud).where(
            PointCloud.session_id == session_id))).scalar() or 0
        if not n_any:
            raise HTTPException(409, "this session has no point clouds; build from cameras instead")
        job = await enqueue(db, "scene3d_occupancy_lidar", {"session_id": str(session_id), "limit": body.limit},
                            session_id=session_id, user_id=user.user_id)
    else:
        job = await enqueue(db, "scene3d_occupancy_camera",
                            {"session_id": str(session_id), "every": body.every, "limit": body.limit},
                            session_id=session_id, user_id=user.user_id)
    return {**_job_dict(job), "mode": mode}
