"""3D cuboid CRUD for the annotation workspace. Create and edit oriented cuboids (centre, L/W/H, yaw, pitch,
roll), ground-snapped to the Phase 1 plane, with the same governed ontology class. Human edits write
object_3d with source=human and bump the version for optimistic locking (a stale write is a 409). A cuboid
projects onto the camera image via the Phase 1 projection, the foundation for the linked workspace.
"""

from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object3D, PointCloud, PointCloudDerived, PointSegmentation, Track3D
from services.api.deps import db_session
from services.autolabel.ontology import get_ontology
from services.lidar.boxes import project_cuboid, snap_to_ground
from services.lidar.clean.ground import segment_ground
from services.lidar.ingest.store import load_cloud

log = get_logger("api_objects3d")
router = APIRouter()


class Cuboid3DIn(BaseModel):
    class_id: int
    center: list[float] = Field(min_length=3, max_length=3)
    dims: list[float] = Field(min_length=3, max_length=3)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    attrs: dict = Field(default_factory=dict)
    object_id: uuid.UUID | None = None
    ground_snap: bool = True


class Cuboid3DEdit(BaseModel):
    class_id: int | None = None
    center: list[float] | None = None
    dims: list[float] | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    attrs: dict | None = None
    ground_snap: bool = False
    expected_version: int | None = None


async def _ground_plane(db: AsyncSession, pc: PointCloud) -> list[float]:
    """The cloud's ground plane: reuse the M-L1.2 derived ground_plane if present, else segment on demand."""
    d = (await db.execute(
        select(PointCloudDerived.params).where(PointCloudDerived.cloud_id == pc.cloud_id,
        PointCloudDerived.kind == "ground_plane").order_by(PointCloudDerived.created_at.desc()).limit(1))
        ).scalar_one_or_none()
    if d and isinstance(d.get("plane"), list) and len(d["plane"]) == 4:
        return d["plane"]
    _, plane, _ = segment_ground(load_cloud(pc.cloud_uri))
    return plane


def _serialize(o: Object3D) -> dict:
    onto = get_ontology()
    return {"object_3d_id": str(o.object_3d_id), "cloud_id": str(o.cloud_id),
            "frame_id": str(o.frame_id) if o.frame_id else None,
            "object_id": str(o.object_id) if o.object_id else None,
            "track_3d_id": str(o.track_3d_id) if o.track_3d_id else None,
            "class_id": o.class_id, "class_name": onto.by_id(o.class_id).name,
            "center": o.center, "dims": o.dims, "yaw": o.yaw, "pitch": o.pitch, "roll": o.roll,
            "conf": o.conf, "box_source": o.box_source, "source": o.source, "state": o.state,
            "is_keyframe": o.is_keyframe, "attrs": o.attrs, "version": o.version}


@router.get("/lidar/clouds/{cloud_id}/objects3d")
async def list_objects3d(cloud_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(Object3D).where(Object3D.cloud_id == cloud_id)
                             .order_by(Object3D.created_at))).scalars().all()
    return {"cloud_id": str(cloud_id), "objects": [_serialize(o) for o in rows]}


@router.post("/lidar/frames/{frame_id}/lift")
async def lift_frame_objects(frame_id: uuid.UUID):
    """AI-assist: lift the frame's 2D objects into ground-snapped 3D cuboids (M-L2.0), seeding the workspace
    for the annotator to refine. Machine cuboids are rewritten; human cuboids survive."""
    from services.lidar.detect3d import lift_frame
    return await lift_frame(frame_id)


@router.post("/lidar/clouds/{cloud_id}/lift")
async def lift_cloud_objects(cloud_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """AI-assist lift addressed by cloud: resolve the synchronized camera frame and lift its 2D objects."""
    pc = await db.get(PointCloud, cloud_id)
    if pc is None:
        raise HTTPException(404, "cloud not found")
    frame = (await db.execute(select(Frame.frame_id).where(Frame.session_id == pc.session_id,
             Frame.ts_ns == pc.ts_ns).limit(1))).scalar_one_or_none()
    if frame is None:
        raise HTTPException(400, "no synchronized camera frame for this cloud")
    from services.lidar.detect3d import lift_frame
    return await lift_frame(frame)


@router.post("/lidar/clouds/{cloud_id}/objects3d")
async def create_object3d(cloud_id: uuid.UUID, body: Cuboid3DIn, db: AsyncSession = Depends(db_session)):
    pc = await db.get(PointCloud, cloud_id)
    if pc is None:
        raise HTTPException(404, "cloud not found")
    onto = get_ontology()
    try:
        onto.by_id(body.class_id)
    except Exception as exc:
        raise HTTPException(422, f"class_id {body.class_id} not in ontology") from exc
    center = body.center
    if body.ground_snap:
        center = snap_to_ground(center, body.dims, await _ground_plane(db, pc))
    frame = (await db.execute(select(Frame.frame_id).where(Frame.session_id == pc.session_id,
             Frame.ts_ns == pc.ts_ns).limit(1))).scalar_one_or_none()
    row = Object3D(cloud_id=cloud_id, frame_id=frame, object_id=body.object_id, class_id=body.class_id,
                   center=center, dims=body.dims, yaw=body.yaw, pitch=body.pitch, roll=body.roll,
                   conf=1.0, box_source="manual", source="human", state="accepted", is_keyframe=True,
                   attrs=body.attrs, provenance={"author": "human"})
    db.add(row)
    await db.flush()
    out = _serialize(row)
    await db.commit()
    log.info("objects3d.create", cloud=str(cloud_id), cls=out["class_name"])
    return out


@router.patch("/lidar/objects3d/{object_3d_id}")
async def edit_object3d(object_3d_id: uuid.UUID, body: Cuboid3DEdit, db: AsyncSession = Depends(db_session)):
    o = await db.get(Object3D, object_3d_id)
    if o is None:
        raise HTTPException(404, "object_3d not found")
    if body.expected_version is not None and body.expected_version != o.version:
        raise HTTPException(409, f"stale write: expected version {body.expected_version}, have {o.version}")
    if body.class_id is not None:
        try:
            get_ontology().by_id(body.class_id)
        except Exception as exc:
            raise HTTPException(422, f"class_id {body.class_id} not in ontology") from exc
        o.class_id = body.class_id
    for field in ("dims", "yaw", "pitch", "roll"):
        v = getattr(body, field)
        if v is not None:
            setattr(o, field, v)
    if body.center is not None:
        o.center = body.center
    if body.attrs is not None:
        o.attrs = body.attrs
    if body.ground_snap:
        pc = await db.get(PointCloud, o.cloud_id)
        o.center = snap_to_ground(o.center, o.dims, await _ground_plane(db, pc))
    o.source = "human"
    o.state = "accepted"
    o.is_keyframe = True
    o.version += 1
    await db.flush()
    out = _serialize(o)
    await db.commit()
    log.info("objects3d.edit", id=str(object_3d_id), version=out["version"])
    return out


@router.delete("/lidar/objects3d/{object_3d_id}")
async def delete_object3d(object_3d_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    res = await db.execute(delete(Object3D).where(Object3D.object_3d_id == object_3d_id))
    await db.commit()
    if res.rowcount == 0:
        raise HTTPException(404, "object_3d not found")
    return {"deleted": str(object_3d_id)}


@router.post("/lidar/sessions/{session_id}/track3d")
async def run_tracking(session_id: uuid.UUID):
    """Track the session's 3D objects across frames into track_3d, linked to the 2D tracks, with a dynamic
    state per track (M-L2.2)."""
    from services.lidar.track3d import track_session
    return await track_session(session_id)


@router.get("/lidar/sessions/{session_id}/tracks3d")
async def list_tracks3d(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    onto = get_ontology()
    rows = (await db.execute(select(Track3D).where(Track3D.session_id == session_id)
                             .order_by(Track3D.first_ts_ns))).scalars().all()
    return {"session_id": str(session_id), "tracks": [
        {"track_3d_id": str(t.track_3d_id), "track_id": str(t.track_id) if t.track_id else None,
         "class_id": t.class_id, "class_name": onto.by_id(t.class_id).name, "first_ts_ns": t.first_ts_ns,
         "last_ts_ns": t.last_ts_ns, "dynamic_state": t.dynamic_state,
         "n_points": len((t.trajectory or {}).get("points", []))} for t in rows]}


@router.post("/lidar/tracks3d/{track_3d_id}/interpolate")
async def interpolate_track(track_3d_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Fill the cuboid pose between the human keyframes on a 3D track, writing interpolated object_3d for the
    clouds in the span that lack one (M-L2.2 keyframe interpolation)."""
    from services.lidar.track3d import interpolate_cuboids
    tr = await db.get(Track3D, track_3d_id)
    if tr is None:
        raise HTTPException(404, "track_3d not found")
    members = (await db.execute(
        select(Object3D, PointCloud.ts_ns).join(PointCloud, Object3D.cloud_id == PointCloud.cloud_id)
        .where(Object3D.track_3d_id == track_3d_id))).all()
    keyframes = [{"ts_ns": int(ts), "center": o.center, "dims": o.dims, "yaw": o.yaw}
                 for o, ts in members if o.is_keyframe or o.source == "human"]
    if len(keyframes) < 2:
        return {"track_3d_id": str(track_3d_id), "filled": 0, "reason": "need at least two keyframes"}
    keyframes.sort(key=lambda k: k["ts_ns"])
    have_ts = {int(ts) for _, ts in members}
    clouds = (await db.execute(
        select(PointCloud.cloud_id, PointCloud.ts_ns).where(
            PointCloud.session_id == tr.session_id, PointCloud.ts_ns > keyframes[0]["ts_ns"],
            PointCloud.ts_ns < keyframes[-1]["ts_ns"]).order_by(PointCloud.ts_ns))).all()
    targets = [(cid, int(ts)) for cid, ts in clouds if int(ts) not in have_ts]
    by_ts = {c["ts_ns"]: c for c in interpolate_cuboids(keyframes, [ts for _, ts in targets])}
    filled = 0
    for cid, ts in targets:
        c = by_ts.get(ts)
        if not c:
            continue
        frame = (await db.execute(select(Frame.frame_id).where(Frame.session_id == tr.session_id,
                 Frame.ts_ns == ts).limit(1))).scalar_one_or_none()
        db.add(Object3D(cloud_id=cid, frame_id=frame, track_3d_id=track_3d_id, class_id=tr.class_id,
                        center=c["center"], dims=c["dims"], yaw=c["yaw"], conf=0.7, box_source="lifted",
                        source="fused", state="auto_accept", is_keyframe=False, interp_source="linear"))
        filled += 1
    await db.commit()
    log.info("objects3d.interpolate", track=str(track_3d_id), filled=filled)
    return {"track_3d_id": str(track_3d_id), "filled": filled}


@router.post("/lidar/clouds/{cloud_id}/segment")
async def segment(cloud_id: uuid.UUID):
    """Segment a cloud into per-point semantic and instance labels (M-L2.3). PTv3 on the burst node, else the
    projected fallback locally; the low-confidence fraction is recorded for review."""
    from services.lidar.segment3d import segment_cloud
    return await segment_cloud(cloud_id)


@router.get("/lidar/clouds/{cloud_id}/segmentation")
async def get_segmentation(cloud_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    row = (await db.execute(select(PointSegmentation).where(PointSegmentation.cloud_id == cloud_id)
           .order_by(PointSegmentation.created_at.desc()).limit(1))).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "no segmentation for this cloud")
    return {"seg_id": str(row.seg_id), "cloud_id": str(cloud_id), "kind": row.kind, "method": row.method,
            "model_version": row.model_version, "n_points": row.n_points, "low_conf_frac": row.low_conf_frac,
            "labels_uri": row.labels_uri}


@router.get("/lidar/clouds/{cloud_id}/segmentation/points")
async def segmentation_points(cloud_id: uuid.UUID, max_points: int = Query(300000, alias="max", ge=1000),
                              db: AsyncSession = Depends(db_session)):
    """Packed Float32 [x, y, z, semantic_class] for the viewer's segmentation overlay, decimated with the
    labels aligned to the same points."""
    from services.lidar.segment3d import load_segmentation
    pc = await db.get(PointCloud, cloud_id)
    if pc is None:
        raise HTTPException(404, "cloud not found")
    row = (await db.execute(select(PointSegmentation).where(PointSegmentation.cloud_id == cloud_id)
           .order_by(PointSegmentation.created_at.desc()).limit(1))).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "no segmentation for this cloud")
    cloud = load_cloud(pc.cloud_uri)
    labels = load_segmentation(row.labels_uri)
    sem = labels["semantic"].astype(np.float32)
    n = cloud.n
    idx = np.arange(n)
    if n > max_points:
        idx = np.sort(np.random.default_rng(pc.ts_ns % (2**31)).choice(n, max_points, replace=False))
    packed = np.empty((len(idx), 4), dtype=np.float32)
    packed[:, :3] = cloud.xyz[idx]
    packed[:, 3] = sem[idx]
    classes = sorted({int(x) for x in np.unique(labels["semantic"]) if x != -1})
    return Response(content=packed.tobytes(), media_type="application/octet-stream",
                    headers={"X-Point-Count": str(len(idx)), "X-Classes": ",".join(map(str, classes)),
                             "X-Low-Conf-Frac": f"{row.low_conf_frac:.4f}",
                             "Access-Control-Expose-Headers": "X-Point-Count,X-Classes,X-Low-Conf-Frac"})


@router.get("/lidar/objects3d/{object_3d_id}/projection")
async def project_object3d(object_3d_id: uuid.UUID, cam_id: str = "cam_f", w: int = 1280, h: int = 960,
                           db: AsyncSession = Depends(db_session)):
    o = await db.get(Object3D, object_3d_id)
    if o is None:
        raise HTTPException(404, "object_3d not found")
    proj = project_cuboid(o.center, o.dims, o.yaw, cam_id, w, h, o.pitch, o.roll)
    return {"object_3d_id": str(object_3d_id), "cam_id": cam_id, **proj}


@router.post("/lidar/objects3d/{object_3d_id}/consistency")
async def object3d_consistency(object_3d_id: uuid.UUID):
    """Cross-sensor 2D-3D consistency: reproject the cuboid into every camera that has a 2D detection of the
    same object and score the box agreement. A gross mismatch writes a quality_flag_3d that routes the cuboid
    to review, the same loop as the geometric checks."""
    from services.lidar.quality3d.checker import check_object_consistency
    res = await check_object_consistency(object_3d_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


class AggLabelIn(BaseModel):
    center: list[float]                       # [x, y, z] in the aggregated-map frame
    dims: list[float]                         # [L, W, H] metres, the one size used across the whole track
    yaw: float = 0.0
    class_id: int


@router.post("/lidar/aggregate/{agg_id}/label")
async def aggregate_label(agg_id: uuid.UUID, body: AggLabelIn):
    """One-shot 4D label: a cuboid drawn once in the aggregated scene propagates to every clip frame as a 3D
    track with one consistent size, each box transformed into that frame's ego pose. Routed to review."""
    from services.lidar.aggregate.label_propagate import propagate_aggregate_label
    res = await propagate_aggregate_label(agg_id, body.center, body.dims, body.yaw, body.class_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


# ---- M-L2.4: linked identity, properties, correction ----
class BatchCorrectIn(BaseModel):
    object_3d_ids: list[uuid.UUID]
    class_id: int | None = None
    dims: list[float] | None = None


@router.post("/lidar/clouds/{cloud_id}/link")
async def link_cloud_objects(cloud_id: uuid.UUID):
    """Link unlinked 3D cuboids on a cloud to the 2D objects by projection IoU (the unifying identity)."""
    from services.lidar.link import link_cloud
    return await link_cloud(cloud_id)


@router.get("/lidar/objects3d/{object_3d_id}/linked")
async def object3d_linked(object_3d_id: uuid.UUID):
    """The 2D object and the per-camera projections of a 3D object: select it in the cloud, see it in every
    synchronized camera view."""
    from services.lidar.link import linked_views
    res = await linked_views(object_3d_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.get("/lidar/objects2d/{object_id}/linked3d")
async def object2d_linked3d(object_id: uuid.UUID):
    """The reverse: the 3D cuboid a 2D object belongs to, and its per-camera projections."""
    from services.lidar.link import linked_from_2d
    return await linked_from_2d(object_id)


@router.post("/lidar/objects3d/{object_3d_id}/properties")
async def compute_properties_endpoint(object_3d_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Auto-compute distance, heading, velocity, acceleration, and occlusion for a 3D object and store them."""
    from db.models import Track3D
    from services.lidar.link import compute_object_properties
    from services.lidar.segment3d import points_in_cuboid
    o = await db.get(Object3D, object_3d_id)
    if o is None:
        raise HTTPException(404, "object_3d not found")
    pc = await db.get(PointCloud, o.cloud_id)
    frame = (await db.execute(select(Frame).where(Frame.session_id == pc.session_id,
             Frame.ts_ns == pc.ts_ns).limit(1))).scalar_one_or_none()
    traj = None
    if o.track_3d_id:
        tr = await db.get(Track3D, o.track_3d_id)
        traj = (tr.trajectory or {}).get("points") if tr else None
    w, h, cam = (frame.width, frame.height, frame.cam_id) if frame else (1280, 960, "cam_f")
    proj = project_cuboid(o.center, o.dims, o.yaw, cam, w, h)
    in_image_frac = sum(1 for x in proj["in_image"] if x) / 8.0
    cloud = load_cloud(pc.cloud_uri)
    nb = int(points_in_cuboid(cloud.xyz, {"center": o.center, "dims": o.dims, "yaw": o.yaw}).sum())
    props = compute_object_properties(o.center, o.dims, o.yaw, trajectory=traj,
                                      ego_speed=float(frame.ego_speed or 0.0) if frame else 0.0,
                                      in_image_frac=in_image_frac, points_in_box=nb)
    o.attrs = {**(o.attrs or {}), **props}
    await db.commit()
    return {"object_3d_id": str(object_3d_id), "properties": props}


@router.get("/lidar/objects3d/{object_3d_id}/similar")
async def similar_objects3d(object_3d_id: uuid.UUID, k: int = Query(10, ge=1, le=100)):
    """Objects similar to a 3D object (class plus dimensions), the candidates for a batch correction."""
    from services.lidar.link import find_similar
    res = await find_similar(object_3d_id, k=k)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.post("/lidar/objects3d/batch_correct")
async def batch_correct_objects3d(body: BatchCorrectIn):
    """Apply a class or dimension correction to a batch of 3D objects as a human edit."""
    from services.lidar.link import batch_correct
    return await batch_correct(body.object_3d_ids, class_id=body.class_id, dims=body.dims)
