"""API for the sensors tools, agent E: #10 radar and event cameras, #15 scene reconstruction, #16 synthetic variants.

Mounted under /api; every path here starts with /tools/sensors/.

#10 Radar and events (/sensors)
- GET  overview                                 sessions with radar sweeps, and every event stream
- POST radar/import                             queue `sensors_radar_import` (nuScenes / MCAP / PCD) (reviewer)
- GET  radar/sessions/{id}/sweeps               the session's sweeps and its camera frames to overlay them on
- GET  radar/frames/{frame_id}/overlay          returns near a frame's instant, projected into it
- GET  radar/sessions/{id}/bev?ts_ns=           returns from above in the ego frame, with the 3D boxes there
- POST radar/sessions/{id}/fuse                 queue `sensors_radar_fuse`: Doppler into Track3D speed (reviewer)
- GET  radar/sessions/{id}/fusion               what the last fusion wrote, per track
- POST events/import                            queue `sensors_events_import` (reviewer)
- GET  events/streams/{id}/preview.jpg          one accumulated image at a time, written nowhere
- POST events/streams/{id}/accumulate           queue `sensors_events_accumulate`: frames to label (reviewer)
- GET  events/streams/{id}/frames               the frames an accumulation wrote

#15 Reconstruction (/reconstruct)
- GET  reconstruct/sessions                     sessions with ego poses and LiDAR, their cameras
- POST reconstruct                              a reconstruction row + its job (gsplat: reviewer, GPU, held in
                                                office hours; pointmap: annotator, CPU)
- GET  reconstructions[?session_id=]            list; GET reconstructions/{id} one, with its views
- GET  reconstructions/{id}/views/{n}.jpg       a rendered view; preview.bin the cloud; splats.ply the splats
- POST reconstructions/{id}/place               a cuboid in the reconstruction -> the label-once job (agent C's
                                                `scene3d_label_once`) puts it on every scan and camera frame
- GET  reconstructions/{id}/placed              what was placed from this reconstruction
- DELETE reconstructions/{id}                   the row and its blobs (reviewer)

#16 Synthetic variants (/synthetic)
- GET  variants/perturbations                   the photometric set with default strengths
- GET  variants/hard-frames                     candidate source frames, hardest first, with eligibility
- GET  variants/preview.jpg?frame_id&perturbation&strength   one variant rendered, written nowhere
- POST variants                                 queue `sensors_variants` (reviewer)
- GET  variants/runs, variants/summary          what has been made
- POST variants/runs/{job_id}/revert            delete exactly what a run made (reviewer)
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.origin import REAL
from core.storage import get_object_store
from db.models import (
    EgoPose,
    EventStream,
    Frame,
    PointCloud,
    RadarSweep,
    Reconstruction,
    ToolJob,
    Track3D,
)
from db.models import Session as DbSession
from services.api.deps import db_session, require_role, require_user
from services.tools.registry import enqueue

log = get_logger("tools_sensors")

router = APIRouter()

_BIN = "application/octet-stream"


def _job(job) -> dict:
    return {"job_id": str(job.job_id), "kind": job.kind, "status": job.status,
            "session_id": str(job.session_id) if job.session_id else None}


def _check_source(uri: str) -> None:
    from services.imports.run import SourceNotAllowed, check_request_source

    try:
        check_request_source(uri)
    except SourceNotAllowed as exc:
        raise HTTPException(422, str(exc)) from exc


def _jpeg(rgb: np.ndarray, quality: int = 88) -> bytes:
    import cv2

    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "could not encode the image")
    return enc.tobytes()


def _img(data: bytes) -> Response:
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


# ============================================================ #10 radar and event cameras


@router.get("/tools/sensors/overview", dependencies=[Depends(require_user)])
async def overview(db: AsyncSession = Depends(db_session)):
    """Every session with radar, every event stream: what the radar and events page opens on."""
    radar = (await db.execute(
        select(RadarSweep.session_id, func.count(), func.count(func.distinct(RadarSweep.sensor)),
               func.min(RadarSweep.ts_ns), func.max(RadarSweep.ts_ns), func.sum(RadarSweep.n_points))
        .group_by(RadarSweep.session_id))).all()
    sids = [r[0] for r in radar]
    names = dict((await db.execute(select(DbSession.session_id, func.coalesce(DbSession.route, DbSession.vehicle_id))
                                   .where(DbSession.session_id.in_(sids)))).all()) if sids else {}
    streams = (await db.execute(select(EventStream, func.coalesce(DbSession.route, DbSession.vehicle_id)).join(
        DbSession, DbSession.session_id == EventStream.session_id).order_by(EventStream.created_at.desc())
        .limit(200))).all()
    return {
        "radar": [{"session_id": str(s), "name": names.get(s), "sweeps": int(n), "sensors": int(ns),
                   "first_ts_ns": int(a), "last_ts_ns": int(b), "points": int(p or 0)}
                  for s, n, ns, a, b, p in radar],
        "events": [{"stream_id": str(st.stream_id), "session_id": str(st.session_id), "session_name": nm,
                    "sensor": st.sensor, "source": st.source, "width": st.width, "height": st.height,
                    "n_events": int(st.n_events), "t0_ns": int(st.t0_ns), "t1_ns": int(st.t1_ns),
                    "stats": st.stats or {}, "accumulation": st.accumulation,
                    "created_at": st.created_at.isoformat() if st.created_at else None}
                   for st, nm in streams],
    }


class RadarImportIn(BaseModel):
    session_id: uuid.UUID
    source_uri: str = Field(min_length=3, max_length=2048)
    format: Literal["nuscenes", "mcap", "pcd"] = "nuscenes"
    channels: list[str] | None = None
    keyframes_only: bool = False
    filter_points: bool = True
    topics: list[str] | None = None
    extrinsics: dict[str, list[list[float]]] | None = None
    sensor_to_ego: list[list[float]] | None = None
    time_source: Literal["log", "header"] = "log"
    sensor: str | None = Field(default=None, max_length=48)
    ts_ns: int | None = None


@router.post("/tools/sensors/radar/import")
async def radar_import(body: RadarImportIn, db: AsyncSession = Depends(db_session),
                       user=Depends(require_role("reviewer"))):
    _check_source(body.source_uri)
    if await db.get(DbSession, body.session_id) is None:
        raise HTTPException(404, "session not found")
    if body.format == "pcd" and body.ts_ns is None:
        raise HTTPException(422, "a single .pcd needs ts_ns: the file does not say when it was captured")
    job = await enqueue(db, "sensors_radar_import", body.model_dump(mode="json"), session_id=body.session_id,
                        user_id=user.user_id)
    return _job(job)


@router.get("/tools/sensors/radar/sessions/{session_id}/sweeps", dependencies=[Depends(require_user)])
async def radar_sweeps(session_id: uuid.UUID, db: AsyncSession = Depends(db_session),
                       limit: int = Query(2000, ge=1, le=20000)):
    """The sweeps (time, sensor, size) and the session's real camera frames, each with the nearest sweep's
    time offset, so the page can step through frames that actually have radar."""
    rows = (await db.execute(select(RadarSweep).where(RadarSweep.session_id == session_id)
                             .order_by(RadarSweep.ts_ns).limit(limit))).scalars().all()
    frames = (await db.execute(select(Frame.frame_id, Frame.ts_ns, Frame.cam_id).where(
        Frame.session_id == session_id, Frame.origin == REAL).order_by(Frame.ts_ns).limit(5000))).all()
    ts = np.array([r.ts_ns for r in rows], dtype=np.int64)
    out_frames = []
    for fid, fts, cam in frames:
        if len(ts):
            i = int(np.clip(np.searchsorted(ts, fts), 0, len(ts) - 1))
            j = min((i - 1, i), key=lambda k: abs(int(ts[max(k, 0)]) - fts))
            dt = (int(ts[max(j, 0)]) - fts) / 1e6
        else:
            dt = None
        out_frames.append({"frame_id": str(fid), "ts_ns": int(fts), "cam_id": cam,
                           "radar_dt_ms": round(dt, 1) if dt is not None else None})
    return {"session_id": str(session_id),
            "sweeps": [{"sweep_id": str(r.sweep_id), "ts_ns": r.ts_ns, "sensor": r.sensor, "source": r.source,
                        "n_points": r.n_points, "stats": r.stats or {}} for r in rows],
            "sensors": sorted({r.sensor for r in rows}), "frames": out_frames}


@router.get("/tools/sensors/radar/frames/{frame_id}/overlay", dependencies=[Depends(require_user)])
async def radar_overlay(frame_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.sensors.radar import frame_overlay

    try:
        return await frame_overlay(db, frame_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/tools/sensors/radar/sessions/{session_id}/bev", dependencies=[Depends(require_user)])
async def radar_bev(session_id: uuid.UUID, ts_ns: int = Query(...), db: AsyncSession = Depends(db_session)):
    from services.sensors.radar import bev

    return await bev(db, session_id, ts_ns)


class FuseIn(BaseModel):
    tol_mps: float = Field(default=3.0, gt=0, le=30)
    apply: bool = True


@router.post("/tools/sensors/radar/sessions/{session_id}/fuse")
async def radar_fuse(session_id: uuid.UUID, body: FuseIn, db: AsyncSession = Depends(db_session),
                     user=Depends(require_role("reviewer"))):
    n = (await db.execute(select(func.count()).select_from(RadarSweep).where(
        RadarSweep.session_id == session_id))).scalar_one()
    if not n:
        raise HTTPException(409, "this session has no radar sweeps; import radar first")
    job = await enqueue(db, "sensors_radar_fuse", {"session_id": str(session_id), **body.model_dump()},
                        session_id=session_id, user_id=user.user_id)
    return _job(job)


@router.get("/tools/sensors/radar/sessions/{session_id}/fusion", dependencies=[Depends(require_user)])
async def radar_fusion(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    rows = (await db.execute(select(Track3D).where(Track3D.session_id == session_id)
                             .order_by(Track3D.first_ts_ns))).scalars().all()
    out = []
    for t in rows:
        rs = (t.trajectory or {}).get("radar_speed")
        if not rs:
            continue
        try:
            cname = onto.by_id(t.class_id).name
        except Exception:  # noqa: BLE001
            cname = str(t.class_id)
        out.append({"track_3d_id": str(t.track_3d_id), "class_name": cname, "first_ts_ns": t.first_ts_ns,
                    "last_ts_ns": t.last_ts_ns, **rs})
    total = (await db.execute(select(func.count()).select_from(Track3D).where(
        Track3D.session_id == session_id))).scalar_one()
    return {"session_id": str(session_id), "tracks_3d": int(total), "fused_tracks": out}


class EventsImportIn(BaseModel):
    source_uri: str = Field(min_length=3, max_length=2048)
    format: Literal["npy", "h5", "mcap"] | None = None
    session_id: uuid.UUID | None = None
    vehicle_id: str | None = Field(default=None, max_length=64)
    city: str | None = Field(default=None, max_length=64)
    sensor: str = Field(default="event_front", min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")
    width: int | None = Field(default=None, ge=1, le=8192)
    height: int | None = Field(default=None, ge=1, le=8192)
    columns: str = "x,y,t,p"
    time_unit: Literal["ns", "us", "ms", "s"] = "us"
    topics: list[str] | None = None
    time_offset_ns: int = 0


@router.post("/tools/sensors/events/import")
async def events_import(body: EventsImportIn, db: AsyncSession = Depends(db_session),
                        user=Depends(require_role("reviewer"))):
    _check_source(body.source_uri)
    if body.session_id is not None and await db.get(DbSession, body.session_id) is None:
        raise HTTPException(404, "session not found")
    params = body.model_dump(mode="json")
    if body.session_id is None:
        params["new_session"] = {"vehicle_id": body.vehicle_id or "EVENT-CAM", "city": body.city}
    job = await enqueue(db, "sensors_events_import", params, session_id=body.session_id, user_id=user.user_id)
    return _job(job)


# A few decoded streams kept in memory so dragging the preview slider does not re-read the npz each time.
_EVENTS_CACHE: OrderedDict[str, object] = OrderedDict()
_EVENTS_CACHE_MAX = 3


def _load_events(uri: str):
    from services.sensors.events import Events

    if uri in _EVENTS_CACHE:
        _EVENTS_CACHE.move_to_end(uri)
        return _EVENTS_CACHE[uri]
    ev = Events.from_npz_bytes(get_object_store().get_bytes(uri))
    _EVENTS_CACHE[uri] = ev
    while len(_EVENTS_CACHE) > _EVENTS_CACHE_MAX:
        _EVENTS_CACHE.popitem(last=False)
    return ev


@router.get("/tools/sensors/events/streams/{stream_id}/preview.jpg", dependencies=[Depends(require_user)])
async def events_preview(stream_id: uuid.UUID, t_ns: int | None = None,
                         mode: Literal["count", "time_surface"] = "count",
                         window_ms: float = Query(33.0, gt=0.01, le=5000), tau_ms: float = Query(30.0, gt=0.01, le=5000),
                         db: AsyncSession = Depends(db_session)):
    """One window of the stream accumulated as the frames would be, ending at `t_ns` (default: the first window)."""
    from services.sensors.events import count_image, time_surface

    st = await db.get(EventStream, stream_id)
    if st is None:
        raise HTTPException(404, "event stream not found")
    ev = await asyncio.to_thread(_load_events, st.uri)
    win = int(window_ms * 1e6)
    end = int(t_ns) if t_ns is not None else int(ev.t[0]) + win
    i, j = (int(x) for x in np.searchsorted(ev.t, [end - win, end], side="left"))
    img = count_image(ev, i, j) if mode == "count" else time_surface(ev, i, j, end, tau_ms * 1e6)
    r = _img(_jpeg(img, 92))
    r.headers["X-Events"] = str(j - i)
    r.headers["Access-Control-Expose-Headers"] = "X-Events"
    return r


class AccumulateIn(BaseModel):
    mode: Literal["count", "time_surface"] = "count"
    window_ms: float = Field(default=33.0, gt=0.01, le=5000)
    tau_ms: float = Field(default=30.0, gt=0.01, le=5000)
    max_frames: int = Field(default=600, ge=1, le=5000)
    replace: bool = True


@router.post("/tools/sensors/events/streams/{stream_id}/accumulate")
async def events_accumulate(stream_id: uuid.UUID, body: AccumulateIn, db: AsyncSession = Depends(db_session),
                            user=Depends(require_role("reviewer"))):
    st = await db.get(EventStream, stream_id)
    if st is None:
        raise HTTPException(404, "event stream not found")
    job = await enqueue(db, "sensors_events_accumulate", {"stream_id": str(stream_id), **body.model_dump()},
                        session_id=st.session_id, user_id=user.user_id)
    return _job(job)


@router.get("/tools/sensors/events/streams/{stream_id}/frames", dependencies=[Depends(require_user)])
async def events_frames(stream_id: uuid.UUID, db: AsyncSession = Depends(db_session),
                        limit: int = Query(500, ge=1, le=5000)):
    from db.models import Object

    st = await db.get(EventStream, stream_id)
    if st is None:
        raise HTTPException(404, "event stream not found")
    n_obj = func.count(Object.object_id)
    rows = (await db.execute(
        select(Frame.frame_id, Frame.ts_ns, Frame.tags, n_obj).join(Object, Object.frame_id == Frame.frame_id,
                                                                    isouter=True)
        .where(Frame.session_id == st.session_id, Frame.tags.contains([f"stream:{stream_id}"]))
        .group_by(Frame.frame_id).order_by(Frame.ts_ns).limit(limit))).all()
    return {"stream_id": str(stream_id), "session_id": str(st.session_id), "accumulation": st.accumulation,
            "frames": [{"frame_id": str(f), "ts_ns": int(t), "tags": tags or [], "objects": int(n)}
                       for f, t, tags, n in rows]}


# ============================================================ #15 reconstruction


@router.get("/tools/sensors/reconstruct/sessions", dependencies=[Depends(require_user)])
async def reconstruct_sessions(db: AsyncSession = Depends(db_session)):
    """Sessions a reconstruction can be built from: ego poses and LiDAR (or pseudo-LiDAR) scans, with cameras."""
    poses = select(EgoPose.session_id, func.count().label("n")).group_by(EgoPose.session_id).subquery()
    clouds = select(PointCloud.session_id, func.count().label("n")).group_by(PointCloud.session_id).subquery()
    rows = (await db.execute(
        select(DbSession.session_id, func.coalesce(DbSession.route, DbSession.vehicle_id), poses.c.n, clouds.c.n)
        .join(poses, poses.c.session_id == DbSession.session_id)
        .join(clouds, clouds.c.session_id == DbSession.session_id)
        .where(DbSession.origin == REAL).order_by(clouds.c.n.desc()).limit(200))).all()
    from services.sensors.reconstruct.window import session_cameras

    out = []
    for sid, name, n_poses, n_clouds in rows:
        cams = await session_cameras(db, sid)
        if cams:
            out.append({"session_id": str(sid), "name": name, "poses": int(n_poses), "clouds": int(n_clouds),
                        "cameras": cams})
    return {"sessions": out, "gsplat": _gsplat_available()}


def _gsplat_available() -> dict:
    try:
        import importlib.metadata as md

        return {"installed": True, "version": md.version("gsplat")}
    except Exception:  # noqa: BLE001
        return {"installed": False, "version": None}


class ReconstructIn(BaseModel):
    session_id: uuid.UUID
    method: Literal["gsplat", "pointmap"] = "gsplat"
    cam_id: str | None = Field(default=None, max_length=32)
    start_ts_ns: int | None = None
    n_frames: int = Field(default=50, ge=3, le=200)
    stride: int = Field(default=1, ge=1, le=20)
    iters: int = Field(default=7000, ge=100, le=30000)
    max_width: int = Field(default=1280, ge=320, le=3840)


def _recon_dict(r: Reconstruction, *, views: bool = False) -> dict:
    out = {"recon_id": str(r.recon_id), "session_id": str(r.session_id), "method": r.method, "status": r.status,
           "cam_id": r.cam_id, "first_ts_ns": r.first_ts_ns, "last_ts_ns": r.last_ts_ns, "params": r.params or {},
           "metrics": r.metrics or {}, "error": r.error, "job_id": str(r.job_id) if r.job_id else None,
           "created_at": r.created_at.isoformat() if r.created_at else None,
           "finished_at": r.finished_at.isoformat() if r.finished_at else None,
           "has_ply": bool((r.artifacts or {}).get("ply")),
           "preview_points": (r.artifacts or {}).get("preview_points"), "n_views": len(r.views or [])}
    if views:
        out["views"] = [{k: v for k, v in vw.items() if k != "uri"} for vw in (r.views or [])]
    return out


@router.post("/tools/sensors/reconstruct")
async def reconstruct(body: ReconstructIn, db: AsyncSession = Depends(db_session),
                      user=Depends(require_role("annotator"))):
    """A reconstruction of one window. Training splats spends the GPU on a session window, so it is a reviewer
    action and waits out office hours; the point map is CPU work any annotator may start."""
    from services.api.deps import role_rank

    if body.method == "gsplat" and role_rank(user.role) < role_rank("reviewer"):
        raise HTTPException(403, "training a reconstruction needs the reviewer role; the point map does not")
    if body.method == "gsplat" and not _gsplat_available()["installed"]:
        raise HTTPException(409, "gsplat is not installed on this server; build a point map instead")
    if await db.get(DbSession, body.session_id) is None:
        raise HTTPException(404, "session not found")
    rec = Reconstruction(session_id=body.session_id, method=body.method, cam_id=body.cam_id,
                         params=body.model_dump(mode="json", exclude={"session_id", "method"}),
                         created_by=user.user_id)
    db.add(rec)
    await db.commit()
    kind = "sensors_reconstruct" if body.method == "gsplat" else "sensors_pointmap"
    job = await enqueue(db, kind, {"recon_id": str(rec.recon_id)}, session_id=body.session_id,
                        user_id=user.user_id)
    rec = await db.get(Reconstruction, rec.recon_id)
    rec.job_id = job.job_id
    await db.commit()
    return {"reconstruction": _recon_dict(rec), "job": _job(job)}


@router.get("/tools/sensors/reconstructions", dependencies=[Depends(require_user)])
async def reconstructions(session_id: uuid.UUID | None = None, db: AsyncSession = Depends(db_session),
                          limit: int = Query(50, ge=1, le=500)):
    q = select(Reconstruction).order_by(Reconstruction.created_at.desc()).limit(limit)
    if session_id is not None:
        q = q.where(Reconstruction.session_id == session_id)
    return {"reconstructions": [_recon_dict(r) for r in (await db.execute(q)).scalars().all()]}


async def _recon(db: AsyncSession, recon_id: uuid.UUID) -> Reconstruction:
    r = await db.get(Reconstruction, recon_id)
    if r is None:
        raise HTTPException(404, "reconstruction not found")
    return r


@router.get("/tools/sensors/reconstructions/{recon_id}", dependencies=[Depends(require_user)])
async def reconstruction(recon_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    r = await _recon(db, recon_id)
    out = _recon_dict(r, views=True)
    frames = [v["frame_id"] for v in out["views"] if v.get("kind") == "camera"]
    out["frames"] = frames
    return out


@router.get("/tools/sensors/reconstructions/{recon_id}/views/{index}.jpg", dependencies=[Depends(require_user)])
async def reconstruction_view(recon_id: uuid.UUID, index: int, db: AsyncSession = Depends(db_session)):
    r = await _recon(db, recon_id)
    views = r.views or []
    if not 0 <= index < len(views):
        raise HTTPException(404, "no such view")
    try:
        data = await asyncio.to_thread(get_object_store().get_bytes, views[index]["uri"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "view image unavailable") from exc
    return _img(data)


@router.get("/tools/sensors/reconstructions/{recon_id}/preview.bin", dependencies=[Depends(require_user)])
async def reconstruction_preview(recon_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Float32 [x, y, z, r, g, b] per point, x/y/z minus the anchor in X-Anchor (so the browser's float32 stays
    exact); add the anchor back for world coordinates."""
    r = await _recon(db, recon_id)
    uri = (r.artifacts or {}).get("preview")
    if not uri:
        raise HTTPException(404, "this reconstruction has no preview cloud yet")
    data = await asyncio.to_thread(get_object_store().get_bytes, uri)
    arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 6).copy()
    anchor = arr[:, :3].astype(np.float64).mean(axis=0) if len(arr) else np.zeros(3)
    anchor = np.round(anchor, 1)
    arr[:, :3] = (arr[:, :3].astype(np.float64) - anchor).astype(np.float32)
    headers = {"X-Count": str(len(arr)), "X-Anchor": ",".join(f"{v:.1f}" for v in anchor)}
    headers["Access-Control-Expose-Headers"] = ",".join(headers)
    return Response(content=arr.tobytes(), media_type=_BIN, headers=headers)


@router.get("/tools/sensors/reconstructions/{recon_id}/splats.ply", dependencies=[Depends(require_user)])
async def reconstruction_ply(recon_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    r = await _recon(db, recon_id)
    uri = (r.artifacts or {}).get("ply")
    if not uri:
        raise HTTPException(404, "only a trained reconstruction has splats")
    data = await asyncio.to_thread(get_object_store().get_bytes, uri)
    return Response(content=data, media_type=_BIN,
                    headers={"Content-Disposition": f'attachment; filename="recon-{recon_id}.ply"'})


class PlaceIn(BaseModel):
    center: list[float] = Field(min_length=3, max_length=3)       # world (session scene) frame, metres
    dims: list[float] = Field(min_length=3, max_length=3)         # L, W, H
    yaw: float = 0.0
    class_id: int
    ground_snap: bool = True
    max_range_m: float = Field(default=60.0, gt=1, le=200)


@router.post("/tools/sensors/reconstructions/{recon_id}/place")
async def reconstruction_place(recon_id: uuid.UUID, body: PlaceIn, db: AsyncSession = Depends(db_session),
                               user=Depends(require_role("annotator"))):
    """One cuboid placed in the reconstruction goes to every scan in range and every camera frame at those
    scans, through the label-once job: the reconstruction and the scans share the session scene frame."""
    from services.autolabel.ontology import get_ontology

    r = await _recon(db, recon_id)
    try:
        get_ontology().by_id(body.class_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"class {body.class_id} is not in the ontology") from exc
    if min(body.dims) <= 0.05 or max(body.dims) > 30:
        raise HTTPException(422, "dims must be between 5 cm and 30 m")
    if not all(np.isfinite(body.center)):
        raise HTTPException(422, "center must be finite")
    try:
        job = await enqueue(db, "scene3d_label_once",
                            {"session_id": str(r.session_id), "center": [float(v) for v in body.center],
                             "dims": [float(v) for v in body.dims], "yaw": float(body.yaw),
                             "class_id": body.class_id, "ground_snap": body.ground_snap,
                             "max_range_m": body.max_range_m, "project": True, "recon_id": str(recon_id)},
                            session_id=r.session_id, user_id=user.user_id)
    except KeyError as exc:
        raise HTTPException(409, "the label-once propagation is not available on this server") from exc
    return _job(job)


@router.get("/tools/sensors/reconstructions/{recon_id}/placed", dependencies=[Depends(require_user)])
async def reconstruction_placed(recon_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    r = await _recon(db, recon_id)
    jobs = (await db.execute(select(ToolJob).where(
        ToolJob.kind == "scene3d_label_once", ToolJob.session_id == r.session_id,
        ToolJob.params["recon_id"].astext == str(recon_id)).order_by(ToolJob.created_at.desc()).limit(100))
    ).scalars().all()
    return {"placed": [{"job_id": str(j.job_id), "status": j.status, "progress": j.progress, "error": j.error,
                        "center": (j.params or {}).get("center"), "dims": (j.params or {}).get("dims"),
                        "yaw": (j.params or {}).get("yaw"), "class_id": (j.params or {}).get("class_id"),
                        "result": j.result, "created_at": j.created_at.isoformat() if j.created_at else None}
                       for j in jobs]}


@router.delete("/tools/sensors/reconstructions/{recon_id}")
async def reconstruction_delete(recon_id: uuid.UUID, db: AsyncSession = Depends(db_session),
                                user=Depends(require_role("reviewer"))):
    r = await _recon(db, recon_id)
    if r.status in ("pending", "running"):
        raise HTTPException(409, "cancel the job before deleting the reconstruction")
    store = get_object_store()
    uris = [v for k, v in (r.artifacts or {}).items() if isinstance(v, str) and k != "preview_points"]
    uris += [v.get("uri") for v in (r.views or []) if v.get("uri")]
    for u in uris:
        await asyncio.to_thread(store.remove, u)
    await db.delete(r)
    await db.commit()
    return {"deleted": str(recon_id), "blobs": len(uris)}


# ============================================================ #16 synthetic variants


@router.get("/tools/sensors/variants/perturbations", dependencies=[Depends(require_user)])
async def variant_perturbations():
    from services.verdyx.perturb import DEFAULT_STRENGTHS, PHOTOMETRIC

    return {"perturbations": [{"name": n, "default_strength": DEFAULT_STRENGTHS[n]} for n in PHOTOMETRIC]}


@router.get("/tools/sensors/variants/hard-frames", dependencies=[Depends(require_user)])
async def variant_hard_frames(source: Literal["errors", "corrected", "session"] = "errors",
                              session_id: uuid.UUID | None = None, limit: int = Query(60, ge=1, le=500),
                              include_review: bool = False, db: AsyncSession = Depends(db_session)):
    from services.synth.variants import hard_frames

    try:
        return {"frames": await hard_frames(db, source=source, session_id=session_id, limit=limit,
                                            include_review=include_review)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/tools/sensors/variants/preview.jpg", dependencies=[Depends(require_user)])
async def variant_preview(frame_id: uuid.UUID, perturbation: str, strength: float | None = Query(None, ge=0, le=1),
                          seed: int = 7, max_width: int = Query(960, ge=160, le=3840),
                          db: AsyncSession = Depends(db_session)):
    """One variant rendered in the request and written nowhere, to judge a strength before making any."""
    import cv2

    from services.synth.variants import check_perturbations
    from services.verdyx.perturb import apply

    try:
        check_perturbations([perturbation])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    fr = await db.get(Frame, frame_id)
    if fr is None:
        raise HTTPException(404, "frame not found")
    data = await asyncio.to_thread(get_object_store().get_bytes, fr.img_uri)

    def _do() -> bytes:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("frame image could not be decoded")
        if img.shape[1] > max_width:
            s = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        out = apply(perturbation, img, strength=strength, seed=seed)
        return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    try:
        return _img(await asyncio.to_thread(_do))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class VariantsIn(BaseModel):
    frame_ids: list[uuid.UUID] = Field(min_length=1, max_length=500)
    perturbations: list[str] = Field(min_length=1, max_length=5)
    strengths: dict[str, float] = Field(default_factory=dict)
    seed: int = 7
    include_review: bool = False


class EligibilityIn(BaseModel):
    frame_ids: list[uuid.UUID] = Field(min_length=1, max_length=500)
    include_review: bool = False


@router.post("/tools/sensors/variants/check")
async def variant_check(body: EligibilityIn, db: AsyncSession = Depends(db_session),
                        _user=Depends(require_role("annotator"))):
    from services.synth.variants import eligibility

    return await eligibility(db, body.frame_ids, include_review=body.include_review)


@router.post("/tools/sensors/variants")
async def variants_build(body: VariantsIn, db: AsyncSession = Depends(db_session),
                         user=Depends(require_role("reviewer"))):
    from services.synth.variants import check_perturbations, eligibility

    try:
        names = check_perturbations(body.perturbations)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if any(not 0.0 <= v <= 1.0 for v in body.strengths.values()):
        raise HTTPException(422, "strengths are between 0 and 1")
    elig = await eligibility(db, body.frame_ids, include_review=body.include_review)
    if not elig["eligible"]:
        raise HTTPException(409, {"message": "no selected frame can be varied", "refused": elig["refused"]})
    job = await enqueue(db, "sensors_variants",
                        {"frame_ids": elig["eligible"], "perturbations": names, "strengths": body.strengths,
                         "seed": body.seed, "include_review": body.include_review}, user_id=user.user_id)
    return {**_job(job), "eligible": len(elig["eligible"]), "refused": elig["refused"],
            "variants": len(elig["eligible"]) * len(names)}


@router.get("/tools/sensors/variants/runs", dependencies=[Depends(require_user)])
async def variant_runs(db: AsyncSession = Depends(db_session), limit: int = Query(30, ge=1, le=200)):
    jobs = (await db.execute(select(ToolJob).where(ToolJob.kind == "sensors_variants")
                             .order_by(ToolJob.created_at.desc()).limit(limit))).scalars().all()
    out = []
    for j in jobs:
        res = dict(j.result or {})
        out.append({"job_id": str(j.job_id), "status": j.status, "progress": j.progress, "stage": j.stage,
                    "error": j.error, "perturbations": (j.params or {}).get("perturbations"),
                    "sources": len((j.params or {}).get("frame_ids") or []), "frames": res.get("frames"),
                    "objects": res.get("objects"), "session_id": res.get("session_id"),
                    "per_perturbation": res.get("per_perturbation"), "reverted": res.get("reverted"),
                    "first_frame_id": (res.get("frame_ids") or [None])[0],
                    "created_at": j.created_at.isoformat() if j.created_at else None})
    return {"runs": out}


@router.get("/tools/sensors/variants/summary", dependencies=[Depends(require_user)])
async def variant_summary_route(db: AsyncSession = Depends(db_session)):
    from services.synth.variants import variant_summary

    return await variant_summary(db)


@router.post("/tools/sensors/variants/runs/{job_id}/revert")
async def variant_revert(job_id: uuid.UUID, db: AsyncSession = Depends(db_session),
                         user=Depends(require_role("reviewer"))):
    from services.synth.variants import revert_variants

    j = await db.get(ToolJob, job_id)
    if j is None or j.kind != "sensors_variants":
        raise HTTPException(404, "variants run not found")
    if j.status != "done":
        raise HTTPException(409, f"the run is {j.status}; only a finished run can be reverted")
    res = dict(j.result or {})
    if res.get("reverted"):
        raise HTTPException(409, "this run was already reverted")
    out = await revert_variants(db, list(res.get("frame_ids") or []),
                                res.get("session_ids") or ([res["session_id"]] if res.get("session_id") else []))
    j = await db.get(ToolJob, job_id)
    j.result = {**res, "reverted": {**out, "by": str(user.user_id)}}
    await db.commit()
    return out

