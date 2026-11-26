"""API for the driving annotation tools, agent D: driving (#7 lanes and HD map, #8 paths and intent, #9 camera-LiDAR
alignment). Mounted under /api; every path here starts with /tools/driving/.

#7 Lanes in a top-down view (/annotate/lane/[frameId], bird's-eye mode; the warp itself is GET /frames/{id}/bev)
- GET    frames/{frame_id}/bev-lanes          lanes in BEV pixels, their relations, the frame's signals, pose status
- PUT    frames/{frame_id}/bev-lanes          create or reshape a lane from BEV points (stored in image space, human)
- POST   lanes/{lane_id}/relations            predecessor / successor / left / right neighbour / controlled_by / stops_at
- DELETE lane-relations/{relation_id}
- POST   frames/{frame_id}/lanes/propagate-pose   carry the lanes along the route through the ego poses (a job when long)
- POST   lanes/export                          Lanelet2 + OpenDRIVE for one or more frames, with the drawn topology

#8 Paths and intent (/trajectories)
- GET    paths/sessions                        sessions with tracks, and whether they have ego poses
- GET    paths/sessions/{session_id}           ego path, every movable track's path, labels, event counts
- GET    paths/tracks/{track_id}/forecast      the CTRV kinematic baseline from an anchor frame, with its error
- PUT    paths/tracks/{track_id}/future        save a future path (baseline or edited)
- DELETE paths/labels/{path_id}
- POST   paths/sessions/{session_id}/ego/smooth    the measured ego path smoothed (not saved)
- PUT    paths/sessions/{session_id}/ego       save the corrected ego path (ego_pose is never modified)
- GET    paths/tracks/{track_id}/intents       events, intents, and the vocabulary that applies
- POST   paths/tracks/{track_id}/intents       an intent span (track_event, mirrored onto Track.intents)
- DELETE paths/intents/{event_id}
- GET    paths/sessions/{session_id}/scenario  the labelled paths and intents as JSON plus an OpenSCENARIO replay

#9 Camera-LiDAR alignment (/calibration/align)
- GET    align/sessions                        sessions with LiDAR scans and synced camera frames
- GET    align/sessions/{session_id}/frames    the synced frames
- GET    align/frames/{frame_id}/view          points for live projection, the edge field, both calibrations, scores
- POST   align/score                           server-side scores of an extrinsic over some frames
- POST   align/solve-pairs                     the extrinsic that explains clicked point pairs (PnP)
- POST   align/refine                          queue `driving_calib_refine`
- POST   align/save                            store as the camera's `manual` calibration beside the dataset's (reviewer)
- DELETE align/sessions/{session_id}/cameras/{cam_id}/manual   remove it, restoring the dataset calibration (reviewer)
- GET    align/sessions/{session_id}/history   every alignment saved or proposed, with scores
"""

from __future__ import annotations

import base64
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EgoPose, Frame, Track, TrackEvent
from db.models import Session as DbSession
from services.api.deps import db_session, require_role, role_rank
from services.api.project_guard import caller_hidden_projects
from services.tools.registry import enqueue

log = get_logger("tools_driving")
router = APIRouter()

ANN = Depends(require_role("annotator"))
SYNC_PROPAGATE_MAX = 30


def _job(job) -> dict:
    return {"job_id": str(job.job_id), "kind": job.kind, "status": job.status,
            "session_id": str(job.session_id) if job.session_id else None}


def _err(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc).strip("'\""))
    return HTTPException(400, str(exc))


# ---- #7 lanes in a top-down view -------------------------------------------------------------------------------

class BevLaneIn(BaseModel):
    lane_id: uuid.UUID | None = None
    bev_points: list[list[float]]
    lane_type: str = "solid"
    is_ego: bool = False


class RelationIn(BaseModel):
    relation: str
    target_lane_id: uuid.UUID | None = None
    target_object_id: uuid.UUID | None = None
    target_element_id: uuid.UUID | None = None


class PropagateIn(BaseModel):
    frames: int = Field(10, ge=1, le=500)
    lane_ids: list[uuid.UUID] | None = None


class ExportIn(BaseModel):
    frame_ids: list[uuid.UUID] = Field(min_length=1, max_length=200)


@router.get("/tools/driving/frames/{frame_id}/bev-lanes", dependencies=[ANN])
async def bev_lanes(frame_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.hdmap.lane_map import bev_state

    try:
        return await bev_state(db, frame_id)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.put("/tools/driving/frames/{frame_id}/bev-lanes")
async def put_bev_lane(frame_id: uuid.UUID, body: BevLaneIn, db: AsyncSession = Depends(db_session), user=ANN):
    from services.hdmap.lane_map import save_bev_lane

    try:
        return await save_bev_lane(db, frame_id, body.bev_points, body.lane_type, body.is_ego, body.lane_id)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.post("/tools/driving/lanes/{lane_id}/relations", status_code=201)
async def add_lane_relation(lane_id: uuid.UUID, body: RelationIn, db: AsyncSession = Depends(db_session), user=ANN):
    from services.hdmap.lane_map import add_relation

    try:
        return await add_relation(db, lane_id, body.relation, target_lane_id=body.target_lane_id,
                                  target_object_id=body.target_object_id, target_element_id=body.target_element_id,
                                  user_id=user.user_id)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.delete("/tools/driving/lane-relations/{relation_id}", dependencies=[ANN])
async def delete_lane_relation(relation_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.hdmap.lane_map import delete_relation

    if not await delete_relation(db, relation_id):
        raise HTTPException(404, "relation not found")
    return {"deleted": str(relation_id)}


@router.post("/tools/driving/frames/{frame_id}/lanes/propagate-pose")
async def propagate_pose(frame_id: uuid.UUID, body: PropagateIn, db: AsyncSession = Depends(db_session), user=ANN):
    """Short ranges answer at once; longer ones run as a `driving_lane_propagate` job."""
    from services.hdmap.lane_map import propagate_by_pose

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    if body.frames > SYNC_PROPAGATE_MAX:
        job = await enqueue(db, "driving_lane_propagate",
                            {"frame_id": str(frame_id), "frames": body.frames,
                             "lane_ids": [str(x) for x in body.lane_ids or []]},
                            session_id=frame.session_id, user_id=user.user_id)
        return {"queued": True, **_job(job)}
    try:
        return {"queued": False, **await propagate_by_pose(db, frame_id, body.frames, body.lane_ids)}
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.post("/tools/driving/lanes/export", dependencies=[ANN])
async def export_lanes(body: ExportIn, db: AsyncSession = Depends(db_session)):
    from services.hdmap.lane_map import export_frames

    try:
        return await export_frames(db, body.frame_ids)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


# ---- #8 paths and intent ---------------------------------------------------------------------------------------

class FutureIn(BaseModel):
    anchor_frame_id: uuid.UUID
    points: list[list[float]]
    source: Literal["human", "baseline_ctrv"] = "human"
    seed: dict | None = None
    notes: str | None = None


class EgoIn(BaseModel):
    points: list[list[float]]
    source: Literal["human", "smoothed"] = "human"
    notes: str | None = None


class SmoothIn(BaseModel):
    window: int = Field(5, ge=3, le=31)


class IntentIn(BaseModel):
    event_type: str
    start_frame_id: uuid.UUID
    end_frame_id: uuid.UUID
    notes: str | None = None


@router.get("/tools/driving/paths/sessions", dependencies=[ANN])
async def path_sessions(limit: int = Query(60, ge=1, le=300), db: AsyncSession = Depends(db_session),
                        hidden=Depends(caller_hidden_projects)):
    """Sessions with tracks, the most tracked first, and whether each has ego poses (world vs ego frame)."""
    q = (select(Track.session_id, func.count(Track.track_id)).group_by(Track.session_id)
         .order_by(func.count(Track.track_id).desc()).limit(limit))
    rows = (await db.execute(q)).all()
    sids = [r[0] for r in rows]
    poses: dict[uuid.UUID, int] = {s: int(n) for s, n in (await db.execute(
        select(EgoPose.session_id, func.count()).where(EgoPose.session_id.in_(sids))
        .group_by(EgoPose.session_id))).all()} if sids else {}
    sess = {s.session_id: s for s in (await db.execute(select(DbSession).where(
        DbSession.session_id.in_(sids)))).scalars()} if sids else {}
    out = []
    for sid, n in rows:
        s = sess.get(sid)
        if s is None or (hidden and s.project_id in hidden):
            continue
        out.append({"session_id": str(sid), "route": s.route, "vehicle_id": s.vehicle_id, "city": s.city,
                    "tracks": int(n), "ego_poses": int(poses.get(sid, 0))})
    return out


@router.get("/tools/driving/paths/sessions/{session_id}", dependencies=[ANN])
async def path_session(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.tools.driving_paths import session_view

    try:
        return await session_view(db, session_id)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.get("/tools/driving/paths/tracks/{track_id}/forecast", dependencies=[ANN])
async def track_forecast(track_id: uuid.UUID, anchor_frame_id: uuid.UUID | None = None,
                         horizon_s: float = Query(3.0, ge=0.5, le=10.0), dt_s: float = Query(0.25, ge=0.05, le=1.0),
                         db: AsyncSession = Depends(db_session)):
    from services.tools.driving_paths import forecast

    try:
        return await forecast(db, track_id, anchor_frame_id, horizon_s, dt_s)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.put("/tools/driving/paths/tracks/{track_id}/future")
async def put_future(track_id: uuid.UUID, body: FutureIn, db: AsyncSession = Depends(db_session), user=ANN):
    from services.tools.driving_paths import save_future

    try:
        return await save_future(db, track_id, body.anchor_frame_id, body.points, body.source, body.seed,
                                 user.user_id, body.notes)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.delete("/tools/driving/paths/labels/{path_id}")
async def delete_path_label(path_id: uuid.UUID, db: AsyncSession = Depends(db_session), user=ANN):
    from db.models import PathLabel

    row = await db.get(PathLabel, path_id)
    if row is None:
        raise HTTPException(404, "path label not found")
    if row.created_by != user.user_id and role_rank(user.role) < role_rank("reviewer"):
        raise HTTPException(403, "only its author or a reviewer can delete someone's path label")
    await db.delete(row)
    await db.commit()
    return {"deleted": str(path_id)}


@router.post("/tools/driving/paths/sessions/{session_id}/ego/smooth", dependencies=[ANN])
async def smooth_ego_path(session_id: uuid.UUID, body: SmoothIn, db: AsyncSession = Depends(db_session)):
    from services.tools.driving_paths import _context, ego_path, smooth_ego

    try:
        ctx = await _context(db, session_id)
    except LookupError as exc:
        raise _err(exc) from exc
    poses = ego_path(ctx)
    if len(poses) < 3:
        raise HTTPException(400, "this session has fewer than three ego poses, so there is nothing to smooth")
    return {"points": smooth_ego(poses, body.window), "anchor_ts_ns": poses[0]["ts_ns"], "window": body.window}


@router.put("/tools/driving/paths/sessions/{session_id}/ego")
async def put_ego_path(session_id: uuid.UUID, body: EgoIn, db: AsyncSession = Depends(db_session), user=ANN):
    from services.tools.driving_paths import save_ego_path

    try:
        return await save_ego_path(db, session_id, body.points, body.source, user.user_id, body.notes)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.get("/tools/driving/paths/tracks/{track_id}/intents", dependencies=[ANN])
async def track_intents(track_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.tools.driving_paths import list_intents

    try:
        return await list_intents(db, track_id)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.post("/tools/driving/paths/tracks/{track_id}/intents", status_code=201)
async def add_track_intent(track_id: uuid.UUID, body: IntentIn, db: AsyncSession = Depends(db_session), user=ANN):
    """An annotator's span lands `proposed` for a reviewer to accept (the track-events floor is reviewer, and this
    does not go around it); a reviewer's lands `accepted`."""
    from services.tools.driving_paths import add_intent

    state = "accepted" if role_rank(user.role) >= role_rank("reviewer") else "proposed"
    try:
        return await add_intent(db, track_id, body.event_type, body.start_frame_id, body.end_frame_id,
                                state=state, user=user, notes=body.notes)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.delete("/tools/driving/paths/intents/{event_id}")
async def delete_track_intent(event_id: uuid.UUID, db: AsyncSession = Depends(db_session), user=ANN):
    ev = await db.get(TrackEvent, event_id)
    if ev is None:
        raise HTTPException(404, "event not found")
    mine = ev.source == "human" and ev.state == "proposed" and ev.created_by == getattr(user, "name", None)
    if not mine and role_rank(user.role) < role_rank("reviewer"):
        raise HTTPException(403, "only a reviewer can delete an accepted or someone else's intent")
    tr = await db.get(Track, ev.track_id)
    if tr is not None and tr.intents:
        others = (await db.execute(select(func.count()).select_from(TrackEvent).where(
            TrackEvent.track_id == ev.track_id, TrackEvent.event_type == ev.event_type,
            TrackEvent.event_id != ev.event_id))).scalar()
        if not others:
            tr.intents = [i for i in tr.intents if not (i.get("intent") == ev.event_type and i.get("source") == "human")]
    await db.delete(ev)
    await db.commit()
    return {"deleted": str(event_id)}


@router.get("/tools/driving/paths/sessions/{session_id}/scenario", dependencies=[ANN])
async def scenario(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    from services.tools.driving_paths import scenario_export

    try:
        return await scenario_export(db, session_id)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


# ---- #9 camera-LiDAR alignment ---------------------------------------------------------------------------------

class ExtrinsicIn(BaseModel):
    frame_ids: list[uuid.UUID] = Field(min_length=1, max_length=6)
    rpy_deg: list[float] = Field(min_length=3, max_length=3)
    xyz_m: list[float] = Field(min_length=3, max_length=3)


class RefineIn(BaseModel):
    frame_ids: list[uuid.UUID] = Field(min_length=1, max_length=6)
    rpy_deg: list[float] | None = Field(None, min_length=3, max_length=3)
    xyz_m: list[float] | None = Field(None, min_length=3, max_length=3)


class PairsIn(BaseModel):
    frame_id: uuid.UUID
    points: list[list[float]] = Field(min_length=1)
    uv: list[list[float]] = Field(min_length=1)
    rpy_deg: list[float] | None = Field(None, min_length=3, max_length=3)
    xyz_m: list[float] | None = Field(None, min_length=3, max_length=3)


class SaveIn(ExtrinsicIn):
    method: Literal["manual", "auto", "pairs"] = "manual"
    notes: str | None = None
    pairs_points: list[list[float]] | None = None
    pairs_uv: list[list[float]] | None = None


@router.get("/tools/driving/align/sessions", dependencies=[ANN])
async def align_sessions(db: AsyncSession = Depends(db_session), hidden=Depends(caller_hidden_projects)):
    from services.lidar.calib.align_store import lidar_sessions

    rows = await lidar_sessions(db)
    if hidden:
        keep = {s for (s,) in (await db.execute(select(DbSession.session_id).where(
            DbSession.session_id.in_([uuid.UUID(r["session_id"]) for r in rows]),
            DbSession.project_id.notin_(hidden) | DbSession.project_id.is_(None)))).all()}
        rows = [r for r in rows if uuid.UUID(r["session_id"]) in keep]
    return rows


@router.get("/tools/driving/align/sessions/{session_id}/frames", dependencies=[ANN])
async def align_frames(session_id: uuid.UUID, cam_id: str | None = None, db: AsyncSession = Depends(db_session)):
    from services.lidar.calib.align_store import synced_frames

    return {"session_id": str(session_id), "frames": await synced_frames(db, session_id, cam_id)}


@router.get("/tools/driving/align/frames/{frame_id}/view", dependencies=[ANN])
async def align_view(frame_id: uuid.UUID, max_points: int = Query(40000, ge=1000, le=120000),
                     db: AsyncSession = Depends(db_session)):
    """Points as base64 little-endian Float32 rows of (x, y, z, intensity, edge weight) in the ego frame."""
    from services.lidar.calib.align_store import view_points

    try:
        v = await view_points(db, frame_id, max_points)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc
    pts = v.pop("points")
    v["points_b64"] = base64.b64encode(pts.astype("<f4").tobytes()).decode()
    v["point_count"] = int(len(pts))
    v["point_stride"] = 5
    return v


@router.post("/tools/driving/align/score", dependencies=[ANN])
async def align_score(body: ExtrinsicIn, db: AsyncSession = Depends(db_session)):
    from services.lidar.calib.align_store import score_extrinsic

    try:
        return await score_extrinsic(db, body.frame_ids, body.rpy_deg, body.xyz_m)
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.post("/tools/driving/align/solve-pairs", dependencies=[ANN])
async def align_solve_pairs(body: PairsIn, db: AsyncSession = Depends(db_session)):
    from services.lidar.calib.align import pairs_error, solve_from_pairs, with_extrinsics
    from services.lidar.calib.align_store import calibrations

    if len(body.points) != len(body.uv):
        raise HTTPException(400, "points and uv must pair up one to one")
    frame = await db.get(Frame, body.frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    cur = (await calibrations(db, frame.session_id, frame.cam_id, frame.width, frame.height))["current"]
    if body.rpy_deg and body.xyz_m:
        cur = with_extrinsics(cur, body.rpy_deg, body.xyz_m)
    before = pairs_error(body.points, body.uv, cur, frame.width, frame.height)
    try:
        sol = solve_from_pairs(body.points, body.uv, cur)
    except ValueError as exc:
        raise _err(exc) from exc
    after = pairs_error(body.points, body.uv, with_extrinsics(cur, sol["rpy_deg"], sol["xyz_m"]),
                        frame.width, frame.height)
    return {**sol, "before": before, "after": after, "pairs": len(body.points)}


@router.post("/tools/driving/align/refine")
async def align_refine(body: RefineIn, db: AsyncSession = Depends(db_session), user=ANN):
    frame = await db.get(Frame, body.frame_ids[0])
    if frame is None:
        raise HTTPException(404, "frame not found")
    job = await enqueue(db, "driving_calib_refine",
                        {"frame_ids": [str(f) for f in body.frame_ids], "rpy_deg": body.rpy_deg,
                         "xyz_m": body.xyz_m, "user_id": str(user.user_id)},
                        session_id=frame.session_id, user_id=user.user_id)
    return _job(job)


@router.post("/tools/driving/align/save")
async def align_save(body: SaveIn, db: AsyncSession = Depends(db_session), user=Depends(require_role("reviewer"))):
    """Store the extrinsic as the camera's manual calibration. Reviewer: every 3D projection of the session reads
    it from then on. The dataset row is untouched."""
    from services.lidar.calib.align_store import ResolverNotManualAware, save_manual

    pairs = ({"points": body.pairs_points, "uv": body.pairs_uv}
             if body.pairs_points and body.pairs_uv and len(body.pairs_points) == len(body.pairs_uv) else None)
    try:
        return await save_manual(db, body.frame_ids, body.rpy_deg, body.xyz_m, method=body.method,
                                 user_id=user.user_id, notes=body.notes, pairs=pairs)
    except ResolverNotManualAware as exc:
        raise HTTPException(409, str(exc)) from exc
    except (LookupError, ValueError) as exc:
        raise _err(exc) from exc


@router.delete("/tools/driving/align/sessions/{session_id}/cameras/{cam_id}/manual",
               dependencies=[Depends(require_role("reviewer"))])
async def align_revert(session_id: uuid.UUID, cam_id: str, db: AsyncSession = Depends(db_session)):
    from services.lidar.calib.align_store import revert_manual

    if not await revert_manual(db, session_id, cam_id):
        raise HTTPException(404, "this camera has no manual calibration")
    return {"reverted": True, "session_id": str(session_id), "cam_id": cam_id}


@router.get("/tools/driving/align/sessions/{session_id}/history", dependencies=[ANN])
async def align_history(session_id: uuid.UUID, cam_id: str | None = None, db: AsyncSession = Depends(db_session)):
    from services.lidar.calib.align_store import history

    return {"session_id": str(session_id), "alignments": await history(db, session_id, cam_id)}
