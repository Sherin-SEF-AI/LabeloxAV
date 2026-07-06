"""ORACLYX offline-fusion pseudo-GT endpoints: record a consensus verdict for an object, read the
consensus/disagreement board, and export the distillation set. The multi-view/LiDAR fusion itself lives in
the existing services/lidar and services/autolabel/fusion; this router adds the consensus gate and the
distillation export. Mounted under /api; the platform registry groups /lidar and /hdmap under ORACLYX."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from services.oraclyx.mono_depth import metric_depth
from services.oraclyx.radar import fuse_radar_velocity
from services.oraclyx.run import consensus_summary, export_distillation, record_consensus
from services.oraclyx.tracks4d import stitch_track
from services.oraclyx.uncertainty import pseudo_label_uncertainty, rank_disagreements, soft_target

router = APIRouter()


class StitchIn(BaseModel):
    observations: dict[int, list[float]]
    n_frames: int
    max_heal_gap: int = 8


@router.post("/oraclyx/tracks4d/stitch")
async def tracks4d_stitch(payload: StitchIn):
    """Stitch sparse per-frame consensus boxes into one 4D pseudo-GT track, healing short occlusion gaps."""
    return stitch_track(payload.observations, payload.n_frames, payload.max_heal_gap)


class RadarIn(BaseModel):
    track_box: list[float]
    camera_velocity: float
    radar_returns: list[dict]
    ego_speed: float = 0.0


@router.post("/oraclyx/radar/fuse")
async def radar_fuse(payload: RadarIn):
    """Fuse a radar Doppler range-rate onto a track box, rejecting mis-associated returns."""
    return fuse_radar_velocity(payload.track_box, payload.camera_velocity, payload.radar_returns,
                               payload.ego_speed)


class MonoDepthIn(BaseModel):
    box: list[float]
    class_id: int
    focal_px: float
    cy: float
    cam_height_m: float | None = None
    pitch_rad: float = 0.0


@router.post("/oraclyx/mono-depth")
async def mono_depth(payload: MonoDepthIn):
    """Recover a metric depth for a camera-only object from the ground and known-size priors, or refuse it."""
    r = metric_depth(payload.box, payload.class_id, payload.focal_px, payload.cy, payload.cam_height_m,
                     payload.pitch_rad)
    return r or {"depth_m": None, "reason": "no calibration and no size prior"}


class UncertaintyIn(BaseModel):
    consensus_score: float
    n_views: int = 1
    calib_confidence: float = 1.0
    depth_uncertainty: float = 0.0


@router.post("/oraclyx/uncertainty")
async def uncertainty(payload: UncertaintyIn):
    """Calibrated pseudo-GT uncertainty and the distillation soft-target weight for one label."""
    u = pseudo_label_uncertainty(payload.consensus_score, payload.n_views, payload.calib_confidence,
                                 payload.depth_uncertainty)
    return {"uncertainty": u, "soft_target": soft_target(u)}


class RankIn(BaseModel):
    items: list[dict]


@router.post("/oraclyx/disagreements/rank")
async def disagreements_rank(payload: RankIn):
    """Rank pseudo labels for human review by descending expected information gain (the M14 review queue)."""
    return {"ranked": rank_disagreements(payload.items)}


class Detection(BaseModel):
    path: str
    class_id: int
    bbox: list[float]
    conf: float = 0.0


class ConsensusIn(BaseModel):
    fusion: Detection
    auto_paths: list[Detection]
    fusion_run_id: str | None = None


@router.post("/oraclyx/object/{object_id}/consensus")
async def consensus(object_id: uuid.UUID, payload: ConsensusIn):
    """Vote fusion against the three auto-label paths for an object, persist the pseudo-label, and route the
    object (auto_accept on consensus, review on disagreement)."""
    return await record_consensus(object_id, payload.fusion.model_dump(),
                                  [p.model_dump() for p in payload.auto_paths], payload.fusion_run_id)


@router.get("/oraclyx/board")
async def board(db: AsyncSession = Depends(db_session)):
    """The consensus/disagreement board: auto-accepted vs human-routed pseudo-label counts."""
    return await consensus_summary(db)


@router.get("/oraclyx/distillation")
async def distillation(min_score: float = 0.5, db: AsyncSession = Depends(db_session)):
    """The distillation manifest: consensus pseudo-labels with soft targets, the edge-model training signal."""
    return await export_distillation(db, min_score=min_score)
