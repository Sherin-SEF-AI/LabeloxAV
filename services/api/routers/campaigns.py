"""Campaigns: the improvement loop as an API.

Advancing a campaign is a reviewer action, not an annotator one: a tick can commission review work and
launch a retrain, both of which spend other people's time and a GPU.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role, require_user

router = APIRouter()


class CampaignIn(BaseModel):
    name: str
    class_name: str
    target_metric: str = "recall"
    target_value: float = 0.6
    label_budget: int = 2000
    max_iterations: int = 6
    patience: int = 2
    task_type: str = "detection"
    require_approval: bool = True
    autopilot_stages: list[str] = []
    notes: str | None = None


@router.get("/campaigns")
async def list_campaigns(status: str | None = None, limit: int = Query(100, ge=1, le=500),
                         db: AsyncSession = Depends(db_session)):
    from services.flywheel.campaign import list_campaigns as _list

    return await _list(db, status=status, limit=limit)


@router.post("/campaigns", dependencies=[Depends(require_role("reviewer"))])
async def create_campaign(payload: CampaignIn, user=Depends(require_user),
                          db: AsyncSession = Depends(db_session)):
    from services.flywheel.campaign import CampaignError
    from services.flywheel.campaign import create_campaign as _create

    try:
        return await _create(db, **payload.model_dump(),
                             created_by=getattr(user, "name", None))
    except CampaignError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/campaigns/{campaign_id}")
async def campaign_detail(campaign_id: str, db: AsyncSession = Depends(db_session)):
    from services.flywheel.campaign import CampaignError
    from services.flywheel.campaign import campaign_detail as _detail

    try:
        return await _detail(db, campaign_id)
    except CampaignError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/campaigns/{campaign_id}/tick", dependencies=[Depends(require_role("reviewer"))])
async def tick(campaign_id: str, dry_run: bool = False,
               db: AsyncSession = Depends(db_session)):
    """Advance a campaign by one step.

    One step per call. A long-lived task could not survive a restart, be inspected halfway, or be stopped
    except by killing something, and all three of those matter for a loop that spends a GPU.
    """
    from services.flywheel.campaign import CampaignError
    from services.flywheel.campaign import tick as _tick

    try:
        return await _tick(db, campaign_id, dry_run=dry_run)
    except CampaignError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/campaigns/{campaign_id}/approve", dependencies=[Depends(require_role("reviewer"))])
async def approve_stage(campaign_id: str, stage: str, db: AsyncSession = Depends(db_session)):
    """Approve and run one waiting stage. This is the human in the gate."""
    from services.flywheel.campaign import CampaignError, run_stage

    try:
        return await run_stage(db, campaign_id, stage)
    except CampaignError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/campaigns/{campaign_id}/stop", dependencies=[Depends(require_role("reviewer"))])
async def stop(campaign_id: str, reason: str = "stopped by an operator",
               db: AsyncSession = Depends(db_session)):
    from services.flywheel.campaign import CampaignError, stop_campaign

    try:
        return await stop_campaign(db, campaign_id, reason)
    except CampaignError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- lineage

@router.get("/lineage/model/{model_version}")
async def model_lineage(model_version: str, max_sessions: int = Query(40, ge=1, le=200),
                        db: AsyncSession = Depends(db_session)):
    """Everything a model is made of, as a DAG. The direction an audit asks in."""
    from services.release.lineage_graph import model_lineage as _lineage

    try:
        return await _lineage(db, model_version, max_sessions=max_sessions)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/lineage/session/{session_id}")
async def session_lineage(session_id: str, db: AsyncSession = Depends(db_session)):
    """The forward direction: given this footage, what did it end up in.

    The question an erasure request asks, and the one that previously meant reading every commit's slice
    spec by hand.
    """
    from services.release.lineage_graph import session_lineage as _lineage

    try:
        return await _lineage(db, session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/lineage/dataset/{commit_id}")
async def dataset_lineage(commit_id: str, db: AsyncSession = Depends(db_session)):
    from services.release.lineage_graph import dataset_lineage as _lineage

    try:
        return await _lineage(db, commit_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- tracklets

@router.get("/tracklets/{track_id}")
async def load_tracklet(track_id: str, db: AsyncSession = Depends(db_session)):
    from services.temporal.tracklet import TrackletError
    from services.temporal.tracklet import load_tracklet as _load

    try:
        return await _load(db, track_id)
    except TrackletError as exc:
        raise HTTPException(404, str(exc)) from exc


class KeyframeIn(BaseModel):
    bbox: list[float] | None = None
    is_keyframe: bool = True


@router.post("/tracklets/objects/{object_id}/keyframe",
             dependencies=[Depends(require_role("annotator"))])
async def set_keyframe(object_id: str, payload: KeyframeIn, user=Depends(require_user),
                       db: AsyncSession = Depends(db_session)):
    """Mark a frame as observed, optionally correcting its box.

    Correcting necessarily makes it a keyframe: a corrected box that stayed derived would be overwritten
    by the next derive, which is the single most infuriating thing a video annotator can experience.
    """
    from services.temporal.tracklet import TrackletError
    from services.temporal.tracklet import set_keyframe as _set

    try:
        return await _set(db, object_id, bbox=payload.bbox, is_keyframe=payload.is_keyframe,
                          user_name=getattr(user, "name", None))
    except TrackletError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/tracklets/{track_id}/derive", dependencies=[Depends(require_role("annotator"))])
async def derive(track_id: str, method: str = "linear", overwrite_human: bool = False,
                 db: AsyncSession = Depends(db_session)):
    from services.temporal.tracklet import TrackletError
    from services.temporal.tracklet import derive as _derive

    try:
        return await _derive(db, track_id, method=method, overwrite_human=overwrite_human)
    except TrackletError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/tracklets/objects/{object_id}/propagate",
             dependencies=[Depends(require_role("annotator"))])
async def propagate(object_id: str, direction: str = "both", frames: int = Query(12, ge=1, le=120),
                    refine: bool = True, db: AsyncSession = Depends(db_session)):
    from services.temporal.tracklet import TrackletError
    from services.temporal.tracklet import propagate as _prop

    try:
        return await _prop(db, object_id, direction=direction, frames=frames, refine=refine)
    except TrackletError as exc:
        raise HTTPException(400, str(exc)) from exc


class TrackAttrsIn(BaseModel):
    attrs: dict


@router.post("/tracklets/{track_id}/attributes", dependencies=[Depends(require_role("annotator"))])
async def set_track_attributes(track_id: str, payload: TrackAttrsIn, user=Depends(require_user),
                               db: AsyncSession = Depends(db_session)):
    """Set an attribute once for a whole track. A vehicle's colour does not change between frames."""
    from services.temporal.tracklet import TrackletError
    from services.temporal.tracklet import set_track_attributes as _set

    try:
        return await _set(db, track_id, payload.attrs, user_name=getattr(user, "name", None))
    except TrackletError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/tracklets/{track_id}/suggest-keyframes")
async def suggest_keyframes(track_id: str, budget: int = Query(8, ge=1, le=50),
                            db: AsyncSession = Depends(db_session)):
    """Where the next correction buys the most frames."""
    from services.temporal.tracklet import suggest_keyframes as _suggest

    return await _suggest(db, track_id, budget=budget)


@router.get("/tracklets/stats/summary")
async def tracklet_stats(session_id: str | None = None, db: AsyncSession = Depends(db_session)):
    from services.temporal.tracklet import tracklet_stats as _stats

    return await _stats(db, session_id)
