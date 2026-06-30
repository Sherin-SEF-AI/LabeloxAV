"""Plane-4 inertial endpoints. The derived ego-state series (yaw rate, longitudinal + lateral acceleration,
jerk) computed from each frame's GNSS + CAN speed - the honest signal the inertial timeline and event
tagging ride on until a measured IMU is ingested."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services.api.deps import require_role

router = APIRouter()


class EventIn(BaseModel):
    kind: str
    modality: str                             # imu|audio|scene|geo|crossmodal
    t_start_ns: int
    t_end_ns: int | None = None
    payload: dict = {}


class EventEdit(BaseModel):
    kind: str | None = None
    t_start_ns: int | None = None
    t_end_ns: int | None = None
    payload: dict | None = None
    state: str | None = None
    expected_version: int | None = None


@router.get("/sessions/{session_id}/timeline")
async def timeline(session_id: UUID):
    """The canonical multimodal timeline (Milestone A): modalities present, ts range, and the sync method +
    accumulated-error estimate. The single axis the synchronized workspace scrubs."""
    from services.intelligence.timeline import session_timeline
    return await session_timeline(session_id)


@router.get("/sessions/{session_id}/egostate")
async def ego_state(session_id: UUID):
    """The derived ego-state series for a session (source=derived). Drives the inertial timeline."""
    from services.intelligence.egostate import session_ego_state
    return await session_ego_state(session_id)


@router.get("/sessions/{session_id}/inertial_events")
async def inertial_events(session_id: UUID):
    """Tagged inertial events (hard brake, hard accel, swerve, impact), anomaly pre-marks, and maneuver
    segments for a session, computed from the ego-state series."""
    from services.intelligence.inertial_events import session_inertial_events
    return await session_inertial_events(session_id)


# ---- Milestone B: timeline events (human, auto candidate, crossmodal correlation) ----

@router.get("/sessions/{session_id}/events")
async def list_timeline_events(session_id: UUID, modality: str | None = None):
    """All timeline events for a session, optionally filtered by modality."""
    from services.intelligence.timeline_events import list_events
    return await list_events(session_id, modality)


@router.post("/sessions/{session_id}/events", dependencies=[Depends(require_role("annotator"))])
async def create_timeline_event(session_id: UUID, body: EventIn):
    """Place a human event on the timeline (imu/audio/scene/geo)."""
    from services.intelligence.timeline_events import create_event
    return await create_event(session_id, body.kind, body.modality, body.t_start_ns, body.t_end_ns, body.payload)


@router.patch("/events/{event_id}", dependencies=[Depends(require_role("annotator"))])
async def edit_timeline_event(event_id: UUID, body: EventEdit):
    """Edit an event with optimistic concurrency: a stale expected_version is a 409."""
    from services.intelligence.timeline_events import update_event
    res = await update_event(event_id, body.model_dump(exclude={"expected_version"}), body.expected_version)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    if res.get("conflict"):
        raise HTTPException(409, {"detail": "event changed since you loaded it", "current_version": res["current_version"]})
    return res


@router.delete("/events/{event_id}", dependencies=[Depends(require_role("annotator"))])
async def remove_timeline_event(event_id: UUID, expected_version: int | None = None):
    """Delete an event with optimistic concurrency."""
    from services.intelligence.timeline_events import delete_event
    res = await delete_event(event_id, expected_version)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    if res.get("conflict"):
        raise HTTPException(409, {"detail": "event changed since you loaded it", "current_version": res["current_version"]})
    return res


@router.post("/sessions/{session_id}/events/auto", dependencies=[Depends(require_role("annotator"))])
async def persist_auto_events(session_id: UUID):
    """Persist the auto-detected inertial spikes as unconfirmed candidates (source=auto), never accepted."""
    from services.intelligence.timeline_events import persist_auto_inertial_events
    return await persist_auto_inertial_events(session_id)


@router.post("/sessions/{session_id}/events/scene", dependencies=[Depends(require_role("annotator"))])
async def persist_scene_events_ep(session_id: UUID):
    """Segment adverse-condition runs (rain, fog, night, dusk) from frame.scene into unconfirmed scene events."""
    from services.intelligence.scene_events import persist_scene_events
    return await persist_scene_events(session_id)


@router.get("/sessions/{session_id}/qa/consistency")
async def qa_consistency(session_id: UUID):
    """Milestone C: the worst-first cross-modal QA queue - the 2D-3D reprojection-inconsistency flags ranked
    by severity plus the timestamp-seam flags (events with no aligned camera frame)."""
    from services.intelligence.consistency_qa import consistency_qa_queue
    return await consistency_qa_queue(session_id)


@router.post("/sessions/{session_id}/events/correlate", dependencies=[Depends(require_role("annotator"))])
async def correlate_timeline_event(session_id: UUID, ts_ns: int, window_ns: int = 250_000_000):
    """Bind the inertial spike, the frame, and the audio region at ts into one crossmodal event."""
    from services.intelligence.timeline_events import correlate_event
    return await correlate_event(session_id, ts_ns, window_ns)
