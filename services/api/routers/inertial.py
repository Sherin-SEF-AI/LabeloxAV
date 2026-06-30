"""Plane-4 inertial endpoints. The derived ego-state series (yaw rate, longitudinal + lateral acceleration,
jerk) computed from each frame's GNSS + CAN speed - the honest signal the inertial timeline and event
tagging ride on until a measured IMU is ingested."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

router = APIRouter()


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
