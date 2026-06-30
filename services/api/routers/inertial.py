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
