"""Cloud GPU control API: connect / disconnect a warm RunPod A100 held across a work session, with a live
pollable status (cost meter + countdowns) and orphan detection. The cost safety lives in
compute/runpod/session.py; this is the thin transport. Connect requires the user to echo the hourly rate
shown in the confirm dialog, so a GPU is never fired without the cost in view."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from compute.runpod.orchestrator import RunpodError
from compute.runpod.session import get_manager
from services.api.deps import current_user

router = APIRouter()


class ConnectIn(BaseModel):
    ack_hourly_usd: float  # the rate the user acknowledged in the confirm dialog; must match the live rate


class DisconnectIn(BaseModel):
    pause: bool = False    # default terminates (billing stops); pause keeps the pod (volume still bills)


class TerminatePodIn(BaseModel):
    pod_id: str


@router.post("/cloud/connect")
async def cloud_connect(payload: ConnectIn, user=Depends(current_user)):
    try:
        return await get_manager().connect(payload.ack_hourly_usd)
    except ValueError as exc:
        # rate not acknowledged / stale: 409 so the UI re-fetches the rate and re-confirms
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RunpodError as exc:
        # not configured (no RUNPOD_API_KEY) or RunPod rejected the request
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/cloud/disconnect")
async def cloud_disconnect(payload: DisconnectIn, user=Depends(current_user)):
    return await get_manager().disconnect(terminate=not payload.pause)


@router.get("/cloud/status")
async def cloud_status():
    # cheap and pollable; also enforces the idle / max-session guards on every poll
    return await get_manager().status()


@router.get("/cloud/orphans")
async def cloud_orphans():
    return {"orphans": await get_manager().find_orphans()}


@router.post("/cloud/orphans/terminate")
async def cloud_terminate_orphan(payload: TerminatePodIn, user=Depends(current_user)):
    await get_manager().terminate_pod(payload.pod_id)
    return {"terminated": payload.pod_id}
