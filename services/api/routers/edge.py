"""Edge telemetry: what deployed devices report, and what the gate makes of it.

The ingest routes sit at an annotator floor rather than a reviewer one, because the caller is a device
rather than a person: a fleet of Jetsons holding reviewer credentials would be a far worse posture than one
holding the narrowest token that can post a measurement. The reading routes stay at reviewer, since fleet
performance is operational state.

Also resumable exports, which live here because they share the same shape: long-running work reporting
progress that somebody watches.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role

router = APIRouter()


# ---------------------------------------------------------------- device reporting

class DeviceIn(BaseModel):
    device_id: str
    name: str | None = None
    hardware: str | None = None      # jetson_orin_nx | rpi5 | hailo8 | ...
    runtime: str | None = None       # tensorrt | onnxruntime | litert
    artifact_id: str | None = None
    model_version: str | None = None
    fleet: str | None = None
    meta: dict = {}


@router.post("/edge/devices", dependencies=[Depends(require_role("annotator"))])
async def register_device(payload: DeviceIn, db: AsyncSession = Depends(db_session)):
    """Register or update a device. Idempotent: a device re-registers on every boot."""
    from services.forgyx.edge_feedback import register_device as _register

    return await _register(db, **payload.model_dump())


class TelemetryIn(BaseModel):
    device_id: str
    window_start_ns: int
    window_end_ns: int
    n_inferences: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_max_ms: float | None = None
    fps: float | None = None
    temp_c_max: float | None = None
    throttled_fraction: float | None = None
    power_w_mean: float | None = None
    conf_histogram: list[float] = []
    detections_per_frame: float | None = None
    dropped_frames: int = 0
    artifact_id: str | None = None
    model_version: str | None = None
    meta: dict = {}


@router.post("/edge/telemetry", dependencies=[Depends(require_role("annotator"))])
async def ingest_telemetry(payload: TelemetryIn, db: AsyncSession = Depends(db_session)):
    """Accept one reporting window from one device.

    A window rather than a sample: a device posting every inference would spend its uplink on telemetry,
    and p50, p95 and the thermal ceiling reached are properties of a window anyway.
    """
    from services.forgyx.edge_feedback import TelemetryError
    from services.forgyx.edge_feedback import ingest_telemetry as _ingest

    try:
        return await _ingest(db, **payload.model_dump())
    except TelemetryError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------- reading it back

@router.get("/edge/devices", dependencies=[Depends(require_role("reviewer"))])
async def list_devices(fleet: str | None = None, limit: int = Query(200, ge=1, le=1000),
                       db: AsyncSession = Depends(db_session)):
    from services.forgyx.edge_feedback import list_devices as _list

    return await _list(db, fleet=fleet, limit=limit)


@router.get("/edge/fleet", dependencies=[Depends(require_role("reviewer"))])
async def fleet_summary(hours: int = Query(24, ge=1, le=8760),
                        db: AsyncSession = Depends(db_session)):
    from services.forgyx.edge_feedback import fleet_summary as _summary

    return await _summary(db, hours=hours)


@router.get("/edge/artifacts/{artifact_id}/field", dependencies=[Depends(require_role("reviewer"))])
async def field_report(artifact_id: str, hours: int = Query(168, ge=1, le=8760),
                       db: AsyncSession = Depends(db_session)):
    """What the field says about one artifact, next to what the bench said.

    The comparison is the product. Either number alone is uninteresting; the gap is the thing the gate has
    never been able to see.
    """
    from services.forgyx.edge_feedback import artifact_field_report

    return await artifact_field_report(db, artifact_id, hours=hours)


@router.get("/edge/artifacts/{artifact_id}/gate", dependencies=[Depends(require_role("reviewer"))])
async def field_gate(artifact_id: str, hours: int = Query(168, ge=1, le=8760),
                     db: AsyncSession = Depends(db_session)):
    """The gate's view once the field has been heard from.

    Advisory by construction. Telemetry comes from devices, which are outside the trust boundary, so a
    single misconfigured unit must not be able to demote a champion.
    """
    from services.forgyx.edge_feedback import field_gate as _gate

    return await _gate(db, artifact_id, hours=hours)


# ---------------------------------------------------------------- resumable exports

@router.get("/exports/{job_id}/progress", dependencies=[Depends(require_role("reviewer"))])
async def export_progress(job_id: str, db: AsyncSession = Depends(db_session)):
    from services.export.resumable import ExportResumeError
    from services.export.resumable import export_progress as _progress

    try:
        return await _progress(db, job_id)
    except ExportResumeError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/exports/resumable", dependencies=[Depends(require_role("reviewer"))])
async def list_resumable(limit: int = Query(50, ge=1, le=200),
                         db: AsyncSession = Depends(db_session)):
    """Failed exports that still have a usable checkpoint."""
    from services.export.resumable import list_resumable as _list

    return await _list(db, limit=limit)


@router.post("/exports/{job_id}/resume", dependencies=[Depends(require_role("reviewer"))])
async def resume_export(job_id: str, db: AsyncSession = Depends(db_session)):
    """Continue a failed export from where it stopped.

    The chunks already written are verified before being skipped: a checkpoint believed without checking
    turns a partial write into a dataset that claims completeness.
    """
    from services.export.resumable import ExportResumeError
    from services.export.resumable import resume_export as _resume

    try:
        return await _resume(db, job_id)
    except ExportResumeError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- champion sweep

@router.post("/activelearn/champion-sweep", dependencies=[Depends(require_role("reviewer"))])
async def champion_sweep(session_id: str | None = None,
                         conf_floor: float = Query(0.05, ge=0.0, le=0.5),
                         accept_threshold: float = Query(0.5, ge=0.0, le=1.0),
                         limit: int = Query(500, ge=1, le=5000),
                         top_k: int = Query(100, ge=1, le=1000),
                         db: AsyncSession = Depends(db_session)):
    """Run the champion at a very low confidence over frames it currently reports nothing in.

    Measures a miss directly rather than inferring one from a frame's neighbours. Expensive, which is why
    it is a sweep somebody schedules rather than something the queue does on every read.
    """
    from services.activelearn.false_negatives import (
        FalseNegativeSweepUnavailable,
    )
    from services.activelearn.false_negatives import (
        champion_sweep as _sweep,
    )

    try:
        return await _sweep(db, session_id=session_id, conf_floor=conf_floor,
                            accept_threshold=accept_threshold, limit=limit, top_k=top_k)
    except FalseNegativeSweepUnavailable as exc:
        # 422, not 500: the request is well-formed and the system cannot answer it yet, and the message
        # says what is missing.
        raise HTTPException(422, str(exc)) from exc
