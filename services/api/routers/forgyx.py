"""FORGYX edge-optimization endpoints: what targets this machine can build, the benchmark matrix with Pareto
ranking, the deployment artifact registry, a device-facing benchmark ingest, and the dual (latency + accuracy)
regression gate. Export/quantize/compile run only where the backend exists (capability-gated). Mounted under
/api; the registry groups /export under FORGYX."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Deployment
from services.api.deps import db_session
from services.forgyx.capabilities import available_targets
from services.forgyx.cooptimize import plan_cooptimization
from services.forgyx.gate import dual_gate
from services.forgyx.packaging import plan_rollout
from services.forgyx.run import audit_benchmarks, benchmark_matrix, record_benchmark
from services.forgyx.thermal import thermal_envelope

router = APIRouter()


@router.get("/forgyx/capabilities")
async def capabilities():
    """Which edge targets this machine can actually build (the rest run on their target device)."""
    return {"targets": available_targets()}


@router.get("/forgyx/benchmarks")
async def benchmarks(model_version: str | None = None, db: AsyncSession = Depends(db_session)):
    """The benchmark matrix across (model, target), Pareto-ranked on latency vs accuracy."""
    return await benchmark_matrix(db, model_version)


@router.get("/forgyx/deployments")
async def deployments(model_version: str | None = None, db: AsyncSession = Depends(db_session)):
    """The deployment artifact registry, with lineage to the release and verdict."""
    q = select(Deployment).order_by(Deployment.created_at.desc())
    if model_version:
        q = q.where(Deployment.model_version == model_version)
    rows = (await db.execute(q.limit(200))).scalars().all()
    return {"deployments": [
        {"deployment_id": str(r.deployment_id), "model_version": r.model_version, "target": r.target,
         "export_format": r.export_format, "status": r.status, "artifact_uri": r.artifact_uri,
         "release_commit": r.release_commit,
         "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]}


class BenchIn(BaseModel):
    model_version: str
    target: str
    latency_ms: dict                 # {p50, p95, p99}
    throughput_fps: float | None = None
    power_w: float | None = None
    artifact_uri: str | None = None


@router.post("/forgyx/benchmark")
async def ingest_benchmark(payload: BenchIn, db: AsyncSession = Depends(db_session)):
    """Ingest a benchmark measured on a target device (Jetson/Android/Hailo report their real numbers here)."""
    return await record_benchmark(db, payload.model_version, payload.target, payload.latency_ms,
                                  payload.throughput_fps, payload.power_w, artifact_uri=payload.artifact_uri)


class ExportIn(BaseModel):
    imgsz: int = 640
    runs: int = 50
    prefer_cuda: bool = True


@router.post("/forgyx/export/{model_version}")
async def export_model(model_version: str, payload: ExportIn, db: AsyncSession = Depends(db_session)):
    """Export a registered model to ONNX, store the artifact, and time it on this machine.

    The first thing in FORGYX that produces an artifact. The module could rank targets, sign manifests and
    plan rollouts, but nothing in it could make the thing all of that was about.

    The target is named for the runtime that actually served the measurement, so a CPU fallback is never
    filed as a GPU result. Boards this system does not host report their own numbers through
    POST /forgyx/benchmark instead: an Orin figure has to come from an Orin.
    """
    from services.forgyx.export import export_and_benchmark

    return await export_and_benchmark(db, model_version, imgsz=payload.imgsz, runs=payload.runs,
                                      prefer_cuda=payload.prefer_cuda)


@router.get("/forgyx/benchmarks/audit")
async def benchmarks_audit(db: AsyncSession = Depends(db_session)):
    """Which recorded benchmarks name an artifact that is not in object storage.

    Reports rather than deletes. A missing artifact can mean a fabricated row or a rotated bucket under a
    real measurement, and this cannot tell those apart; naming them is what lets somebody who knows decide.
    """
    return await audit_benchmarks(db)


class GateIn(BaseModel):
    baseline_latency: dict
    candidate_latency: dict
    champion_eval: dict              # {aggregate, per_slice}
    candidate_eval: dict
    protected_slices: list[str]


@router.post("/forgyx/gate")
async def gate(payload: GateIn):
    """The dual regression gate: block on latency regression OR protected-slice accuracy regression."""
    return dual_gate(payload.baseline_latency, payload.candidate_latency, payload.champion_eval,
                     payload.candidate_eval, payload.protected_slices)


class CooptIn(BaseModel):
    target: str
    base_latency_ms: float
    base_map50: float
    prune_grid: list[float] | None = None


@router.post("/forgyx/cooptimize")
async def cooptimize(payload: CooptIn):
    """Plan model-target co-optimization: rank (prune, int8) configs and pick the most accurate one that clears
    the target latency budget."""
    return plan_cooptimization(payload.target, payload.base_latency_ms, payload.base_map50, payload.prune_grid)


class ThermalIn(BaseModel):
    target: str
    readings: dict                   # device-farm run: {peak_temp_c, power_w, throttled_fps}
    cold_fps: float


@router.post("/forgyx/thermal")
async def thermal(payload: ThermalIn):
    """Check a device-farm sustained-load run against the target thermal/power envelope (refused without real
    readings)."""
    return thermal_envelope(payload.target, payload.readings, payload.cold_fps)


class RolloutIn(BaseModel):
    current_state: str               # none|canary|full|rolled_back
    action: str                      # canary|full|rolled_back


@router.post("/forgyx/rollout")
async def rollout(payload: RolloutIn):
    """The rollout/rollback transition for a deployment; refuses an illegal jump (e.g. none straight to full)."""
    return plan_rollout(payload.current_state, payload.action)
