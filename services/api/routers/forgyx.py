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
from services.forgyx.gate import dual_gate
from services.forgyx.run import benchmark_matrix, record_benchmark

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
