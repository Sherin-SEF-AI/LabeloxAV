"""FORGYX persistence and orchestration: record a per-target Benchmark and a verified Deployment, and read the
benchmark matrix with Pareto ranking. The optimize/compile/benchmark stages run only where the backend exists
(capability-gated); a target whose toolchain is absent is recorded as capability-blocked, never faked."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Benchmark, Deployment
from services.forgyx.capabilities import available_targets
from services.forgyx.gate import pareto_rank

log = get_logger("forgyx.run")

_FORMAT = {"agx_orin_trt": "tensorrt", "orin_nano_trt": "tensorrt", "sentrixai_litert": "litert",
           "pi_hailo": "hailo", "onnx": "onnx"}


async def record_benchmark(db: AsyncSession, model_version: str, target: str, latency_ms: dict,
                           throughput_fps: float | None = None, power_w: float | None = None,
                           accuracy_ref: UUID | None = None, artifact_uri: str | None = None) -> dict:
    """Persist a measured (model, target) benchmark. Devices POST their measured latency/power here."""
    row = Benchmark(model_version=model_version, target=target, latency_ms=latency_ms,
                    throughput_fps=throughput_fps, power_w=power_w, accuracy_ref=accuracy_ref,
                    artifact_uri=artifact_uri)
    db.add(row)
    await db.commit()
    log.info("forgyx.benchmark", model=model_version, target=target, p95=latency_ms.get("p95"))
    return {"benchmark_id": str(row.benchmark_id), "target": target, "latency_ms": latency_ms}


async def record_deployment(db: AsyncSession, model_version: str, target: str, artifact_uri: str,
                            release_commit: str | None = None, verdict_ref: UUID | None = None,
                            benchmark_ref: UUID | None = None, status: str = "verified") -> dict:
    """Persist a deployable, verified artifact with lineage to the release, the VERDYX verdict, and the
    FORGYX benchmark that gated it."""
    row = Deployment(model_version=model_version, target=target, artifact_uri=artifact_uri,
                     export_format=_FORMAT.get(target, "onnx"), release_commit=release_commit,
                     verdict_ref=verdict_ref, benchmark_ref=benchmark_ref, status=status)
    db.add(row)
    await db.commit()
    log.info("forgyx.deploy", model=model_version, target=target, status=status)
    return {"deployment_id": str(row.deployment_id), "target": target, "status": status}


async def benchmark_matrix(db: AsyncSession, model_version: str | None = None) -> dict:
    """The benchmark matrix across (model, target), Pareto-ranked on latency vs accuracy."""
    from db.models import Evaluation

    q = select(Benchmark).order_by(Benchmark.created_at.desc())
    if model_version:
        q = q.where(Benchmark.model_version == model_version)
    rows = (await db.execute(q.limit(500))).scalars().all()
    items = []
    for r in rows:
        # denormalize the re-verified accuracy from the linked VERDYX evaluation, so the Pareto plot is
        # genuinely latency vs accuracy (not latency alone).
        map50 = None
        if r.accuracy_ref is not None:
            ev = await db.get(Evaluation, r.accuracy_ref)
            map50 = (ev.aggregate or {}).get("map50") if ev else None
        items.append({"benchmark_id": str(r.benchmark_id), "model_version": r.model_version, "target": r.target,
                      "latency_ms": r.latency_ms, "throughput_fps": r.throughput_fps, "power_w": r.power_w,
                      "map50": map50, "artifact_uri": r.artifact_uri,
                      "created_at": r.created_at.isoformat() if r.created_at else None})
    return {"capabilities": available_targets(), "benchmarks": pareto_rank(items) if items else []}
