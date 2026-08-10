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
    """Persist a measured (model, target) benchmark. Devices POST their measured latency/power here.

    A named artifact must exist. The table's first three rows named `s3://labeloxav/models/demo/*.bin`, which
    is not in object storage and never was, and the Pareto gate ranked them anyway. A benchmark whose
    artifact nobody can fetch is not a weak measurement, it is one that cannot be checked, and it outranks
    real ones because invented numbers are always flattering.

    A benchmark with no artifact at all is still allowed: a device reporting what it measured on hardware
    this system does not host is the normal case, and it is honest about having nothing to upload.
    """
    from services.forgyx.export import artifact_exists

    if artifact_uri and not artifact_exists(artifact_uri):
        return {"ok": False, "benchmark_id": None, "target": target,
                "reason": f"artifact {artifact_uri} is not in object storage; refusing to record a "
                          "benchmark that cannot be verified"}
    row = Benchmark(model_version=model_version, target=target, latency_ms=latency_ms,
                    throughput_fps=throughput_fps, power_w=power_w, accuracy_ref=accuracy_ref,
                    artifact_uri=artifact_uri)
    db.add(row)
    await db.commit()
    log.info("forgyx.benchmark", model=model_version, target=target, p95=latency_ms.get("p95"))
    return {"ok": True, "benchmark_id": str(row.benchmark_id), "target": target, "latency_ms": latency_ms}


async def audit_benchmarks(db: AsyncSession, limit: int = 1000) -> dict:
    """Which recorded benchmarks name an artifact that is not there.

    Written because three of them did, and nothing in the system would ever have said so. Reports rather
    than deletes: a missing artifact can also mean a bucket was rotated under a real measurement, and this
    cannot tell that apart from a fabrication. Naming them is what lets a human decide.
    """
    from services.forgyx.export import artifact_exists

    rows = (await db.execute(select(Benchmark).limit(limit))).scalars().all()
    unverifiable = []
    for b in rows:
        if b.artifact_uri and not artifact_exists(b.artifact_uri):
            unverifiable.append({"benchmark_id": str(b.benchmark_id), "model_version": b.model_version,
                                 "target": b.target, "artifact_uri": b.artifact_uri,
                                 "latency_ms": b.latency_ms})
    return {"n": len(rows), "n_unverifiable": len(unverifiable), "unverifiable": unverifiable,
            "detail": (f"{len(unverifiable)} of {len(rows)} benchmarks name an artifact that is not in "
                       "object storage" if unverifiable else
                       f"all {len(rows)} benchmarks with an artifact resolve to real bytes")}


async def record_deployment(db: AsyncSession, model_version: str, target: str, artifact_uri: str,
                            release_commit: str | None = None, verdict_ref: UUID | None = None,
                            benchmark_ref: UUID | None = None, status: str = "verified") -> dict:
    """Persist a deployable, verified artifact with lineage to the release, the VERDYX verdict, and the
    FORGYX benchmark that gated it. The manifest is HMAC-signed so a swapped artifact or forged verdict is
    detectable (verify_manifest); the signing key was defined but never applied here."""
    from core.config import get_settings
    from services.forgyx.packaging import sign_manifest

    export_format = _FORMAT.get(target, "onnx")
    manifest = {"model_version": model_version, "target": target, "artifact_uri": artifact_uri,
                "export_format": export_format, "release_commit": release_commit,
                "verdict_ref": str(verdict_ref) if verdict_ref else None,
                "benchmark_ref": str(benchmark_ref) if benchmark_ref else None, "status": status}
    signature = sign_manifest(manifest, get_settings().forgyx.deploy_signing_key)
    row = Deployment(model_version=model_version, target=target, artifact_uri=artifact_uri,
                     export_format=export_format, release_commit=release_commit,
                     verdict_ref=verdict_ref, benchmark_ref=benchmark_ref, status=status,
                     signature=signature)
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
