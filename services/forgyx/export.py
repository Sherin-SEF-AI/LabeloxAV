"""Exporting a registered model to a real artifact, and timing that artifact on a real runtime.

FORGYX had a benchmark table, a deployment table, a Pareto gate, a signed manifest and a rollout planner, and
between them they described three benchmarks for a model called `demo-challenger` on `agx_orin_trt`,
`orin_nano_trt` and `sentrixai_litert`. Their artifacts, `s3://labeloxav/models/demo/*.bin`, are not in object
storage and never were. The numbers alongside them (p50 4.2ms, 178 fps, 22W) were written by hand.

So the whole tier rested on three rows of fiction, and the Pareto gate that ranks targets was ranking them.

There was also no export function anywhere in the module. Nothing could have produced an artifact even in
principle: `packaging.py` builds and signs a manifest *about* an artifact, and assumes somebody else made one.

This makes one. It exports a registered model's weights to ONNX, uploads the bytes, and records what they
hash to and how large they are, so a downstream manifest signs something that exists. Then it times that
artifact through onnxruntime and reports the measured distribution.

The targets it can honestly speak about are the ones present on the machine doing the measuring, which is why
they are named for the runtime and the provider (`onnx_cpu`, `onnx_cuda`) rather than for a board nobody
here has. An Orin number has to come from an Orin. The point of this module is that it refuses to invent one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from core.logging import get_logger
from core.storage import get_object_store

log = get_logger("forgyx.export")

ARTIFACT_KEY_ROOT = "artifacts"

# Timing defaults. The warmup is not decoration: the first inferences pay for lazy kernel compilation and
# allocator growth, and folding them into the sample turns a p99 into a measure of startup.
DEFAULT_WARMUP = 10
DEFAULT_RUNS = 50


def percentiles(samples_ms: list[float]) -> dict:
    """p50/p95/p99 plus the range, over measured milliseconds.

    Reported as a distribution rather than a mean because latency is what a device budget is spent on, and a
    mean hides exactly the tail that blows the budget. An empty sample returns nothing rather than zeros: a
    zero here would read as an infinitely fast model.
    """
    if not samples_ms:
        return {}
    a = np.asarray(sorted(samples_ms), dtype=np.float64)
    return {
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "min": round(float(a[0]), 3),
        "max": round(float(a[-1]), 3),
        "n": int(a.size),
    }


def sha256_file(path: str | Path) -> str:
    """Content hash of an artifact, so a manifest signs the bytes rather than the path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_exists(uri: str | None) -> bool:
    """Whether an artifact URI resolves to bytes that are actually there.

    The guard that the three demo rows would have failed. A benchmark naming an artifact nobody can fetch is
    not a weak measurement, it is an unfalsifiable one.
    """
    if not uri:
        return False
    try:
        return bool(get_object_store().exists(uri))
    except Exception:  # noqa: BLE001 - an unreachable store must not be read as "the artifact is fine"
        return False


def export_to_onnx(weights_path: str | Path, out_dir: str | Path, *, imgsz: int = 640,
                   opset: int = 12) -> dict:
    """Export Ultralytics weights to ONNX and return the local artifact and its hash.

    Kept separate from the upload and from the database so the expensive, environment-dependent half can be
    exercised on its own.
    """
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    produced = YOLO(str(weights_path)).export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False)
    took = time.perf_counter() - t0

    src = Path(str(produced))
    dst = out_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.move(str(src), str(dst))
    return {
        "path": str(dst),
        "format": "onnx",
        "bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
        "imgsz": imgsz,
        "opset": opset,
        "export_seconds": round(took, 2),
    }


def benchmark_onnx(model_path: str | Path, *, imgsz: int = 640, runs: int = DEFAULT_RUNS,
                   warmup: int = DEFAULT_WARMUP, prefer_cuda: bool = True) -> dict:
    """Time an ONNX artifact on this machine and report the measured latency distribution.

    The target name carries the provider that actually served the run, not the one that was requested.
    onnxruntime silently falls back to CPU when a CUDA provider fails to initialise, and a CPU number filed
    under a GPU target is the same class of fiction this module exists to remove.
    """
    import onnxruntime as ort

    want = (["CUDAExecutionProvider", "CPUExecutionProvider"] if prefer_cuda
            else ["CPUExecutionProvider"])
    available = ort.get_available_providers()
    providers = [p for p in want if p in available] or ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(model_path), providers=providers)
    served = sess.get_providers()[0]

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else s
             for d, s in zip(inp.shape, [1, 3, imgsz, imgsz], strict=False)]
    x = np.random.rand(*shape).astype(np.float32)
    feed = {inp.name: x}

    for _ in range(max(0, warmup)):
        sess.run(None, feed)

    samples: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        sess.run(None, feed)
        samples.append((time.perf_counter() - t0) * 1000.0)

    lat = percentiles(samples)
    fps = round(1000.0 / lat["p50"], 2) if lat.get("p50") else None
    target = "onnx_cuda" if served == "CUDAExecutionProvider" else "onnx_cpu"
    return {
        "target": target,
        "provider": served,
        "latency_ms": lat,
        "throughput_fps": fps,
        "input_shape": list(shape),
        # Named so nobody reads a laptop number as a board number later.
        "measured_on": os.uname().nodename,
        "runs": len(samples),
        "warmup": warmup,
    }


async def export_and_benchmark(db, model_version: str, *, imgsz: int = 640, runs: int = DEFAULT_RUNS,
                               prefer_cuda: bool = True, upload: bool = True) -> dict:
    """Export a registered model, upload the artifact, time it, and persist the benchmark.

    Refuses a model whose weights are not fetchable, rather than recording a target's numbers against nothing.
    `external://caller-hosted` models are the common case there: the caller runs them, so there is no
    artifact for this system to produce or measure.
    """
    from sqlalchemy import select

    from db.models import ModelRegistry
    from services.forgyx.run import record_benchmark

    reg = (await db.execute(
        select(ModelRegistry).where(ModelRegistry.model_version == model_version))).scalar_one_or_none()
    if reg is None:
        return {"ok": False, "reason": f"model {model_version} is not registered"}
    uri = reg.weights_uri or ""
    if not uri.startswith("s3://"):
        return {"ok": False, "reason": f"weights are {uri or 'absent'}; nothing local to export"}

    store = get_object_store()
    if not artifact_exists(uri):
        return {"ok": False, "reason": f"weights {uri} are not in object storage"}

    with tempfile.TemporaryDirectory(prefix="forgyx-") as tmp:
        local = Path(tmp) / "weights.pt"
        local.write_bytes(store.get_bytes(uri))
        art = export_to_onnx(local, tmp, imgsz=imgsz)

        # Content-addressed by the artifact's own hash, so re-exporting identical weights lands on the same
        # object instead of accumulating near-duplicates that a manifest cannot tell apart.
        key = f"{ARTIFACT_KEY_ROOT}/{model_version}/onnx/model-{art['sha256'][:12]}.onnx"
        artifact_uri = store.uri(key)
        if upload:
            store.put_file(key, art["path"])

        bench = benchmark_onnx(art["path"], imgsz=imgsz, runs=runs, prefer_cuda=prefer_cuda)

    rec = await record_benchmark(
        db, model_version=model_version, target=bench["target"], latency_ms=bench["latency_ms"],
        throughput_fps=bench["throughput_fps"], artifact_uri=artifact_uri if upload else None)

    log.info("forgyx.export_and_benchmark", model=model_version, target=bench["target"],
             p50=bench["latency_ms"].get("p50"), bytes=art["bytes"], sha=art["sha256"][:12])
    return {"ok": True, "model_version": model_version, "artifact": {**art, "uri": artifact_uri},
            "benchmark": bench, "benchmark_id": rec.get("benchmark_id")}
