"""FORGYX benchmark: measure p50/p95/p99 latency and throughput for a compiled model, and power where the
device exposes it (tegrastats/INA on Jetson). Latency for the ONNX Runtime path is measurable on any machine
with onnxruntime; TensorRT/LiteRT/Hailo latency is measured on the target device. Power is capability-gated and
returns None off-device rather than a fabricated number."""

from __future__ import annotations

import shutil
import subprocess
import time

from core.logging import get_logger
from services.forgyx.capabilities import require

log = get_logger("forgyx.benchmark")


def benchmark_onnx_latency(onnx_path: str, input_shape: tuple, provider: str = "CPUExecutionProvider",
                           n_iters: int = 100, warmup: int = 10) -> dict:
    """Time an ONNX model over n_iters with a random input, returning p50/p95/p99 latency (ms) and throughput.
    provider selects the ONNX Runtime execution provider (CPU / CUDA)."""
    require("onnxruntime")
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=[provider])
    name = sess.get_inputs()[0].name
    x = np.random.rand(*input_shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p = lambda q: round(times[min(len(times) - 1, int(q * len(times)))], 3)
    mean_ms = sum(times) / len(times)
    return {"p50": p(0.50), "p95": p(0.95), "p99": p(0.99),
            "throughput_fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
            "provider": provider, "n_iters": n_iters}


def read_power_w() -> float | None:
    """Instantaneous board power (W) from tegrastats on Jetson. Returns None where tegrastats is absent, never
    a fabricated value."""
    if shutil.which("tegrastats") is None:
        return None
    try:
        out = subprocess.run(["tegrastats", "--interval", "100"], capture_output=True, text=True,
                             timeout=1.5).stdout
    except Exception:  # noqa: BLE001
        return None
    # tegrastats prints e.g. "VDD_IN 5100mW/..." parse the total board rail
    for tok in out.split():
        if tok.endswith("mW/") or tok.endswith("mW"):
            try:
                return round(float(tok.rstrip("mW/").split("/")[0]) / 1000.0, 2)
            except ValueError:
                continue
    return None
