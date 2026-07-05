"""FORGYX capability detection. The edge-optimization backends (ONNX, ONNX Runtime, TensorRT for Jetson,
LiteRT for Android/SentrixAI, Hailo Dataflow Compiler for the Pi+Hailo path) each require a toolchain or a
device that is not present in every environment. Rather than fabricate a result, FORGYX detects what is
installed and gates each target: a target whose backend is absent raises CapabilityError, and the real
export/compile/benchmark runs only where the backend exists. This is the honest seam the build rules require.
"""

from __future__ import annotations

import importlib.util


class CapabilityError(RuntimeError):
    """Raised when an edge target's backend is not installed on this machine."""


# target id -> the python module (or list) whose presence enables it
_BACKENDS: dict[str, tuple[str, ...]] = {
    "onnx": ("onnx",),
    "onnxruntime": ("onnxruntime",),
    "sentrixai_litert": ("ai_edge_litert",),           # or tflite_runtime
    "agx_orin_trt": ("tensorrt",),
    "orin_nano_trt": ("tensorrt",),
    "pi_hailo": ("hailo_sdk_client",),
}


def _has(mods: tuple[str, ...]) -> bool:
    return any(importlib.util.find_spec(m) is not None for m in mods)


def available_targets() -> dict[str, bool]:
    """Which optimization targets this machine can actually run."""
    return {t: _has(mods) for t, mods in _BACKENDS.items()}


def require(target: str) -> None:
    """Assert a target's backend is installed, else raise CapabilityError naming what is missing."""
    mods = _BACKENDS.get(target)
    if mods is None:
        raise CapabilityError(f"unknown target {target}")
    if not _has(mods):
        raise CapabilityError(
            f"target {target} needs {' or '.join(mods)}, not installed; "
            f"install the backend on a machine with the toolchain/device to enable it")
