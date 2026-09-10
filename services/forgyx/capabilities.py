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
    # Diffusion harmonisation of copy-paste composites (services/synth). The slot is reserved so a caller
    # can ask for it and get a CapabilityError naming the missing backend; no harmoniser is built yet.
    "diffusion": ("diffusers",),
}

# Capabilities that are a binary on PATH rather than an importable module. `esmini` replays an
# OpenSCENARIO document headlessly, which is how an exported scenario gets checked for being a scenario
# at all rather than a well-formed document describing something impossible.
_BINARIES: dict[str, str] = {
    "esmini": "esmini",
}


def _has(mods: tuple[str, ...]) -> bool:
    return any(importlib.util.find_spec(m) is not None for m in mods)


def _binary_path(target: str) -> str | None:
    """Where this capability's binary is, or None. Config first, then PATH.

    Config first because a deployment that installed esmini somewhere unusual has said so, and searching
    PATH ahead of it would silently prefer a different build to the one an operator pointed at.
    """
    import shutil

    from core.config import get_settings

    name = _BINARIES.get(target)
    if name is None:
        return None
    configured = getattr(getattr(get_settings(), "sim", None), f"{target}_bin", None)
    if configured and shutil.which(str(configured)):
        return str(configured)
    return shutil.which(name)


def available_targets() -> dict[str, bool]:
    """Which optimization targets this machine can actually run."""
    return {**{t: _has(mods) for t, mods in _BACKENDS.items()},
            **{t: _binary_path(t) is not None for t in _BINARIES}}


def require(target: str) -> None:
    """Assert a target's backend is installed, else raise CapabilityError naming what is missing."""
    if target in _BINARIES:
        path = _binary_path(target)
        if path is None:
            raise CapabilityError(
                f"target {target} needs the `{_BINARIES[target]}` binary on PATH or at "
                f"sim.{target}_bin, and it is not there; install it to enable this check")
        return
    mods = _BACKENDS.get(target)
    if mods is None:
        raise CapabilityError(f"unknown target {target}")
    if not _has(mods):
        raise CapabilityError(
            f"target {target} needs {' or '.join(mods)}, not installed; "
            f"install the backend on a machine with the toolchain/device to enable it")
