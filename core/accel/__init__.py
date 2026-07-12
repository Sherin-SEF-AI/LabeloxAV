"""GPU-accelerated kernels for the perception hot paths, capability-gated with a NumPy reference as both
the correctness oracle and the fallback (runs on the GPU through torch when a CUDA device is present, else the
identical NumPy math). Tier 1: fused projection."""

from core.accel.projection import gpu_available, project_cam_batch, project_world_batch

__all__ = ["project_world_batch", "project_cam_batch", "gpu_available"]
