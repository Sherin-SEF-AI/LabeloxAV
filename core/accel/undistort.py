"""Undistort / rectify LUT apply (Tier 3). The fisheye remap map is a function of the calibration only, so it
is precomputed once; applying it to every frame is a per-pixel gather. Instead of a separate cv2.remap per
frame on the CPU, the map is applied as a batched GPU gather (grid_sample), which fuses naturally into the
preprocess pipeline. The map build reuses cv2.fisheye.initUndistortRectifyMap (cheap, once per camera); the
apply runs on the GPU through torch when a CUDA device is present, else the identical torch-CPU path.

apply_map_batch is checked against cv2.remap on the map interior in the tests (bilinear grid_sample vs cv2's
bilinear remap agree away from the extrapolated border)."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def build_fisheye_map(K, dist, size, new_K=None):
    """Precompute the fisheye undistort-rectify map for a camera (once). K (3,3), dist (4,) Kannala-Brandt,
    size (W, H). Returns (map_x, map_y) float32 pixel-coordinate maps, each (H, W). new_K defaults to K."""
    import cv2

    K = np.asarray(K, dtype=np.float64)
    d = np.asarray(dist, dtype=np.float64).reshape(-1, 1)[:4]
    P = np.asarray(new_K, dtype=np.float64) if new_K is not None else K
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, d, np.eye(3), P, tuple(size), cv2.CV_32FC1)
    return map_x, map_y


def apply_map_batch(imgs, map_x, map_y, device=None) -> np.ndarray:
    """Apply a precomputed remap to a batch of frames as one GPU gather. imgs (N, H, W) gray or (N, H, W, C);
    map_x, map_y (H, W) pixel-coordinate maps. Returns the remapped batch in the same layout, bilinear-sampled.
    Out-of-bounds samples read as zero (matching cv2.remap with BORDER_CONSTANT 0)."""
    if not _HAS_TORCH:
        raise RuntimeError("apply_map_batch needs torch")
    a = np.asarray(imgs)
    single_channel = a.ndim == 3
    if single_channel:
        a = a[..., None]                                 # (N, H, W, 1)
    n, H, W, C = a.shape
    dev = torch.device("cuda" if (device == "cuda" or (device != "cpu" and torch.cuda.is_available())) else "cpu")
    x = torch.as_tensor(np.asarray(map_x, dtype=np.float32), device=dev)      # (H, W) target pixel -> source x
    y = torch.as_tensor(np.asarray(map_y, dtype=np.float32), device=dev)
    # normalize source pixel coords to grid_sample's [-1, 1] (align_corners=False, pixel-center convention)
    gx = (2.0 * x + 1.0) / W - 1.0
    gy = (2.0 * y + 1.0) / H - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(n, H, W, 2)      # (N, H, W, 2)
    # transfer the raw frames and cast on the device (a small host->device copy for uint8 input)
    t = torch.as_tensor(a, device=dev).to(torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out.permute(0, 2, 3, 1).cpu().numpy()          # (N, H, W, C)
    if single_channel:
        out = out[..., 0]
    return out
