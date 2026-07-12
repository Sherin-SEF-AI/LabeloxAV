"""Fused auto-label preprocess (Tier 1). NV12/YUV -> RGB -> letterbox -> normalize -> NCHW in one GPU pass,
batched across the cameras of a rig, writing a contiguous tensor ready for an engine's input binding. This is
the classic hidden latency tax on every auto-label run: the CPU path decodes with cv2, resizes with cv2,
normalizes, transposes HWC->CHW, and copies host->device, three or four separate passes over the pixels. Doing
the whole thing on the GPU collapses them into one vectorized pipeline that feeds all three paths (YOLO26,
SAM 3.1, Qwen3-VL).

Runs on the GPU through torch when a CUDA device is present, else the identical torch-CPU path (so the GPU
result matches the CPU result exactly, since it is the same code). The color convert follows BT.601 full-range
like cv2's COLOR_YUV2RGB_NV12, and the letterbox follows the Ultralytics convention (aspect-preserving resize,
pad 114, centered), so a box detected on the letterboxed image un-maps with (scale, pad)."""

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


def letterbox_params(src_hw: tuple[int, int], out_hw: tuple[int, int]) -> dict:
    """The aspect-preserving resize scale and centered padding for a source into an output canvas, so a
    detection on the letterboxed image maps back to source pixels: (x - pad_x) / scale."""
    h, w = src_hw
    oh, ow = out_hw
    scale = min(oh / h, ow / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    pad_y, pad_x = (oh - nh) // 2, (ow - nw) // 2
    return {"scale": scale, "nh": nh, "nw": nw, "pad_x": pad_x, "pad_y": pad_y}


def _nv12_to_rgb(y_uv, H, W):
    """(N, H*3/2, W) uint8 NV12 -> (N, 3, H, W) float RGB in [0,255], torch. BT.601 full-range."""
    Y = y_uv[:, :H, :].float()                                   # (N, H, W)
    uv = y_uv[:, H:, :].reshape(y_uv.shape[0], H // 2, W // 2, 2).float()
    U = uv[..., 0]
    V = uv[..., 1]
    # upsample chroma to full resolution (nearest, matching cv2's 4:2:0 replication)
    U = U.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # (N, H, W)
    V = V.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    # BT.601 limited (studio) range, matching cv2's COLOR_YUV2RGB_NV12 fixed-point coefficients
    yl = 1.164 * (Y - 16.0)
    u = U - 128.0
    v = V - 128.0
    r = yl + 1.596 * v
    g = yl - 0.391 * u - 0.813 * v
    b = yl + 2.018 * u
    rgb = torch.stack([r, g, b], dim=1)                          # (N, 3, H, W)
    return rgb.clamp_(0.0, 255.0)


def _apply_undistort(rgb, map_x, map_y, dev):
    """Apply a precomputed fisheye undistort map to (N, 3, H, W) RGB on the device, staying resident so the
    OpenCV remap drops out of the preprocess path. map_x, map_y are (H, W) source-pixel maps."""
    _, _, H, W = rgb.shape
    x = torch.as_tensor(np.asarray(map_x, dtype=np.float32), device=dev)
    y = torch.as_tensor(np.asarray(map_y, dtype=np.float32), device=dev)
    gx = (2.0 * x + 1.0) / W - 1.0
    gy = (2.0 * y + 1.0) / H - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(rgb.shape[0], H, W, 2)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def preprocess_nv12_batch(nv12, src_hw, out_hw=(640, 640), mean=None, std=None, undistort_map=None, device=None):
    """Fused preprocess of a batch of NV12 frames into a model-ready NCHW tensor.

    nv12: (N, H*3/2, W) uint8 (or a list of such). src_hw = (H, W). Returns (batch (N, 3, out_h, out_w) float32,
    meta dict with the letterbox scale/pad for un-mapping boxes). ``mean``/``std`` (len-3, in [0,1]) apply an
    optional channel normalization after the /255 scale. ``undistort_map`` (map_x, map_y) applies a fisheye
    undistort on the device between the color convert and the letterbox, so a wide rig frame is rectified
    without a separate cv2.remap. ``device`` forces "cpu"/"cuda"; the GPU is used when available."""
    if not _HAS_TORCH:
        raise RuntimeError("preprocess_nv12_batch needs torch")
    H, W = src_hw
    oh, ow = out_hw
    arr = np.asarray(nv12, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[None]
    dev = torch.device("cuda" if (device == "cuda" or (device != "cpu" and torch.cuda.is_available())) else "cpu")
    t = torch.as_tensor(arr, device=dev)                         # (N, H*3/2, W) uint8

    rgb = _nv12_to_rgb(t, H, W)                                  # (N, 3, H, W) [0,255]
    if undistort_map is not None:
        rgb = _apply_undistort(rgb, undistort_map[0], undistort_map[1], dev)
    lb = letterbox_params((H, W), (oh, ow))
    resized = F.interpolate(rgb, size=(lb["nh"], lb["nw"]), mode="bilinear", align_corners=False)
    canvas = torch.full((rgb.shape[0], 3, oh, ow), 114.0, device=dev)
    canvas[:, :, lb["pad_y"]:lb["pad_y"] + lb["nh"], lb["pad_x"]:lb["pad_x"] + lb["nw"]] = resized
    out = canvas / 255.0
    if mean is not None and std is not None:
        m = torch.as_tensor(mean, device=dev, dtype=out.dtype).view(1, 3, 1, 1)
        s = torch.as_tensor(std, device=dev, dtype=out.dtype).view(1, 3, 1, 1)
        out = (out - m) / s
    return out.contiguous(), {**lb, "src_hw": (H, W), "out_hw": (oh, ow), "device": dev.type,
                              "undistorted": undistort_map is not None}
