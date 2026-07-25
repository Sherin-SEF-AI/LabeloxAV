"""Batched sensor-QA image metrics (Tier 3). Sharpness (variance of Laplacian), exposure (mean luma), and
clipping (over/under-exposed fraction) for every frame of a session in one GPU pass, plus inter-camera exposure
consistency across a rig. This is the automated gate that flags bad captures before they poison the label
pipeline; the per-frame loop of cv2.Laplacian(...).var() it replaces is a real cost at session scale.

image_quality_batch matches services.ingest.quality (cv2.Laplacian variance with reflect-101 borders, mean
luma, clipped-pixel fraction). Runs on the GPU through torch when a CUDA device is present, else the identical
torch-CPU path, which is checked against cv2 in the tests."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# the cv2 default Laplacian kernel (ksize=1)
_LAP = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def image_quality_batch(imgs, clip_hi: float = 250.0, clip_lo: float = 5.0, device=None) -> dict:
    """Per-frame QA metrics for a batch of frames. Returns arrays over N frames: blur (variance of Laplacian,
    higher is sharper), mean_luma, clipped_hi (fraction >= clip_hi), clipped_lo (fraction <= clip_lo), plus a
    per-frame list of dicts."""
    if not _HAS_TORCH:
        raise RuntimeError("image_quality_batch needs torch")
    a = np.asarray(imgs)
    n = a.shape[0]
    dev = torch.device("cuda" if (device == "cuda" or (device != "cpu" and torch.cuda.is_available())) else "cpu")
    # transfer the raw (uint8) frames, not an upcast float copy, then convert to gray on the device: an 8x
    # smaller host->device copy. float32 is ample for a thresholded QA metric and runs at full rate on a GPU.
    raw = torch.as_tensor(a, device=dev)
    if raw.ndim == 3:                                    # (N, H, W) gray
        g = raw.to(torch.float32)
    elif raw.ndim == 4 and raw.shape[-1] == 3:           # (N, H, W, 3) BGR
        f = raw.to(torch.float32)
        g = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
    elif raw.ndim == 4 and raw.shape[1] == 3:            # (N, 3, H, W) BGR channel order
        f = raw.to(torch.float32)
        g = 0.114 * f[:, 0] + 0.587 * f[:, 1] + 0.299 * f[:, 2]
    else:
        raise ValueError("images must be (N,H,W), (N,H,W,3), or (N,3,H,W)")
    g = g.unsqueeze(1)                                   # (N,1,H,W)
    k = torch.as_tensor(_LAP, device=dev, dtype=torch.float32).view(1, 1, 3, 3)
    padded = F.pad(g, (1, 1, 1, 1), mode="reflect")      # BORDER_REFLECT_101, matching cv2.Laplacian
    lap = F.conv2d(padded, k)                            # (N,1,H,W)
    blur = lap.view(n, -1).var(dim=1, unbiased=False)    # cv2 .var() is population variance
    mean_luma = g.view(n, -1).mean(dim=1)
    clipped_hi = (g >= clip_hi).view(n, -1).float().mean(dim=1)
    clipped_lo = (g <= clip_lo).view(n, -1).float().mean(dim=1)

    blur_np = blur.cpu().numpy()
    luma_np = mean_luma.cpu().numpy()
    hi_np = clipped_hi.cpu().numpy()
    lo_np = clipped_lo.cpu().numpy()
    per_frame = [{"blur": float(blur_np[i]), "mean_luma": float(luma_np[i]),
                  "clipped_hi": float(hi_np[i]), "clipped_lo": float(lo_np[i])} for i in range(n)]
    return {"blur": blur_np, "mean_luma": luma_np, "clipped_hi": hi_np, "clipped_lo": lo_np,
            "per_frame": per_frame}


def rig_exposure_consistency(mean_lumas) -> dict:
    """Inter-camera exposure consistency across a rig: the spread of per-camera mean luma. A large spread means
    the cameras are exposing inconsistently (one washed out, one dark), which a QA gate should flag."""
    m = np.asarray(mean_lumas, dtype=np.float64).reshape(-1)
    if m.size == 0:
        return {"std": None, "range": None, "mean": None, "n": 0}
    return {"std": float(m.std()), "range": float(m.max() - m.min()), "mean": float(m.mean()), "n": int(m.size)}
