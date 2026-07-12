"""Sensor-QA extras (SANYX). Batched histogram equalization across the rig (a QA normalization and exposure
diagnostic) and a per-frame motion-blur / directional-blur estimate to flag captures where fast ego motion
smeared the frame, since those poison labels silently. Both run over a whole session at once; the motion-blur
estimate is a gradient-statistics kernel (structure tensor), GPU-accelerated when a CUDA device is present.

equalize_hist_batch matches cv2.equalizeHist per frame exactly; motion_blur_score returns a blur extent (low
gradient energy = blurred) and a directional anisotropy + orientation from the gradient structure tensor."""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_EPS = 1e-12


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def _equalize_one(img):
    """cv2.equalizeHist exactly for one uint8 image."""
    hist = np.bincount(img.ravel(), minlength=256).astype(np.int64)
    cdf = np.cumsum(hist)
    total = cdf[-1]
    nz = hist > 0
    if not nz.any():
        return img.copy()
    cdf_min = cdf[np.argmax(nz)]                                     # first non-zero cdf value
    denom = total - cdf_min
    if denom <= 0:
        return img.copy()
    lut = np.round((cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
    return lut[img]


def equalize_hist_batch(imgs) -> np.ndarray:
    """Batched global histogram equalization of (N, H, W) uint8 gray frames, matching cv2.equalizeHist per
    frame. A QA normalization that also makes exposure differences across the rig comparable."""
    a = np.asarray(imgs, dtype=np.uint8)
    single = a.ndim == 2
    if single:
        a = a[None]
    out = np.stack([_equalize_one(f) for f in a])
    return out[0] if single else out


def motion_blur_score(imgs, device=None) -> dict:
    """Per-frame motion-blur estimate from gradient statistics. imgs (N, H, W) gray. Returns arrays over N
    frames: blur_extent (mean gradient magnitude; a smeared frame has low gradient energy), anisotropy
    (0 isotropic .. 1 fully directional, from the structure-tensor eigenvalues; a motion blur suppresses
    gradients along the motion direction, raising anisotropy), and blur_angle (radians, the dominant gradient
    orientation). A sharp frame has high blur_extent and low anisotropy."""
    a = np.asarray(imgs, dtype=np.float64)
    single = a.ndim == 2
    if single:
        a = a[None]
    n = a.shape[0]

    if _HAS_TORCH and device != "cpu" and (device == "cuda" or torch.cuda.is_available()):
        import torch.nn.functional as F
        dev = torch.device("cuda")
        t = torch.as_tensor(a, device=dev, dtype=torch.float32).unsqueeze(1)
        kx = torch.tensor([[-1.0, 0, 1]], device=dev).view(1, 1, 1, 3)
        ky = torch.tensor([[-1.0], [0], [1]], device=dev).view(1, 1, 3, 1)
        gx = F.conv2d(F.pad(t, (1, 1, 0, 0), mode="replicate"), kx)[:, 0]
        gy = F.conv2d(F.pad(t, (0, 0, 1, 1), mode="replicate"), ky)[:, 0]
        gx, gy = gx.reshape(n, -1), gy.reshape(n, -1)
        jxx = (gx * gx).mean(1)
        jyy = (gy * gy).mean(1)
        jxy = (gx * gy).mean(1)
        mag = torch.sqrt(gx * gx + gy * gy).mean(1)
        jxx, jyy, jxy, mag = (v.cpu().numpy() for v in (jxx, jyy, jxy, mag))
    else:
        gy, gx = np.gradient(a, axis=(1, 2))
        gxf, gyf = gx.reshape(n, -1), gy.reshape(n, -1)
        jxx = (gxf * gxf).mean(1)
        jyy = (gyf * gyf).mean(1)
        jxy = (gxf * gyf).mean(1)
        mag = np.sqrt(gxf * gxf + gyf * gyf).mean(1)

    tr = jxx + jyy
    diff = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0.0))
    lam1 = (tr + diff) / 2.0
    lam2 = (tr - diff) / 2.0
    anisotropy = np.divide(lam1 - lam2, lam1 + lam2, out=np.zeros(n), where=(lam1 + lam2) > _EPS)
    angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)                     # dominant gradient orientation
    res = {"blur_extent": mag, "anisotropy": anisotropy, "blur_angle": angle}
    if single:
        res = {k: v[0] for k, v in res.items()}
    return res
