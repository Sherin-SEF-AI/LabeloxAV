"""Mask ops for SAM 3.1 postprocess. Morphology to clean mask boundaries (erode/dilate/open/close as batched
max/min pooling on the GPU), a connected-component area filter to drop spurious fragments, an RLE codec so
masks move through the pipeline bit-packed, and a boundary-tightness QA metric. Morphology and tightness run on
the GPU through torch when a CUDA device is present, else the identical NumPy math (the oracle); the
connected-component labeling uses SciPy's fast C labeler and the RLE codec is exact NumPy."""

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


# --------------------------------------------------------------------------- morphology

def _pool(masks, ksize, op, device):
    """Batched (N, H, W) morphology by pooling with a ksize square structuring element (stride 1), replicate
    border. op 'dilate' = max pool, 'erode' = min pool. Returns float in {0,1}."""
    if not _HAS_TORCH:
        raise RuntimeError("morphology needs torch")
    m = np.asarray(masks, dtype=np.float32)
    single = m.ndim == 2
    if single:
        m = m[None]
    dev = torch.device("cuda" if (device == "cuda" or (device != "cpu" and torch.cuda.is_available())) else "cpu")
    t = torch.as_tensor(m, device=dev).unsqueeze(1)                  # (N,1,H,W)
    pad = ksize // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="replicate")
    if op == "dilate":
        out = F.max_pool2d(t, ksize, stride=1)
    else:                                                            # erode = -dilate(-x)
        out = -F.max_pool2d(-t, ksize, stride=1)
    res = out[:, 0].cpu().numpy()
    return res[0] if single else res


def dilate(masks, ksize=3, device=None):
    return _pool(masks, ksize, "dilate", device)


def erode(masks, ksize=3, device=None):
    return _pool(masks, ksize, "erode", device)


def morph_open(masks, ksize=3, device=None):
    """Erode then dilate: removes speckle without shrinking the main mask."""
    return dilate(erode(masks, ksize, device), ksize, device)


def morph_close(masks, ksize=3, device=None):
    """Dilate then erode: fills small holes in the mask."""
    return erode(dilate(masks, ksize, device), ksize, device)


# --------------------------------------------------------------------------- connected components / area filter

def remove_small_components(masks, min_area: int):
    """Drop connected components smaller than min_area pixels from each (N, H, W) binary mask, cleaning spurious
    SAM fragments. Uses SciPy's C connected-component labeler per mask. Returns the cleaned boolean masks."""
    from scipy import ndimage

    m = np.asarray(masks)
    single = m.ndim == 2
    if single:
        m = m[None]
    out = np.zeros_like(m, dtype=bool)
    for i, mask in enumerate(m):
        lab, n = ndimage.label(mask > 0)
        if n == 0:
            continue
        counts = np.bincount(lab.ravel())
        keep = np.where(counts >= min_area)[0]
        keep = keep[keep != 0]                                       # 0 is background
        out[i] = np.isin(lab, keep)
    return out[0] if single else out


# --------------------------------------------------------------------------- RLE codec

def rle_encode(mask) -> dict:
    """Row-major run-length encode a binary mask to {size: [H, W], counts: [run lengths starting from 0]}. The
    packed form masks move through the pipeline in; rle_decode inverts it exactly."""
    m = np.asarray(mask) > 0
    H, W = m.shape
    flat = m.ravel()
    # run lengths, always starting with the count of leading zeros (0 if the first pixel is set)
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], changes, [flat.size]])
    runs = np.diff(bounds).tolist()
    counts = runs if not flat[0] else [0, *runs]
    return {"size": [H, W], "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    """Invert rle_encode to a boolean (H, W) mask."""
    H, W = rle["size"]
    flat = np.zeros(H * W, dtype=bool)
    idx = 0
    val = False
    for c in rle["counts"]:
        if val:
            flat[idx:idx + c] = True
        idx += c
        val = not val
    return flat.reshape(H, W)


# --------------------------------------------------------------------------- mask-quality QA

def boundary_tightness(masks, ksize=3, device=None) -> np.ndarray:
    """A per-mask boundary-tightness score in [0, 1]: the fraction of the mask that is NOT boundary (mask minus
    its erosion). A tight, compact mask scores high; a thin or ragged mask has a large boundary fraction and
    scores low. Batched, GPU-accelerated via the erosion."""
    m = np.asarray(masks, dtype=np.float32)
    single = m.ndim == 2
    if single:
        m = m[None]
    er = erode(m, ksize, device)
    area = m.reshape(m.shape[0], -1).sum(axis=1)
    boundary = (m - er).reshape(m.shape[0], -1).sum(axis=1)
    tight = np.divide(area - boundary, area, out=np.zeros_like(area), where=area > 0)
    return tight[0] if single else tight
