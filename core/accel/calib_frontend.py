"""Calibration front-end kernels (CALYX). Sub-pixel corner refinement across the full multi-camera capture
(the gradient least-squares refinement cv2.cornerSubPix does, vectorized over the whole detection set), and
binary-descriptor all-pairs Hamming matching for the cross-camera correspondence that seeds extrinsic
calibration and any SfM front end (popcount-native, with the ratio test in the tail).

descriptor_match runs the all-pairs Hamming on the GPU through torch when a CUDA device is present, else the
identical NumPy math; refine_corners is vectorized over corners (batched 2x2 solves) and converges to the true
corner, matching cv2.cornerSubPix closely."""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


# --------------------------------------------------------------------------- sub-pixel corner refinement

def refine_corners(gray, corners, win: int = 5, iters: int = 10, eps: float = 1e-3) -> np.ndarray:
    """Sub-pixel refinement of corners, vectorized over the whole set. gray (H, W) float; corners (N, 2) as
    (x, y). For each corner the algorithm solves, over a (2*win+1) window, for the point where the image
    gradients are orthogonal to the displacement (the cv2.cornerSubPix normal equations A c = b, A = sum g g^T,
    b = sum (g g^T) p), iterating a few times. Returns the refined (N, 2) corners."""
    g = np.asarray(gray, dtype=np.float64)
    H, W = g.shape
    gy, gx = np.gradient(g)
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2).copy()
    off = np.arange(-win, win + 1)
    oy, ox = np.meshgrid(off, off, indexing="ij")                    # (w, w)
    ox = ox.ravel()
    oy = oy.ravel()
    # Gaussian window weight (cv2 uses a similar mask), so nearer pixels count more
    wgt = np.exp(-(ox ** 2 + oy ** 2) / (2.0 * (win / 2.0) ** 2))

    for _ in range(iters):
        cx = np.round(c[:, 0]).astype(int)
        cy = np.round(c[:, 1]).astype(int)
        px = np.clip(cx[:, None] + ox[None, :], 0, W - 1)            # (N, w*w)
        py = np.clip(cy[:, None] + oy[None, :], 0, H - 1)
        Gx = gx[py, px]                                              # (N, w*w)
        Gy = gy[py, px]
        wgx = wgt[None, :] * Gx
        wgy = wgt[None, :] * Gy
        a11 = (wgx * Gx).sum(1)                                      # A = sum g g^T, weighted
        a12 = (wgx * Gy).sum(1)
        a22 = (wgy * Gy).sum(1)
        # b = sum (g g^T) p, p = (px, py)
        b1 = (wgx * Gx * px + wgx * Gy * py).sum(1)
        b2 = (wgy * Gx * px + wgy * Gy * py).sum(1)
        det = a11 * a22 - a12 * a12
        ok = np.abs(det) > 1e-12
        nx = np.where(ok, (a22 * b1 - a12 * b2) / np.where(ok, det, 1.0), c[:, 0])
        ny = np.where(ok, (a11 * b2 - a12 * b1) / np.where(ok, det, 1.0), c[:, 1])
        new = np.stack([nx, ny], axis=1)
        # keep moves inside the window; a wild solve means a flat/degenerate corner, leave it
        move = np.linalg.norm(new - c, axis=1)
        settle = move < win
        c = np.where(settle[:, None], new, c)
        if np.all(move < eps):
            break
    return c


# --------------------------------------------------------------------------- descriptor Hamming matching

def _hamming_all_pairs(a, b, device):
    """(Na, Nb) Hamming distance between packed uint8 binary descriptors (Na, B) and (Nb, B)."""
    if _HAS_TORCH and device != "cpu" and torch.cuda.is_available():
        ta = torch.as_tensor(a, device="cuda").to(torch.int32)
        tb = torch.as_tensor(b, device="cuda").to(torch.int32)
        lut = torch.as_tensor(_POPCOUNT8, device="cuda")
        x = ta[:, None, :] ^ tb[None, :, :]
        return lut[x.long()].sum(dim=2).cpu().numpy()
    x = a[:, None, :].astype(np.int32) ^ b[None, :, :].astype(np.int32)
    return _POPCOUNT8[x].sum(axis=2)


def descriptor_match(desc_a, desc_b, ratio: float = 0.75, mutual: bool = True, device=None) -> dict:
    """Match two sets of binary descriptors (ORB-style, packed uint8) by Hamming distance with Lowe's ratio
    test. desc_a (Na, B), desc_b (Nb, B). Returns {matches (K, 2) [index_a, index_b], distances (K,)} for the
    a-descriptors whose best match beats the second-best by the ratio (and, when mutual, is each other's best).
    The all-pairs Hamming runs on the GPU; the ratio-test tail is cheap."""
    a = np.asarray(desc_a, dtype=np.uint8).reshape(len(desc_a), -1)
    b = np.asarray(desc_b, dtype=np.uint8).reshape(len(desc_b), -1)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {"matches": np.zeros((0, 2), np.int64), "distances": np.zeros(0)}
    D = _hamming_all_pairs(a, b, device)                             # (Na, Nb)
    order = np.argsort(D, axis=1)
    best = order[:, 0]
    second = order[:, 1] if D.shape[1] > 1 else order[:, 0]
    best_d = D[np.arange(D.shape[0]), best]
    second_d = D[np.arange(D.shape[0]), second]
    passes = best_d < ratio * np.maximum(second_d, 1e-9)
    if D.shape[1] == 1:
        passes = np.ones(D.shape[0], bool)                          # no second-best to compare
    if mutual:
        b_best = np.argmin(D, axis=0)                               # each b's best a
        passes &= (b_best[best] == np.arange(D.shape[0]))
    idx = np.flatnonzero(passes)
    matches = np.stack([idx, best[idx]], axis=1)
    return {"matches": matches.astype(np.int64), "distances": best_d[idx]}
