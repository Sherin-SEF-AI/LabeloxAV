"""Curation / dedup kernels (SIEVYX). The all-pairs and greedy operations under near-duplicate detection and
diversity sampling: an embedding cosine-similarity matrix, a perceptual-hash + Hamming matrix as a cheap coarse
near-dup filter upstream of the embedding pass, and the k-center greedy min-distance update that is the
bottleneck of coreset selection.

cosine_sim_matrix and hamming_matrix run on the GPU through torch when a CUDA device is present, else the
identical NumPy math (the oracle); kcenter_greedy accelerates the per-iteration distance update on the GPU while
keeping the argmax tie-break on the CPU so it matches services.sievyx.batch.select_coreset exactly."""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_EPS = 1e-12
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


# --------------------------------------------------------------------------- cosine similarity

def cosine_sim_matrix(embeddings, device=None) -> np.ndarray:
    """All-pairs cosine similarity (N, N) over row embeddings. L2-normalizes then does one GEMM."""
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    if _HAS_TORCH and device != "cpu" and torch.cuda.is_available():
        t = torch.as_tensor(X, device="cuda")
        t = t / t.norm(dim=1, keepdim=True).clamp_min(_EPS)
        return (t @ t.t()).cpu().numpy()
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), _EPS, None)
    return (Xn @ Xn.T).astype(np.float32)


def nearest_neighbor_sim(embeddings, device=None) -> np.ndarray:
    """Per-row nearest-neighbor cosine similarity (excluding self). Novelty = 1 - this; high novelty is a
    coverage gap, matching services.analytics.curation."""
    S = cosine_sim_matrix(embeddings, device=device)
    n = S.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    np.fill_diagonal(S, -np.inf)
    return S.max(axis=1)


# --------------------------------------------------------------------------- perceptual hash + Hamming

def phash_batch(imgs, hash_size=8, device=None) -> np.ndarray:
    """Perceptual hash (DCT-based) per frame. imgs (N, H, W) gray or (N, H, W, 3) BGR. Returns (N, 8) uint8, a
    64-bit hash per frame: the low-frequency DCT block thresholded at its median. Same-content frames hash
    close in Hamming distance, so it is a cheap coarse near-dup gate before the embedding pass."""
    if not _HAS_TORCH:
        raise RuntimeError("phash_batch needs torch")
    import torch.nn.functional as F

    a = np.asarray(imgs)
    if a.ndim == 4 and a.shape[-1] == 3:                              # BGR -> gray
        a = (0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2])
    elif a.ndim == 4 and a.shape[1] == 3:
        a = (0.114 * a[:, 0] + 0.587 * a[:, 1] + 0.299 * a[:, 2])
    n = a.shape[0]
    img_size = hash_size * 4
    dev = torch.device("cuda" if (device == "cuda" or (device != "cpu" and torch.cuda.is_available())) else "cpu")
    g = torch.as_tensor(a, device=dev).to(torch.float32).unsqueeze(1)         # (N,1,H,W)
    small = F.interpolate(g, size=(img_size, img_size), mode="bilinear", align_corners=False)[:, 0]  # (N,S,S)
    # DCT-II via a basis matrix: dct2 = D @ img @ D^T
    k = torch.arange(img_size, device=dev, dtype=torch.float32)
    D = torch.cos(np.pi * (k[:, None] + 0.0) * (2 * k[None, :] + 1) / (2 * img_size))   # (S,S)
    dct = D @ small @ D.t()                                          # (N,S,S)
    low = dct[:, :hash_size, :hash_size].reshape(n, -1)              # (N, 64)
    med = low.median(dim=1, keepdim=True).values
    bits = (low > med).to(torch.uint8)                              # (N, 64)
    packed = np.packbits(bits.cpu().numpy(), axis=1)                 # (N, 8) uint8
    return packed


def hamming_matrix(hashes, device=None, chunk: int = 512) -> np.ndarray:
    """All-pairs Hamming distance (N, N) over packed uint8 hashes (N, B). popcount(a XOR b) via a byte lookup,
    the coarse near-dup metric. Bit-packed and popcount-native. Computed in row-chunks so peak memory is
    (chunk, N, B), not (N, N, B): at N=20k, B=8 the full XOR tensor is ~3.2 GB and OOMs the GPU; chunked it is
    ~80 MB per block."""
    H = np.asarray(hashes, dtype=np.uint8)
    if H.ndim != 2 or H.shape[0] == 0:
        return np.zeros((H.shape[0], H.shape[0]), dtype=np.int64)
    n = H.shape[0]
    out = np.empty((n, n), dtype=np.int64)
    if _HAS_TORCH and device != "cpu" and torch.cuda.is_available():
        t = torch.as_tensor(H, device="cuda").to(torch.int32)
        lut = torch.as_tensor(_POPCOUNT8, device="cuda")
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            x = t[i:j, None, :] ^ t[None, :, :]                      # (chunk, N, B)
            out[i:j] = lut[x.long()].sum(dim=2).cpu().numpy()
        return out
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        x = H[i:j, None, :].astype(np.int32) ^ H[None, :, :].astype(np.int32)
        out[i:j] = _POPCOUNT8[x].sum(axis=2)
    return out


# --------------------------------------------------------------------------- k-center greedy

def kcenter_greedy(embeddings, budget: int, device=None) -> list[int]:
    """Greedy k-center (farthest-point) selection: start from the point farthest from the centroid, then
    repeatedly add the point farthest from the selected set, updating the distance-to-nearest-selected each
    iteration. Matches services.sievyx.batch.select_coreset; the per-iteration distance update runs on the GPU,
    the argmax tie-break on the CPU so the selection is identical."""
    X = np.asarray(embeddings, dtype=np.float64)
    n = X.shape[0]
    budget = min(max(1, budget), n)
    use_gpu = _HAS_TORCH and device != "cpu" and torch.cuda.is_available()
    if use_gpu:
        Xt = torch.as_tensor(X, device="cuda")

        def dist_to(i):
            return (Xt - Xt[i]).norm(dim=1).cpu().numpy()

        centroid = Xt.mean(0)
        first = int((Xt - centroid).norm(dim=1).argmax().item())
    else:
        def dist_to(i):
            return np.linalg.norm(X - X[i], axis=1)

        first = int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))

    selected = [first]
    dist = dist_to(first)
    while len(selected) < budget:
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, dist_to(nxt))
    return selected
