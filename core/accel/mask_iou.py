"""All-pairs mask IoU (Tier 1), a hand-written Triton kernel over bit-packed masks. Used constantly in the SAM
dedup, the recall-recovery overlap test, and active-learning triage, so all-pairs IoU over N masks is a hot
path. Each H x W binary mask is packed to 32 bits per word, and the kernel compares two masks with a bitwise
AND + SWAR popcount over the words instead of rasterizing the full grid: the intersection is popcount(A & B)
and the union is |A| + |B| - intersection from the precomputed areas, so no second popcount is needed.

The kernel runs on the GPU through Triton (no nvcc, no numpy 2, compiles on the RTX 5080 / sm_120) when a CUDA
device is present; otherwise the identical result comes from a NumPy raster reference, which is also the
correctness oracle the tests check the kernel against."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # torch/triton are optional; the NumPy path always works
    _HAS_TRITON = False


def gpu_available() -> bool:
    """True when the Triton kernel path will be taken by default."""
    return _HAS_TRITON and torch.cuda.is_available()


def pack_masks(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack (N, H, W) boolean masks to (N, Wp) uint32 words (bit i = pixel i), and return the per-mask areas.
    Bits are packed big-endian so a bitwise AND of two words is the per-bit intersection."""
    m = np.asarray(masks)
    n = m.shape[0]
    flat = m.reshape(n, -1).astype(np.uint8)
    areas = flat.sum(1).astype(np.int32)
    pad = (-flat.shape[1]) % 32
    if pad:
        flat = np.concatenate([flat, np.zeros((n, pad), np.uint8)], axis=1)
    packed = np.packbits(flat, axis=1).view(np.uint32)
    return np.ascontiguousarray(packed), areas


def _iou_matrix_np(masks: np.ndarray) -> np.ndarray:
    """Exact NumPy raster reference: intersection via a boolean matmul, union from areas."""
    M = np.asarray(masks).reshape(len(masks), -1).astype(np.float32)
    inter = M @ M.T
    area = M.sum(1)
    uni = area[:, None] + area[None, :] - inter
    return np.where(uni > 0, inter / np.maximum(uni, 1e-9), 0.0).astype(np.float32)


if _HAS_TRITON:

    @triton.jit
    def _mask_iou_kernel(pk_ptr, area_ptr, out_ptr, N, W, BJ: tl.constexpr, BW: tl.constexpr):
        i = tl.program_id(0)                                  # mask row
        jb = tl.program_id(1)                                 # block of columns
        js = jb * BJ + tl.arange(0, BJ)
        inter = tl.zeros((BJ,), tl.int32)
        for w0 in range(0, W, BW):
            ws = w0 + tl.arange(0, BW)
            wm = ws < W
            a = tl.load(pk_ptr + i * W + ws, mask=wm, other=0)                               # (BW,)
            b = tl.load(pk_ptr + js[:, None] * W + ws[None, :],
                        mask=(js < N)[:, None] & wm[None, :], other=0)                       # (BJ, BW)
            c = a[None, :] & b
            # SWAR popcount over each 32-bit word
            c = c - ((c >> 1) & 0x55555555)
            c = (c & 0x33333333) + ((c >> 2) & 0x33333333)
            c = (c + (c >> 4)) & 0x0F0F0F0F
            pc = (c * 0x01010101) >> 24
            inter += tl.sum(pc.to(tl.int32), axis=1)
        ai = tl.load(area_ptr + i)
        aj = tl.load(area_ptr + js, mask=js < N, other=0)
        uni = ai + aj - inter
        iou = tl.where(uni > 0, inter.to(tl.float32) / uni.to(tl.float32), 0.0)
        tl.store(out_ptr + i * N + js, iou, mask=js < N)

    def _iou_matrix_triton(masks: np.ndarray) -> np.ndarray:
        packed, areas = pack_masks(masks)
        n, w = packed.shape
        pk = torch.as_tensor(packed.view(np.int32), device="cuda")   # uint32 bit patterns in int32 storage
        ar = torch.as_tensor(areas, device="cuda")
        out = torch.zeros((n, n), device="cuda", dtype=torch.float32)
        bj, bw = 64, 32
        _mask_iou_kernel[(n, triton.cdiv(n, bj))](pk, ar, out, n, w, BJ=bj, BW=bw)
        torch.cuda.synchronize()
        return out.cpu().numpy()


def mask_iou_matrix(masks: np.ndarray, device: str | None = None) -> np.ndarray:
    """All-pairs IoU over (N, H, W) boolean masks, returning an (N, N) float32 matrix. Runs the bit-packed
    Triton kernel on the GPU when available; ``device="cpu"`` forces the NumPy reference."""
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("masks must be (N, H, W)")
    if masks.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if _HAS_TRITON and device != "cpu" and torch.cuda.is_available():
        return _iou_matrix_triton(masks)
    return _iou_matrix_np(masks)
