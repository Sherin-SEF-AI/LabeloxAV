"""Active-learning uncertainty kernels (the ORACLYX -> VERDYX signal). The scalars triage ranks on: per-detection
softmax entropy and top1/top2 margin, ensemble disagreement (BALD-style mutual information across the three
auto-label paths), and temporal flicker (box jitter across consecutive frames for the same object). High
flicker with high confidence is a classic auto-label failure signature that scalar confidence misses entirely,
so this turns a vibe into a number.

All three run batched across every detection or track in a session, on the GPU through torch when a CUDA device
is present, else the identical NumPy math which is also the correctness oracle."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_EPS = 1e-12


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def entropy_margin(logits, normalize=True, device=None) -> dict:
    """Per-detection uncertainty from class logits (N, C). Returns {entropy (N,), margin (N,) = top1 - top2
    probability, top1 (N,) class index, top1_conf (N,)}. When normalize, entropy is divided by log(C) so it is
    in [0, 1] and comparable across ontologies. High entropy or small margin = uncertain."""
    a = np.asarray(logits, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    n, C = a.shape
    if _HAS_TORCH and device != "cpu" and (device == "cuda" or torch.cuda.is_available()):
        dev = torch.device("cuda")
        t = torch.as_tensor(a, device=dev)
        logp = F.log_softmax(t, dim=1)
        p = logp.exp()
        ent = -(p * logp).sum(dim=1)
        if normalize and C > 1:
            ent = ent / np.log(C)
        top2 = torch.topk(p, min(2, C), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if C > 1 else torch.zeros_like(top2[:, 0]))
        top1 = p.argmax(dim=1)
        return {"entropy": ent.cpu().numpy(), "margin": margin.cpu().numpy(),
                "top1": top1.cpu().numpy(), "top1_conf": top2[:, 0].cpu().numpy()}
    # numpy
    m = a - a.max(axis=1, keepdims=True)
    e = np.exp(m)
    p = e / e.sum(axis=1, keepdims=True)
    ent = -(p * np.log(np.clip(p, _EPS, 1.0))).sum(axis=1)
    if normalize and C > 1:
        ent = ent / np.log(C)
    order = np.sort(p, axis=1)[:, ::-1]
    margin = order[:, 0] - (order[:, 1] if C > 1 else 0.0)
    return {"entropy": ent, "margin": margin, "top1": p.argmax(axis=1), "top1_conf": order[:, 0]}


def ensemble_disagreement(path_probs, device=None) -> dict:
    """Ensemble disagreement across K auto-label paths over the same detections. path_probs is (K, N, C) of
    per-path class probabilities for N overlapping detections. Returns per-detection {mi (N,) BALD mutual
    information = H(mean) - E[H], pred_entropy (N,), exp_entropy (N,), class_var (N,) variance of the top mean
    class probability across paths}. High MI = the paths disagree in an informative way; those detections are
    the ones to route to a human."""
    a = np.asarray(path_probs, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("path_probs must be (K, N, C)")
    K, N, C = a.shape

    def _ent(p):
        return -(p * np.log(np.clip(p, _EPS, 1.0))).sum(axis=-1)

    if _HAS_TORCH and device != "cpu" and (device == "cuda" or torch.cuda.is_available()):
        t = torch.as_tensor(a, device="cuda")
        mean = t.mean(dim=0)                                          # (N, C)
        pred_ent = -(mean * torch.log(mean.clamp_min(_EPS))).sum(dim=1)
        exp_ent = -(t * torch.log(t.clamp_min(_EPS))).sum(dim=2).mean(dim=0)
        mi = (pred_ent - exp_ent).clamp_min(0.0)
        top_class = mean.argmax(dim=1)
        top_prob = t.gather(2, top_class.view(1, N, 1).expand(K, N, 1)).squeeze(2)  # (K, N)
        class_var = top_prob.var(dim=0, unbiased=False)
        return {"mi": mi.cpu().numpy(), "pred_entropy": pred_ent.cpu().numpy(),
                "exp_entropy": exp_ent.cpu().numpy(), "class_var": class_var.cpu().numpy()}
    mean = a.mean(axis=0)                                             # (N, C)
    pred_ent = _ent(mean)
    exp_ent = _ent(a).mean(axis=0)
    mi = np.maximum(pred_ent - exp_ent, 0.0)
    top_class = mean.argmax(axis=1)
    top_prob = np.take_along_axis(a, top_class[None, :, None], axis=2)[:, :, 0]      # (K, N)
    class_var = top_prob.var(axis=0)
    return {"mi": mi, "pred_entropy": pred_ent, "exp_entropy": exp_ent, "class_var": class_var}


def flicker_scores(boxes, confs=None, valid=None, flicker_thr=0.15, conf_thr=0.6, device=None) -> dict:
    """Temporal flicker (jitter) per track. boxes (M, T, 4) xyxy for M tracks over T frames; valid (M, T) marks
    frames where the track has a box (default all). Returns per-track {flicker (M,) scale-normalized centroid +
    size jitter across consecutive frames, mean_conf (M,), suspect (M,) bool = high flicker AND high confidence,
    the auto-label failure signature}. A stable object has near-zero flicker; a box that jumps or breathes
    frame-to-frame while the model stays confident is flagged."""
    b = np.asarray(boxes, dtype=np.float64)
    if b.ndim != 3 or b.shape[2] != 4:
        raise ValueError("boxes must be (M, T, 4)")
    M, T, _ = b.shape
    v = np.ones((M, T), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    c = np.zeros((M, T)) if confs is None else np.asarray(confs, dtype=np.float64)

    cx = (b[:, :, 0] + b[:, :, 2]) / 2.0
    cy = (b[:, :, 1] + b[:, :, 3]) / 2.0
    w = np.clip(b[:, :, 2] - b[:, :, 0], 0, None)
    h = np.clip(b[:, :, 3] - b[:, :, 1], 0, None)
    size = np.sqrt(np.clip(w * h, _EPS, None))                        # (M, T) scale for normalization

    pair = v[:, 1:] & v[:, :-1]                                       # consecutive-valid pairs
    dcx = np.abs(np.diff(cx, axis=1))
    dcy = np.abs(np.diff(cy, axis=1))
    dw = np.abs(np.diff(w, axis=1))
    dh = np.abs(np.diff(h, axis=1))
    scale = (size[:, 1:] + size[:, :-1]) / 2.0
    jit = (dcx + dcy + dw + dh) / np.clip(scale, _EPS, None)          # per-pair normalized jitter (M, T-1)

    npair = pair.sum(axis=1)
    flicker = np.divide((jit * pair).sum(axis=1), npair, out=np.zeros(M), where=npair > 0)
    conf_n = v.sum(axis=1)
    mean_conf = np.divide((c * v).sum(axis=1), conf_n, out=np.zeros(M), where=conf_n > 0)
    suspect = (flicker >= flicker_thr) & (mean_conf >= conf_thr)
    return {"flicker": flicker, "mean_conf": mean_conf, "suspect": suspect}
