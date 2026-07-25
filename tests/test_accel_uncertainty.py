"""Active-learning uncertainty kernels verification: entropy/margin against scipy, ensemble BALD MI properties
(zero when paths agree, positive when they disagree), the flicker score flagging a jittery high-confidence
track, and GPU == CPU. A session-scale timing is printed."""

import time

import numpy as np
import pytest

from core.accel.uncertainty import ensemble_disagreement, entropy_margin, flicker_scores, gpu_available


def test_entropy_margin_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(500, 12))
    out = entropy_margin(logits, normalize=False, device="cpu")
    # reference entropy via scipy on the softmax
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ref_ent = sp.entropy(p, axis=1)
    assert np.allclose(out["entropy"], ref_ent, atol=1e-5)
    # margin = top1 - top2 probability
    srt = np.sort(p, axis=1)[:, ::-1]
    assert np.allclose(out["margin"], srt[:, 0] - srt[:, 1], atol=1e-6)
    assert np.array_equal(out["top1"], p.argmax(1))
    # a near-uniform row has entropy ~1 when normalized; a peaked row ~0
    peaked = entropy_margin(np.array([[10.0, 0, 0, 0]]), normalize=True, device="cpu")
    uniform = entropy_margin(np.array([[1.0, 1, 1, 1]]), normalize=True, device="cpu")
    assert peaked["entropy"][0] < 0.05 and uniform["entropy"][0] > 0.99


def test_ensemble_disagreement_bald():
    # three paths that AGREE (same confident prediction) -> MI ~ 0
    agree = np.tile(np.array([[[0.9, 0.05, 0.05]]]), (3, 1, 1))
    a = ensemble_disagreement(agree, device="cpu")
    assert a["mi"][0] < 1e-6 and a["class_var"][0] < 1e-6
    # three paths that DISAGREE confidently (each sure of a different class) -> high MI
    disagree = np.array([[[0.9, 0.05, 0.05]], [[0.05, 0.9, 0.05]], [[0.05, 0.05, 0.9]]])
    d = ensemble_disagreement(disagree, device="cpu")
    assert d["mi"][0] > 0.5 and d["class_var"][0] > 0.1
    assert d["mi"][0] > a["mi"][0]


def test_flicker_flags_jittery_confident_track():
    T = 12
    # track 0: stable box, high conf -> not suspect
    stable = np.tile(np.array([100.0, 100, 140, 180]), (T, 1))
    # track 1: box jumps around frame-to-frame, high conf -> suspect (the failure signature)
    rng = np.random.default_rng(3)
    jumpy = np.array([100.0, 100, 140, 180]) + rng.normal(0, 25, size=(T, 4))
    boxes = np.stack([stable, jumpy])
    confs = np.full((2, T), 0.9)
    out = flicker_scores(boxes, confs)
    assert out["flicker"][0] < 0.02 and not out["suspect"][0]
    assert out["flicker"][1] > out["flicker"][0] and out["suspect"][1]


def test_gpu_matches_cpu_and_measurable():
    if not gpu_available():
        return
    rng = np.random.default_rng(4)
    logits = rng.normal(0, 3, size=(50000, 40))
    cpu = entropy_margin(logits, device="cpu")
    gpu = entropy_margin(logits, device="cuda")
    assert np.allclose(cpu["entropy"], gpu["entropy"], atol=1e-4)
    assert np.allclose(cpu["margin"], gpu["margin"], atol=1e-4)

    for _ in range(3):
        entropy_margin(logits, device="cuda")
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        entropy_margin(logits, device="cuda")
    gpu_ms = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(n):
        entropy_margin(logits, device="cpu")
    cpu_ms = (time.perf_counter() - t0) / n * 1000
    print(f"\nentropy+margin {logits.shape[0]:,} detections x {logits.shape[1]} classes: "
          f"GPU {gpu_ms:.2f} ms | NumPy {cpu_ms:.2f} ms | speedup {cpu_ms / gpu_ms:.1f}x")
