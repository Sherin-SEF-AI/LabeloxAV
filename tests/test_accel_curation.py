"""Curation/dedup kernels (SIEVYX) verification: cosine matrix vs sklearn, the pHash+Hamming near-dup gate
(duplicates hash close, different frames far), the Hamming matrix vs a brute reference, and k-center greedy
matching services.sievyx.batch.select_coreset exactly. A session-scale timing is printed."""

import time

import numpy as np
import pytest

from core.accel.curation import (
    cosine_sim_matrix,
    gpu_available,
    hamming_matrix,
    kcenter_greedy,
    nearest_neighbor_sim,
    phash_batch,
)
from services.sievyx.batch import select_coreset


def test_cosine_matches_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 128)).astype(np.float32)
    S = cosine_sim_matrix(X, device="cpu")
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert np.allclose(S, Xn @ Xn.T, atol=1e-5)
    assert np.allclose(np.diag(S), 1.0, atol=1e-5)         # self-similarity 1
    nn = nearest_neighbor_sim(X, device="cpu")
    assert nn.shape == (64,) and (nn <= 1.0 + 1e-5).all()


def test_phash_hamming_near_dup():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    dup = base.copy()
    dup[0, 0] = 255 - dup[0, 0]                             # a 1-pixel change: still a near-duplicate
    diff = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    hashes = phash_batch(np.stack([base, dup, diff]), device="cpu")
    assert hashes.shape == (3, 8)
    Hm = hamming_matrix(hashes, device="cpu")
    # brute reference
    def _ham(a, b):
        return int(np.unpackbits(a ^ b).sum())
    for i in range(3):
        for j in range(3):
            assert Hm[i, j] == _ham(hashes[i], hashes[j])
    assert Hm[0, 1] <= 4                                    # near-duplicate: small Hamming
    assert Hm[0, 2] > Hm[0, 1]                              # unrelated frame: larger Hamming


def test_kcenter_matches_select_coreset():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 32))
    ids = [str(i) for i in range(200)]
    ref = select_coreset(X, ids, 25)
    got = kcenter_greedy(X, 25, device="cpu")
    assert [str(i) for i in got] == ref                    # identical greedy selection

    if gpu_available():
        got_gpu = kcenter_greedy(X, 25, device="cuda")
        assert [str(i) for i in got_gpu] == ref            # GPU distance update, same selection


def test_measurable():
    if not gpu_available():
        return
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 768)).astype(np.float32)
    assert np.allclose(cosine_sim_matrix(X, device="cuda"), cosine_sim_matrix(X, device="cpu"), atol=1e-3)
    for _ in range(3):
        cosine_sim_matrix(X, device="cuda")
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        cosine_sim_matrix(X, device="cuda")
    gpu_ms = (time.perf_counter() - t0) / reps * 1000
    t0 = time.perf_counter()
    for _ in range(reps):
        cosine_sim_matrix(X, device="cpu")
    cpu_ms = (time.perf_counter() - t0) / reps * 1000
    print(f"\ncosine sim {X.shape[0]}x{X.shape[0]} (D={X.shape[1]}): GPU {gpu_ms:.2f} ms | NumPy {cpu_ms:.2f} ms "
          f"| speedup {cpu_ms / gpu_ms:.1f}x")
