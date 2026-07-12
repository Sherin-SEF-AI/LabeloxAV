"""Integration: the dedup pre-filter's batched pHash-Hamming + cosine matrices must select the exact same
duplicate pairs as the original O(n^2) Python double loop (imagehash pHash Hamming AND DINOv3 cosine)."""

import numpy as np

from core.accel.curation import hamming_matrix


def test_matrix_dedup_matches_loop():
    rng = np.random.default_rng(0)
    n = 60
    # synthetic 64-bit pHashes (8x8 bool) and unit DINOv3 vectors
    hashes = rng.random((n, 8, 8)) > 0.5
    vecs = rng.normal(size=(n, 64)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    # seed a few near-duplicate pairs (close hash + high cosine)
    for a, b in [(1, 2), (10, 11), (30, 45)]:
        hashes[b] = hashes[a].copy()
        hashes[b].flat[0] = not hashes[b].flat[0]        # 1-bit difference
        vecs[b] = vecs[a]
    ph_thresh, cos_thresh = 6, 0.9

    # reference: the original double loop
    ref = set()
    for i in range(n):
        for j in range(i + 1, n):
            ham = int(np.count_nonzero(hashes[i] != hashes[j]))
            if ham <= ph_thresh and float(vecs[i] @ vecs[j]) >= cos_thresh:
                ref.add((i, j))

    # new: batched matrices (the wired path)
    packed = np.stack([np.packbits(h.ravel()) for h in hashes])
    Hm = hamming_matrix(packed)
    S = vecs @ vecs.T
    dup = (Hm <= ph_thresh) & (S >= cos_thresh)
    got = set(zip(*(x.tolist() for x in np.where(np.triu(dup, 1)))))

    assert got == ref
    assert (1, 2) in got and (10, 11) in got and (30, 45) in got   # the seeded near-dups are found
