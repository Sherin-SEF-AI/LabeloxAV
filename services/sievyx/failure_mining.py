"""SIEVYX failure-driven mining (M12). Instead of mining where the model is merely uncertain, mine exactly
where the deployed model FAILS: consume the VERDYX failure clusters (embeddings of the false negatives and
false positives) and the recall-recovery signals, and rank the unlabeled pool by similarity to those failure
modes, so the next label batch targets the model's actual weaknesses. Pure over embeddings, so it is testable.
"""

from __future__ import annotations

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def mine_from_failures(failure_centroids, pool_embeddings, pool_ids: list[str], k: int = 100) -> list[dict]:
    """Rank the unlabeled pool by its maximum similarity to any failure cluster centroid, returning the top-k
    most-like-a-known-failure samples with the failure mode each is closest to."""
    if not len(failure_centroids) or not len(pool_ids):
        return []
    F = _normalize(np.asarray(failure_centroids, dtype=np.float64))
    P = _normalize(np.asarray(pool_embeddings, dtype=np.float64))
    sim = P @ F.T                       # (pool, failure)
    best = sim.max(axis=1)
    which = sim.argmax(axis=1)
    order = np.argsort(-best)[:k]
    return [{"id": pool_ids[i], "similarity": round(float(best[i]), 4), "failure_mode": int(which[i])}
            for i in order]
