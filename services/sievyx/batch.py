"""SIEVYX budget-aware batch selection (M12). Uncertainty-plus-rarity ranking tends to pick a redundant batch
of near-identical uncertain samples. This selects a diverse, informative batch under a label budget using a
core-set (greedy farthest-point) cover over the embeddings, weighted by expected model change, so the batch
spans the space instead of clustering on one hard case. Pure over an embedding matrix, so it is testable and
its coverage is comparable to the naive top-k selector."""

from __future__ import annotations

import numpy as np


def expected_model_change(uncertainty: float) -> float:
    """A cheap proxy for expected model change: a more uncertain sample yields a larger gradient step. Kept
    monotonic and bounded so it combines cleanly with core-set coverage."""
    u = max(0.0, min(1.0, float(uncertainty)))
    return u


def select_coreset(embeddings, ids: list[str], budget: int) -> list[str]:
    """Greedy k-center (farthest-point) selection: repeatedly add the point farthest from the already-selected
    set, so the chosen budget covers the embedding space rather than crowding one region."""
    X = np.asarray(embeddings, dtype=np.float64)
    n = X.shape[0]
    budget = min(max(1, budget), n)
    # start from the point farthest from the centroid (a boundary case), deterministic
    centroid = X.mean(0)
    first = int(np.argmax(np.linalg.norm(X - centroid, axis=1)))
    selected = [first]
    dist = np.linalg.norm(X - X[first], axis=1)
    while len(selected) < budget:
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    return [ids[i] for i in selected]


def select_batch(candidates: list[dict], budget: int, w_change: float = 0.5) -> list[str]:
    """Select a batch from scored candidates (each {id, embedding, uncertainty}) balancing core-set coverage
    with expected model change. Coverage picks a diverse spanning set; the change weight breaks ties toward the
    more informative samples by pre-filtering to the top change-scored pool before covering it."""
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda c: expected_model_change(c.get("uncertainty", 0.0)), reverse=True)
    pool = ranked[: max(budget * 4, budget)]   # cover a diverse subset of the most informative pool
    emb = [c["embedding"] for c in pool]
    ids = [c["id"] for c in pool]
    return select_coreset(emb, ids, budget)
