"""SIEVYX unsupervised long-tail discovery (M12). The Indian long tail is enormous and mostly unlabeled; this
auto-clusters the embedding space and surfaces the rare, isolated groups for a human to name, so the scenario
ontology grows from data rather than from a fixed list. HDBSCAN is used when installed; a deterministic
threshold-connected-components fallback runs everywhere so the capability is never faked.

Reuses the existing SIEVYX DINO embeddings; pure over an embedding matrix, so it is testable."""

from __future__ import annotations

import importlib.util

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _connected_components(cos: np.ndarray, thr: float) -> list[list[int]]:
    n = cos.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if cos[i, j] >= thr:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def discover_rare_clusters(embeddings, ids: list[str], min_size: int = 2, sim_thr: float = 0.6) -> list[dict]:
    """Cluster the embeddings and rank clusters by rarity (small and isolated from the global centroid = rare).
    Returns clusters with their member ids, size, and rarity, rarest first."""
    X = _normalize(np.asarray(embeddings, dtype=np.float64))
    n = X.shape[0]
    if n < min_size:
        return []

    if importlib.util.find_spec("hdbscan") is not None:
        import hdbscan
        labels = hdbscan.HDBSCAN(min_cluster_size=max(2, min_size)).fit_predict(X)
        clusters = [[i for i in range(n) if labels[i] == c] for c in sorted(set(labels)) if c != -1]
        method = "hdbscan"
    else:
        clusters = _connected_components(X @ X.T, sim_thr)
        method = "threshold_cc"

    centroid = _normalize(X.mean(0, keepdims=True))[0]
    out = []
    for members in clusters:
        if len(members) < min_size:
            continue
        c = _normalize(X[members].mean(0, keepdims=True))[0]
        isolation = 1.0 - float(c @ centroid)                 # far from the global centroid = rare
        size_rarity = 1.0 / len(members)
        rarity = round(0.6 * isolation + 0.4 * size_rarity, 4)
        out.append({"member_ids": [ids[i] for i in members], "size": len(members),
                    "rarity": rarity, "method": method})
    return sorted(out, key=lambda x: x["rarity"], reverse=True)
