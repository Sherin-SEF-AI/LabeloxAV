"""SIEVYX unsupervised long-tail discovery (M12). The Indian long tail is enormous and mostly unlabeled; this
auto-clusters the embedding space and surfaces the rare, isolated groups for a human to name, so the scenario
ontology grows from data rather than from a fixed list. HDBSCAN is used when installed; a deterministic
threshold-connected-components fallback runs everywhere so the capability is never faked.

Reuses the existing SIEVYX DINO embeddings; pure over an embedding matrix, so it is testable."""

from __future__ import annotations

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _connected_components(cos: np.ndarray, thr: float) -> list[list[int]]:
    """Transitive closure of "similar above thr", from core.clustering.

    Components rather than greedy centroids on purpose: a rare group strung out in embedding space is still
    one group, and splitting it because its members form a chain is exactly the failure this detector exists
    to avoid. That chaining would be wrong for the ontology steward, which is why the two linkages stay
    distinct rather than being merged into one clustering function.
    """
    from core.clustering import connected_components

    return connected_components(cos, thr)


def discover_rare_clusters(embeddings, ids: list[str], min_size: int = 2, sim_thr: float = 0.6) -> list[dict]:
    """Cluster the embeddings and rank clusters by rarity (small and isolated from the global centroid = rare).
    Returns clusters with their member ids, size, and rarity, rarest first."""
    X = _normalize(np.asarray(embeddings, dtype=np.float64))
    n = X.shape[0]
    if n < min_size:
        return []

    from core.clustering import hdbscan_labels

    labels = hdbscan_labels(X, min_cluster_size=max(2, min_size))
    if labels is not None:
        clusters = [[i for i in range(n) if labels[i] == c] for c in sorted(set(labels)) if c != -1]
        method = "hdbscan"
    else:
        # Recorded in the output, because the two methods do not agree and a run that silently substituted
        # one for the other would be uncomparable with the run before it.
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
