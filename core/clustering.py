"""Cosine clustering primitives, in one place.

Three modules cluster embeddings here and each grew its own arithmetic: `curation/projection.py` for the 2D
map, `agent/ontology_steward.py` for promoting a fallback cluster to a class, and `sievyx/discovery.py` for
finding rare ones. Three implementations of clustering is a fair thing to flag, and the obvious response,
one function they all call, would be the wrong fix. They ask genuinely different questions: what shape does
this corpus have, is this bag of crops one thing worth naming, and which groups are unusual. Forcing those
into one signature produces a function with three modes and four flags that nobody can change safely.

So what is shared here is the mechanism, not the policy. Two ways to group vectors by cosine similarity, one
availability check for HDBSCAN, and each caller keeps its own thresholds, scoring and interpretation.

The two linkages are not interchangeable and the choice matters:

**Greedy centroid** assigns each vector to the nearest cluster centroid above a threshold, or opens a new
one. Every member stays within the threshold of the running centroid, so a cluster is tight and coherent,
which is what the ontology steward needs before proposing "these 400 crops are one new class". It is
order-dependent, and that is the price.

**Connected components** joins any two vectors above the threshold and takes the transitive closure. Order
independent, and it will happily chain A to B to C where A and C are nothing alike, which is right for
discovery (a rare group should not be split because its members form a chain) and wrong for promotion.
"""

from __future__ import annotations

import importlib.util

import numpy as np


def normed(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n


def normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-9)


def greedy_cosine(rows: list[tuple[str, object]], sim_thresh: float) -> list[dict]:
    """Assign each vector to the nearest centroid above `sim_thresh`, else open a cluster.

    O(n*k) and fine for a bounded sample. Returns [{"vec": centroid, "members": [id, ...]}, ...].

    Order-dependent by construction: the same vectors in a different order can produce different clusters,
    because the first member of a cluster anchors it. That is acceptable where the caller controls the
    ordering and wants tight clusters, and it is why the ontology steward orders its sample deterministically
    rather than taking whatever the database returned.
    """
    clusters: list[dict] = []
    for oid, vec in rows:
        v = normed(vec)
        best_s, best_i = sim_thresh, -1
        for i, c in enumerate(clusters):
            s = float(c["vec"] @ v)
            if s > best_s:
                best_s, best_i = s, i
        if best_i >= 0:
            c = clusters[best_i]
            n = len(c["members"])
            c["vec"] = normed(c["vec"] * n + v)
            c["members"].append(str(oid))
        else:
            clusters.append({"vec": v, "members": [str(oid)]})
    return clusters


def connected_components(sim: np.ndarray, thr: float) -> list[list[int]]:
    """Transitive closure of "similar above thr". Returns member index lists.

    Order independent, and chains: A joins B joins C even when A and C are dissimilar. Right for discovering
    a rare group that happens to be strung out in embedding space, wrong for deciding that a set of crops is
    one class.
    """
    n = sim.shape[0]
    seen = [False] * n
    out: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.nonzero(sim[k] >= thr)[0]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        out.append(sorted(comp))
    return out


def hdbscan_available() -> bool:
    """Whether density clustering can be used at all.

    Checked rather than caught, so a caller can choose its fallback deliberately and record which method
    produced a result. A silent fallback would mean two runs of the same analysis clustering differently
    with nothing in the output saying so.
    """
    return importlib.util.find_spec("hdbscan") is not None


def hdbscan_labels(X: np.ndarray, min_cluster_size: int = 15) -> np.ndarray | None:
    """Density labels over row vectors, -1 for noise. None when hdbscan is absent or fails.

    None rather than a fallback: the caller knows which alternative suits its question, and picking one here
    would hide the substitution from whatever reports the result.
    """
    if not hdbscan_available():
        return None
    try:
        import hdbscan

        return np.asarray(hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X), dtype=int)
    except Exception:  # noqa: BLE001
        return None
