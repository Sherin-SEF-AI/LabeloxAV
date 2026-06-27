"""Gate B, part (b): Safe-mIoU. A confusion-weighted score that penalizes UNSAFE class confusions
hardest, derived from the ontology l0/l1 hierarchy. Confusing a pedestrian for a pole (VRU vs fixed
infra) costs maximally; confusing a sedan for a hatchback (both four_wheeler) costs little. This
mirrors IDD's tree-distance Safe-mIoU idea.
"""

from __future__ import annotations

import numpy as np

from services.autolabel.ontology import Ontology

_VRU_ANIMAL = {"vru", "animal"}


def affinity_cost(onto: Ontology, a: int, b: int) -> float:
    """Safety cost of confusing class a for class b, in [0, 1]. 0 = same class; 1 = maximally unsafe."""
    if a == b:
        return 0.0
    ca, cb = onto.by_id(a), onto.by_id(b)
    a_vru = ca.l1 in _VRU_ANIMAL
    b_vru = cb.l1 in _VRU_ANIMAL
    if a_vru != b_vru:
        cost = 1.0                      # a vulnerable road user confused with anything non-VRU
    elif ca.l1 == cb.l1:
        cost = 0.2                      # same superclass (sedan vs hatchback)
    elif ca.l0 == cb.l0:
        cost = 0.5                      # same top level, different superclass
    else:
        cost = 0.8                      # different top level
    if ca.india or cb.india or ca.l1 == "fallback" or cb.l1 == "fallback":
        cost = min(1.0, cost + 0.1)
    return cost


def safe_miou(matrix, class_ids: list[int], onto: Ontology, safety_weight: float = 2.0) -> float | None:
    """Score in [0, 1] from a class confusion matrix (rows=true, cols=pred), aligned to class_ids.
    Off-diagonal mass is weighted by affinity_cost so unsafe confusions sink the score fastest.
    Returns None when there is no class-vs-class mass (undefined, not 'maximally unsafe')."""
    m = np.asarray(matrix, dtype=float)
    total = float(m.sum())
    if total <= 0:
        return None
    n = len(class_ids)
    werr = 0.0
    for i in range(n):
        for j in range(n):
            if i == j or m[i, j] == 0:
                continue
            werr += m[i, j] * affinity_cost(onto, class_ids[i], class_ids[j])
    return float(max(0.0, min(1.0, 1.0 - safety_weight * (werr / total))))
