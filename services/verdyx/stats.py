"""VERDYX statistical rigor (M15). A challenger that scores 0.61 against a champion's 0.59 has not necessarily
improved; on a small gold set that gap can be noise. This adds the two statistics a promotion decision needs: a
bootstrap confidence interval on a per-object metric (so a point estimate carries its uncertainty), and a
paired significance test between champion and challenger on the same objects (so a promotion is only claimed
when the improvement is unlikely to be chance). Deterministic given a seed, so the intervals and p-values are
testable and byte-reproducible."""

from __future__ import annotations

import random


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 12345) -> dict:
    """Percentile bootstrap CI for the mean of per-object scores (e.g. per-object hit=1/miss=0 gives a recall
    CI). Deterministic under ``seed``. Returns the point mean and the [lo, hi] interval at 1-alpha."""
    n = len(values)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"mean": round(sum(values) / n, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n}


def paired_significance(champion: list[float], challenger: list[float], n_perm: int = 2000,
                        seed: int = 12345) -> dict:
    """Paired permutation test on the same objects: is the challenger's mean strictly better than the
    champion's, or is the gap chance? champion[i] and challenger[i] are the two models' per-object scores on
    object i. Returns the observed delta, a two-sided p-value, and significant at p<0.05. Deterministic under
    ``seed``."""
    if len(champion) != len(challenger):
        raise ValueError("paired significance needs equal-length per-object score vectors")
    n = len(champion)
    if n == 0:
        return {"delta": None, "p_value": None, "significant": False, "n": 0}
    diffs = [challenger[i] - champion[i] for i in range(n)]
    observed = sum(diffs) / n
    rng = random.Random(seed)
    at_least = 0
    for _ in range(n_perm):
        # under H0 the sign of each paired diff is exchangeable
        perm = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(perm) >= abs(observed):
            at_least += 1
    p = (at_least + 1) / (n_perm + 1)
    return {"delta": round(observed, 4), "p_value": round(p, 4), "significant": p < 0.05, "n": n}
