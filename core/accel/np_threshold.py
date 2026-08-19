"""Neyman-Pearson operating points: the smallest score that keeps the false-accept rate under a bound.

The auto-accept thresholds this engine ships (0.95 benign, 0.99 safety) are hand-picked constants. They
are described in the gate as precision floors, and they are not: a threshold is only a precision floor if
somebody measured the precision at that threshold, and nobody did. Two classes with the same nominal 0.95
can sit at wildly different real precisions, and a recalibration moves both without moving the constant.

This fits the threshold from measured outcomes instead. Given (score, matched) pairs for one class, the
false-accept rate at a threshold t is

    FAR(t) = P(not matched | score >= t)

which is one minus precision among the things the gate would accept. That is the quantity the gate exists
to bound: the harm from auto-accept is a wrong label entering the corpus unseen, and it is proportional to
how many accepted boxes are wrong, not to how many rejected boxes were right. The fit returns

    t* = min{ t : FAR(t) <= alpha }

so that, in the Neyman-Pearson shape, the accepted set is as large as the constraint allows.

Three details in that definition are load-bearing.

THE CONSTRAINT IS EVALUATED AT THE CHOSEN THRESHOLD AND NOWHERE ELSE. FAR(t) is not monotone in t on a
finite sample, which invites a rule that also requires the bound to hold at every shallower cut. That rule
is wrong here, and expensively so. A wrong detection with a very high score is inside the accepted set at
every threshold below it, so it is already counted in FAR(t); refusing every cut beneath it discards
acceptance the data supports without removing a single error. On a twenty-detection fixture the two rules
give 3 accepted against 10, both at a measured 0.20. Non-monotonicity is a question about how well the
threshold is located, and the bootstrap below is what answers it.

DEEPEST, NOT SHALLOWEST. Among the cuts that satisfy the bound, the deepest accepts the most, which is
what Neyman-Pearson asks for. It is also the better-estimated end: a shallow cut's FAR is a ratio over a
handful of detections and dips under alpha easily by luck, while a deep one averages over most of the
sample.

THE CANDIDATES ARE THE OBSERVED SCORES. Interpolating between them would report a threshold no detection
ever had, and the sample cannot distinguish it from its neighbours.

SUPPORT IS REFUSED RATHER THAN EXTRAPOLATED. Below `min_support` pairs, or with no positives, or when no
threshold in the range achieves the bound, the fit returns measured False with a reason and no threshold.
A class with four labelled examples has not earned an operating point, and returning the highest observed
score would look like a strict threshold while resting on nothing.

The confidence interval is a seeded bootstrap over the pairs, so the same data always yields the same
interval and a refit that moves is data moving rather than the estimator. A wide interval is the honest
signal that the threshold is not yet located, and the caller is expected to refuse to activate on it
rather than to round it off.

NumPy is the oracle; the torch path mirrors the bootstrap (the resampling is the expensive part) and must
agree to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

# Enough labelled outcomes for a tail probability of a few percent to mean anything. Below this the
# estimate is dominated by whether one or two particular boxes happened to be in the sample.
MIN_SUPPORT = 50
N_BOOTSTRAP = 1000
_SEED = 20260819


@dataclass(frozen=True)
class ThresholdEstimate:
    """One fitted operating point, with everything needed to decide whether to trust it.

    `threshold` is the score at which the gate should accept. `far_at` is the measured false-accept rate
    there, `accept_rate` the share of detections it would accept, and `n_accept` how many that was, so a
    threshold that clears the bound by accepting almost nothing is visible as such rather than looking
    like a good result.

    `lo`/`hi` bound the threshold itself over a seeded bootstrap. They are not a bound on FAR: the question
    they answer is "would another sample of this size have put the threshold somewhere else", and a wide
    interval means the operating point is not yet located. They are None when no resample admitted any
    threshold, which is the strongest form of the same statement; `n_boot_fit` says how many did.

    A caller deciding whether to switch a threshold on must look at the interval and not only at
    `measured`. A fit can be real and still be too loosely located to act on, and this type reports that
    rather than deciding it.

    `measured` is False, with a reason, when the pairs cannot support a fit at all.
    """

    measured: bool
    threshold: float | None
    lo: float | None
    hi: float | None
    far_at: float | None
    accept_rate: float | None
    n_accept: int
    n_pairs: int
    n_positive: int
    alpha: float
    # How many of the bootstrap resamples admitted any threshold at all. A fit whose own data, resampled,
    # usually yields nothing is not located, however clean the point estimate looks.
    n_boot_fit: int = 0
    reason: str | None = None


def _unfittable(n_pairs: int, n_positive: int, alpha: float, reason: str,
                n_boot_fit: int = 0) -> ThresholdEstimate:
    return ThresholdEstimate(
        measured=False, threshold=None, lo=None, hi=None, far_at=None, accept_rate=None,
        n_accept=0, n_pairs=n_pairs, n_positive=n_positive, alpha=alpha, n_boot_fit=n_boot_fit,
        reason=reason)


def _fit_np(scores: npt.NDArray[np.float64], matched: npt.NDArray[np.bool_],
            alpha: float) -> tuple[float, float, int] | None:
    """The threshold, its FAR, and the accepted count. None when no threshold achieves the bound.

    Sorting descending and accumulating gives, at each candidate, the counts for everything at or above it
    in one pass, so `far[i]` is exactly the false-accept rate the gate would have thresholded there.
    """
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    m = matched[order].astype(np.float64)

    k = np.arange(1, s.size + 1, dtype=np.float64)          # how many accepted at each cut
    far = 1.0 - np.cumsum(m) / k                            # FAR of the accepted set at each cut

    # Ties: every detection sharing the boundary score is accepted or rejected together, so a candidate is
    # only valid at the last index of its run of equal scores. A cut inside a run would describe an accept
    # rule the gate cannot implement, since it thresholds on the score alone.
    last_of_run = np.empty(s.size, dtype=bool)
    last_of_run[:-1] = s[:-1] > s[1:]
    last_of_run[-1] = True

    ok = (far <= alpha) & last_of_run
    if not ok.any():
        return None
    # The deepest admissible cut, which is the smallest threshold and the largest accepted set.
    i = int(np.flatnonzero(ok)[-1])
    return float(s[i]), float(far[i]), i + 1


def _bootstrap_np(scores: npt.NDArray[np.float64], matched: npt.NDArray[np.bool_], alpha: float,
                  n_boot: int, seed: int) -> npt.NDArray[np.float64]:
    """Refit the threshold on `n_boot` resamples. Resamples that cannot be fit contribute nothing.

    Dropping them rather than substituting a value is deliberate: a resample where no threshold achieves
    the bound is evidence the fit is fragile, and imputing the maximum score would hide that inside a
    tighter-looking interval. The count that survived is what the caller checks.
    """
    rng = np.random.default_rng(seed)
    n = scores.size
    out = np.empty(n_boot, dtype=np.float64)
    kept = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        got = _fit_np(scores[idx], matched[idx], alpha)
        if got is not None:
            out[kept] = got[0]
            kept += 1
    return out[:kept]


def _bootstrap_torch(scores: npt.NDArray[np.float64], matched: npt.NDArray[np.bool_], alpha: float,
                     n_boot: int, seed: int, device: str
                     ) -> npt.NDArray[np.float64]:  # pragma: no cover - GPU path
    """The same resampling, all replicates at once. Mirrors the NumPy path exactly, including the seed.

    torch.Generator with a manual seed on the CPU produces the same integers as NumPy's default_rng only by
    coincidence of neither being guaranteed, so the indices are drawn on the CPU with the NumPy generator
    and moved across. What runs on the device is the sort and the cumulative sums, which is where the work
    is; the parity test then compares like with like rather than two different random samples.
    """
    rng = np.random.default_rng(seed)
    n = scores.size
    idx = rng.integers(0, n, (n_boot, n))
    dev = torch.device(device)
    s = torch.as_tensor(scores, device=dev, dtype=torch.float64)[torch.as_tensor(idx, device=dev)]
    m = torch.as_tensor(matched, device=dev, dtype=torch.float64)[torch.as_tensor(idx, device=dev)]

    s, order = torch.sort(s, dim=1, descending=True, stable=True)
    m = torch.gather(m, 1, order)
    k = torch.arange(1, n + 1, device=dev, dtype=torch.float64).unsqueeze(0)
    far = 1.0 - torch.cumsum(m, dim=1) / k

    last_of_run = torch.ones_like(s, dtype=torch.bool)
    last_of_run[:, :-1] = s[:, :-1] > s[:, 1:]
    ok = (far <= alpha) & last_of_run
    # The last valid index per row, or -1 where the row has none.
    pos = torch.where(ok, torch.arange(n, device=dev).unsqueeze(0), torch.full_like(k, -1.0).long())
    i = pos.max(dim=1).values
    keep = i >= 0
    return s[keep, i[keep]].cpu().numpy()


def np_threshold(scores: npt.ArrayLike, matched: npt.ArrayLike, *, alpha: float,
                 min_support: int = MIN_SUPPORT, n_boot: int = N_BOOTSTRAP, seed: int = _SEED,
                 device: str | None = None) -> ThresholdEstimate:
    """Fit the smallest score whose whole tail keeps the false-accept rate at or under `alpha`.

    `scores` and `matched` are paired per detection: the score the gate would threshold on, and whether
    that detection turned out to be right. `alpha` is the tolerated false-accept rate among accepted.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = np.asarray(matched).reshape(-1).astype(bool)
    if s.size != m.size:
        raise ValueError(f"{s.size} scores against {m.size} outcomes")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n_pairs, n_pos = int(s.size), int(m.sum())
    if n_pairs < min_support:
        return _unfittable(n_pairs, n_pos, alpha,
                           f"only {n_pairs} labelled outcomes, below the {min_support} needed to locate a "
                           f"{alpha:.0%} tail")
    if n_pos == 0:
        return _unfittable(n_pairs, n_pos, alpha,
                           "no detection in this sample was correct, so no threshold accepts anything")

    fitted = _fit_np(s, m, alpha)
    if fitted is None:
        # Every cut, including accepting only the single highest-scoring detection, is above the bound. The
        # honest answer is that this class cannot be auto-accepted at this alpha, not a threshold of 1.0.
        return _unfittable(n_pairs, n_pos, alpha,
                           f"no threshold holds the false-accept rate at or under {alpha:.3f}; the best "
                           f"achievable is {float(1.0 - m[np.argmax(s)]):.3f} at the top score alone")
    t, far_at, n_accept = fitted

    if _HAS_TORCH and device is not None and str(device) != "cpu" and torch.cuda.is_available():
        boots = _bootstrap_torch(s, m, alpha, n_boot, seed, str(device))  # pragma: no cover - GPU path
    else:
        boots = _bootstrap_np(s, m, alpha, n_boot, seed)

    # No refusal here on how many resamples survived. Whether a threshold is located well enough to switch
    # on is an activation decision, and this engine already puts those in the service layer rather than the
    # primitive (services/autolabel/gold_calibrate.py fits, reports `trustworthy`, and does not activate).
    # Every construction that stresses this lands near half, so any constant chosen here would fire on
    # seed noise for exactly the data it was meant to catch. The estimate carries what the decision needs:
    # the interval, and how many resamples it came from.
    lo, hi = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
              if boots.size else (None, None))

    return ThresholdEstimate(
        measured=True,
        threshold=round(t, 6),
        lo=round(lo, 6) if lo is not None else None,
        hi=round(hi, 6) if hi is not None else None,
        far_at=round(far_at, 6),
        accept_rate=round(n_accept / n_pairs, 6),
        n_accept=n_accept, n_pairs=n_pairs, n_positive=n_pos, alpha=alpha,
        n_boot_fit=int(boots.size))


def fit_per_class(pairs: dict[int, tuple[npt.ArrayLike, npt.ArrayLike]], *,
                  alpha_for: dict[int, float], default_alpha: float,
                  min_support: int = MIN_SUPPORT, n_boot: int = N_BOOTSTRAP,
                  seed: int = _SEED, device: str | None = None) -> dict[str, Any]:
    """Fit one operating point per class. Returns {"per_class", "n_fitted", "n_refused"}.

    Each class carries its own alpha, because the cost of a wrong auto-accept is not the same across the
    ontology: a mislabelled pedestrian and a mislabelled bollard are not equally expensive, and one bound
    for both means the bound is wrong for at least one of them. Classes with no entry take `default_alpha`.

    A class that cannot be fit appears in the output with measured False and its reason, never omitted:
    the caller has to be able to tell a class that earned no threshold from a class nobody looked at.
    """
    out: dict[int, ThresholdEstimate] = {}
    for cid, (s, m) in sorted(pairs.items()):
        out[cid] = np_threshold(s, m, alpha=alpha_for.get(cid, default_alpha),
                                min_support=min_support, n_boot=n_boot, seed=seed, device=device)
    return {"per_class": out,
            "n_fitted": sum(1 for e in out.values() if e.measured),
            "n_refused": sum(1 for e in out.values() if not e.measured)}


__all__ = ["ThresholdEstimate", "np_threshold", "fit_per_class", "MIN_SUPPORT", "N_BOOTSTRAP"]
