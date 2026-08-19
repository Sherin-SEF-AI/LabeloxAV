"""Capture-recapture population estimation (Tier 4 metric primitive). Every recall number this engine
produces is recall against objects somebody found, and the two observers are not equally likely to find
things: a human confirms a machine box far more readily than they draw a new one, so the gold denominator
is biased toward what the model already sees. The fixed prediction plane stopped predictions being
destroyed; it did nothing about the denominator.

Lincoln-Petersen treats the model and a blind human annotator as two independent observers of one
population. Objects found by both, by the model only, and by the human only determine, in closed form, how
many were found by neither, hence the true population and the true recall of each observer.

Chapman's bias-corrected form is the default because the per-stratum counts here are small (a stratified
200-frame audit can put a dozen objects in a cell) and the raw Lincoln-Petersen ratio n1*n2/m2 is badly
biased upward at small m2 and undefined at m2 = 0. Chapman (1951), variance from Seber (1970, 1982):

    N_hat   = (n1 + 1)(n2 + 1) / (m2 + 1) - 1
    var     = (n1 + 1)(n2 + 1)(n1 - m2)(n2 - m2) / [(m2 + 1)^2 (m2 + 2)]

INDEPENDENCE IS THE ASSUMPTION AND IT IS NOT FULLY MET. A small, occluded, badly lit object is harder for
both observers, so the two captures are positively correlated, m2 is larger than independence predicts, and
N_hat is therefore biased DOWN. Every number this module produces is a lower bound on the missed
population, and an upper bound on recall. It is reported that way and must never be presented as exact.

NumPy is the oracle; the torch path mirrors it for the stratified form (many strata at once) and must agree
to 1e-6. The scalar form is arithmetic on four numbers and never leaves NumPy.
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

# Two-sided normal quantile for a 95% interval. The normal approximation is what Seber gives for the
# Chapman variance; it is adequate once m2 is a handful and is stated rather than assumed.
_Z95 = 1.959963984540054


@dataclass(frozen=True)
class RecaptureEstimate:
    """One capture-recapture estimate, with the counts it was computed from.

    `population` is the estimated true number of objects present, `model_recall` and `human_recall` the
    fraction each observer found. `lo`/`hi` bound the population at 95%; `recall_lo`/`recall_hi` bound the
    model recall, and are the population interval inverted (recall falls as the population rises), so
    recall_lo pairs with hi and not with lo.

    `measured` is False when the estimator cannot say anything, which is the m2 = 0 case: with no object
    found by both observers there is no overlap to estimate from and the population is unbounded above.
    """

    measured: bool
    population: float | None
    lo: float | None
    hi: float | None
    model_recall: float | None
    human_recall: float | None
    recall_lo: float | None
    recall_hi: float | None
    n_both: int
    n_model_only: int
    n_human_only: int
    variance: float | None = None
    reason: str | None = None


def _unmeasured(both: int, model_only: int, human_only: int, reason: str) -> RecaptureEstimate:
    return RecaptureEstimate(
        measured=False, population=None, lo=None, hi=None, model_recall=None, human_recall=None,
        recall_lo=None, recall_hi=None, n_both=both, n_model_only=model_only, n_human_only=human_only,
        reason=reason)


def _chapman_np(n1: npt.NDArray[np.float64], n2: npt.NDArray[np.float64],
                m2: npt.NDArray[np.float64], chapman: bool
                ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Population estimate and its variance, elementwise over strata. Returns (n_hat, var)."""
    if chapman:
        n_hat = (n1 + 1.0) * (n2 + 1.0) / (m2 + 1.0) - 1.0
        var = ((n1 + 1.0) * (n2 + 1.0) * (n1 - m2) * (n2 - m2)) / (((m2 + 1.0) ** 2) * (m2 + 2.0))
    else:
        safe_m2 = np.where(m2 > 0, m2, 1.0)
        n_hat = np.where(m2 > 0, n1 * n2 / safe_m2, np.inf)
        # Seber's variance for the uncorrected ratio. Undefined at m2 = 0 along with the estimate itself.
        var = np.where(m2 > 0, (n1 * n1 * n2 * (n2 - m2)) / (safe_m2 ** 3), np.inf)
    return n_hat, var


def _chapman_torch(n1: npt.NDArray[np.float64], n2: npt.NDArray[np.float64],
                   m2: npt.NDArray[np.float64], chapman: bool, device: str
                   ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:  # pragma: no cover - GPU
    # Distinct names from the ndarray arguments: rebinding them to tensors reads as if one type flowed
    # through, and float64 throughout is what makes this comparable to the NumPy oracle at 1e-6.
    f64 = torch.float64
    t1 = torch.as_tensor(n1, device=device, dtype=f64)
    t2 = torch.as_tensor(n2, device=device, dtype=f64)
    tm = torch.as_tensor(m2, device=device, dtype=f64)
    if chapman:
        n_hat = (t1 + 1.0) * (t2 + 1.0) / (tm + 1.0) - 1.0
        var = ((t1 + 1.0) * (t2 + 1.0) * (t1 - tm) * (t2 - tm)) / (((tm + 1.0) ** 2) * (tm + 2.0))
    else:
        ok = tm > 0
        safe_m2 = torch.where(ok, tm, torch.ones_like(tm))
        inf = torch.full_like(tm, float("inf"))
        n_hat = torch.where(ok, t1 * t2 / safe_m2, inf)
        # Seber's variance for the uncorrected ratio. Undefined at m2 = 0 along with the estimate itself.
        var = torch.where(ok, (t1 * t1 * t2 * (t2 - tm)) / (safe_m2 ** 3), inf)
    out_n: npt.NDArray[np.float64] = n_hat.cpu().numpy()
    out_v: npt.NDArray[np.float64] = var.cpu().numpy()
    return out_n, out_v


def lincoln_petersen(n_both: int, n_model_only: int, n_human_only: int,
                     *, chapman: bool = True) -> RecaptureEstimate:
    """Estimate the population both observers were sampling from, and each observer's recall of it.

    n_both is the recaptured count (found by model and human), n_model_only and n_human_only the
    exclusives. Returns a RecaptureEstimate; `measured` is False with a reason when the counts cannot
    support an estimate.
    """
    both, model_only, human_only = int(n_both), int(n_model_only), int(n_human_only)
    if min(both, model_only, human_only) < 0:
        raise ValueError("capture counts cannot be negative")
    if both == 0:
        # No overlap means no information about what neither observer saw: the ratio is unbounded and the
        # Chapman correction would return a finite number that means nothing. Refuse rather than report it.
        return _unmeasured(both, model_only, human_only,
                           "no object was found by both observers; the population is unbounded above")
    n1 = float(both + model_only)   # everything the model found
    n2 = float(both + human_only)   # everything the human found
    if n1 == 0.0 or n2 == 0.0:
        return _unmeasured(both, model_only, human_only, "one observer found nothing")

    n_hat_a, var_a = _chapman_np(np.array([n1]), np.array([n2]), np.array([float(both)]), chapman)
    n_hat, var = float(n_hat_a[0]), float(var_a[0])
    # The population cannot be smaller than the union actually observed, whatever the arithmetic says.
    observed = float(both + model_only + human_only)
    n_hat = max(n_hat, observed)
    half = _Z95 * float(np.sqrt(max(var, 0.0)))
    lo, hi = max(n_hat - half, observed), n_hat + half
    return RecaptureEstimate(
        measured=True,
        population=round(n_hat, 4),
        lo=round(lo, 4),
        hi=round(hi, 4),
        model_recall=round(n1 / n_hat, 6),
        human_recall=round(n2 / n_hat, 6),
        # Inverted: a larger population means a smaller share of it was found.
        recall_lo=round(min(n1 / hi, 1.0), 6),
        recall_hi=round(min(n1 / lo, 1.0), 6),
        n_both=both, n_model_only=model_only, n_human_only=human_only,
        variance=round(var, 6),
    )


def stratified_recapture(counts: npt.ArrayLike, *, labels: list[str] | None = None,
                         chapman: bool = True, device: str | None = None) -> dict[str, Any]:
    """Per-stratum estimates plus a pooled estimate with propagated variance.

    `counts` is (S, 3): each row is (n_both, n_model_only, n_human_only) for one stratum. Pooling is a sum
    of per-stratum populations rather than an estimate over the collapsed counts, because collapsing
    assumes one capture probability across every stratum, and stratifying exists precisely because that is
    false: a crowded frame and an empty highway do not share a detection rate. Strata are treated as
    independent, so the pooled variance is the sum of the per-stratum variances.

    A stratum the estimator cannot measure (no overlap) is excluded from the pool and named in
    `unmeasured`, never silently counted as zero missed.

    Returns {"measured", "per_stratum", "pooled", "unmeasured", "n_strata"}.
    """
    arr = np.asarray(counts, dtype=np.float64).reshape(-1, 3)
    if arr.shape[0] == 0:
        return {"measured": False, "reason": "no strata", "per_stratum": [], "pooled": None,
                "unmeasured": [], "n_strata": 0}
    if np.any(arr < 0):
        raise ValueError("capture counts cannot be negative")
    names = labels if labels is not None else [f"stratum_{i}" for i in range(arr.shape[0])]
    if len(names) != arr.shape[0]:
        raise ValueError(f"labels has {len(names)} entries for {arr.shape[0]} strata")

    both, model_only, human_only = arr[:, 0], arr[:, 1], arr[:, 2]
    n1, n2 = both + model_only, both + human_only
    if _HAS_TORCH and device != "cpu" and torch.cuda.is_available():  # pragma: no cover - GPU path
        n_hat, var = _chapman_torch(n1, n2, both, chapman, str(device or "cuda"))
    else:
        n_hat, var = _chapman_np(n1, n2, both, chapman)

    observed = n1 + n2 - both
    n_hat = np.maximum(n_hat, observed)
    ok = both > 0

    per_stratum, unmeasured = [], []
    for i, name in enumerate(names):
        if not ok[i]:
            unmeasured.append(name)
            per_stratum.append({"stratum": name,
                                **_unmeasured(int(both[i]), int(model_only[i]), int(human_only[i]),
                                              "no overlap in this stratum").__dict__})
            continue
        half = _Z95 * float(np.sqrt(max(float(var[i]), 0.0)))
        lo = max(float(n_hat[i]) - half, float(observed[i]))
        hi = float(n_hat[i]) + half
        per_stratum.append({
            "stratum": name, "measured": True,
            "population": round(float(n_hat[i]), 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "model_recall": round(float(n1[i]) / float(n_hat[i]), 6),
            "human_recall": round(float(n2[i]) / float(n_hat[i]), 6),
            "recall_lo": round(min(float(n1[i]) / hi, 1.0), 6),
            "recall_hi": round(min(float(n1[i]) / lo, 1.0), 6),
            "n_both": int(both[i]), "n_model_only": int(model_only[i]),
            "n_human_only": int(human_only[i]), "variance": round(float(var[i]), 6),
        })

    if not ok.any():
        return {"measured": False, "reason": "no stratum had an object found by both observers",
                "per_stratum": per_stratum, "pooled": None, "unmeasured": unmeasured,
                "n_strata": int(arr.shape[0])}

    pop = float(n_hat[ok].sum())
    pooled_var = float(var[ok].sum())
    found_by_model = float(n1[ok].sum())
    found_by_human = float(n2[ok].sum())
    half = _Z95 * float(np.sqrt(max(pooled_var, 0.0)))
    lo, hi = max(pop - half, float(observed[ok].sum())), pop + half
    pooled = {
        "population": round(pop, 4), "lo": round(lo, 4), "hi": round(hi, 4),
        "model_recall": round(found_by_model / pop, 6),
        "human_recall": round(found_by_human / pop, 6),
        "recall_lo": round(min(found_by_model / hi, 1.0), 6),
        "recall_hi": round(min(found_by_model / lo, 1.0), 6),
        "variance": round(pooled_var, 6),
        "n_both": int(both[ok].sum()), "n_model_only": int(model_only[ok].sum()),
        "n_human_only": int(human_only[ok].sum()),
        "n_strata_pooled": int(ok.sum()),
    }
    return {"measured": True, "per_stratum": per_stratum, "pooled": pooled,
            "unmeasured": unmeasured, "n_strata": int(arr.shape[0])}
