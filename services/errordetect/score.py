"""The one place a detector's score is turned into a score.

Every detector here emits a candidate onto a single queue that `list_candidates` ranks by score across all
of them, and the active-learning selector reads the same field. That makes the scale a shared contract
rather than each detector's private business, and it was not being kept:

  - near_dup stored frame similarity, which cannot fall below its own 0.96 gate, so 45,313 candidates sat
    between 0.986 and 1.0 and took 98.5% of the top thousand
  - embedding_outlier stored raw cosine distance, which runs to 2.0, and the corpus held a candidate at
    1.065 ranked above a detector reporting certainty
  - policy_violation and critic_flag emit hand-assigned constants which happen to be in range, with nothing
    saying they have to be

The first two were caught by reading. The third is the interesting case: it is correct today and correct by
luck, because a new rule written with a severity of 5.0 would rank above every calibrated detector and
nothing would say so.

`as_suspicion` is that contract in one function. It clamps rather than raising, because a detector is a
best-effort signal and taking the queue down over a badly-scaled rule would be a worse failure than ranking
it slightly wrong; the clamp is visible at every call site, which is what the guard in
tests/test_score_semantics.py checks for.

**What the number means.** How much this detector suspects this object is mislabelled, on [0, 1], comparable
with every other detector. It is not a probability: only `confident_learning` emits one of those, and
nothing here is calibrated until a detector has enough human verdicts for `detector_precision` to measure
it. Ranking currency, not a prediction.
"""

from __future__ import annotations


def as_suspicion(value: float) -> float:
    """Clamp a detector's raw quantity onto the shared [0, 1] suspicion scale, rounded for storage.

    A NaN becomes 0.0 rather than propagating: NaN compares false against every threshold, so a NaN score
    silently sorts to one end of the queue and never triggers a bound check.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:      # NaN
        return 0.0
    return round(min(1.0, max(0.0, v)), 4)
