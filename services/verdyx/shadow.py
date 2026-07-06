"""VERDYX continuous shadow eval + regression triage (M15). The champion should be watched between formal
evaluations: run it in shadow over the incoming stream, accumulate per-slice recall online, and raise a
regression the moment a protected slice degrades past a margin, with the offending slice named so triage is not
a hunt. Pure accumulator + a triage function over slice deltas, so the online mean and the breach logic are
testable without a live stream."""

from __future__ import annotations


class ShadowEval:
    """An online per-slice recall accumulator for the shadow-running champion. Feed it per-object outcomes as
    the stream arrives; it keeps a running recall per slice without holding the objects. Cheap enough to run
    continuously alongside serving."""

    def __init__(self) -> None:
        self._hit: dict[str, int] = {}
        self._tot: dict[str, int] = {}

    def observe(self, slice_id: str, detected: bool) -> None:
        self._tot[slice_id] = self._tot.get(slice_id, 0) + 1
        if detected:
            self._hit[slice_id] = self._hit.get(slice_id, 0) + 1

    def recall(self, slice_id: str) -> float | None:
        t = self._tot.get(slice_id, 0)
        return round(self._hit.get(slice_id, 0) / t, 4) if t else None

    def snapshot(self) -> dict:
        return {s: {"recall": self.recall(s), "n": self._tot[s]} for s in sorted(self._tot)}


def regression_triage(baseline: dict[str, float], current: dict[str, float], margin: float = 0.05,
                      protected: set[str] | None = None) -> dict:
    """Compare a current shadow snapshot's per-slice recall to a baseline. A slice that drops more than
    ``margin`` is a regression; a drop on a protected slice is escalated regardless of headline movement.
    Returns the regressed slices worst-first and whether any protected slice regressed (the shadow alarm)."""
    protected = protected or set()
    regressions = []
    for s, base in baseline.items():
        cur = current.get(s)
        if cur is None or base is None:
            continue
        delta = cur - base
        if delta < -margin:
            regressions.append({"slice": s, "baseline": round(base, 4), "current": round(cur, 4),
                                "delta": round(delta, 4), "protected": s in protected})
    regressions.sort(key=lambda r: r["delta"])
    protected_hit = any(r["protected"] for r in regressions)
    return {"regressions": regressions, "n_regressions": len(regressions),
            "protected_regression": protected_hit,
            "alarm": protected_hit or len(regressions) > 0}
