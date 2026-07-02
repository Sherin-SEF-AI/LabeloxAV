"""Per-agent token/call budget: the enforcement behind the "per-agent token budgets" architecture rule.

An agent charges the budget once per LLM/VLM call and stops when it is exhausted, so no agent can run away
with unbounded model spend on a nightly patrol. Kept deliberately minimal (a call counter, not a token
tokenizer) until a second agent needs more; the LLM/VLM calls this fleet makes are the expensive unit, so
counting them is the right granularity.
"""

from __future__ import annotations


class TokenBudget:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max(0, int(max_calls))
        self.used = 0

    def charge(self, n: int = 1) -> None:
        self.used += n

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_calls

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def as_dict(self) -> dict:
        return {"max_calls": self.max_calls, "used": self.used, "remaining": self.remaining}
