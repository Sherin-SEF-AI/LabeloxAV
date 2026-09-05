# The autonomy program

One entry per work package of the feature program that started on 2026-09-05, in landing order.
Every number here comes from a real run over the real corpus or from the rule itself evaluated
exhaustively; where a number could not be produced honestly, its reason is printed instead.

The invariants every package keeps: `accepted` means a person ruled, forever, and the machine never
writes it; every autonomous write is one revertible chunked `AgentRun`; refuse with a reason rather
than guess, and an unmeasured quantity is never reported as zero; GPU work runs batch by batch behind
`gpu_slot`, `training_holds_gpu` and `VramGuard`; the killswitch gates all of it; every schema change
ships an Alembic migration whose `downgrade()` has been run.

---

## WP1. Sequential acceptance (SPRT) and the verdict-minute allocator

### The defect it fixes

Settlement (migration 0106) proved a class clean by a fixed draw: `sample_target(far)` crops, sized so
one defect still passes Wilson at the class's failure-rate bound. That is 110 crops at far 0.05, 283 at
0.02, 562 at 0.01. A fixed draw spends the same number of human verdicts on a class that is obviously
clean and on one that is obviously not, and the two live lots (motorcycle at far 0.01, traffic sign at
far 0.05) were waiting on 672 verdicts, roughly 67 human minutes, before either could say anything.

### The rule

`services/labelops/sampling.py::sprt_decision(defects, n, *, p0, p1, alpha=0.05, beta=0.10)` is Wald's
sequential probability ratio test on the defect rate. `p0 = far` is the rate the lot must reject,
`p1 = far / 2` the rate it must accept (`settlement.py::sprt_params`). The log-likelihood ratio moves
by `ln(p0/p1) = ln 2` on every defect and by `ln((1-p0)/(1-p1))` on every clean verdict; it stops at
`ln(beta/(1-alpha)) = -2.2513` (accept) or `ln((1-beta)/alpha) = 2.8904` (reject). The operating
characteristic is solved by bisection and reported per lot, and the expected remaining verdicts (Wald's
ASN from the current llr) is what the allocator reads.

Three things about how it is wired, because each one is where a sequential test goes wrong:

- **Accept is conjunctive.** A lot accepts only when the SPRT crosses its accept bound and
  `acceptance_decision` (the Wilson rule from 0106) also accepts at the same `(defects, n)`. Measured
  over every `(k, n)` up to the cap: the SPRT is the binding rule at 0 to 2 defects and Wilson is the
  binding rule at 3 or more, so neither implies the other and the conjunction is not decorative. Reject
  is the SPRT alone; it is the conservative direction and steps the class down exactly as before.
- **Verdicts arrive in draw order, and only whole increments count.** Triage served every batch in
  `(1-conf) * rarity * boost` order, which would have made "the verdicts so far" the hardest-first prefix
  and the early stop a biased one. For batches whose cycle id starts with `settle-`, triage now ranks by
  the crop's index in the lot's draw (`sample_object_ids`); the formula for every other batch is
  untouched and a test holds it there. `tally_lot` counts only increments judged to the completion floor
  (0.9), never a partial one, and `top_up_lot` refuses to draw while the latest increment is incomplete.
- **Increments are a prefix of one permutation.** The draw keeps the `settle-sample` salt and the
  `not in have` exclusion, so 25 more crops are the next 25 of the same md5 order. At
  `cap_n = sample_target(far)` the sample is identical to the old fixed draw and Wilson decides verbatim
  (`rule="wilson_at_cap"`), so the worst case of the sequential rule is exactly the old rule.

Migration `0107_settlement_sprt` adds `rule`, `cap_n`, `llr`, `sprt` and `increments` to
`settlement_lot`. `llr` is null until computed, never 0. Downgrade drops the five columns and a judging
lot then tallies as Wilson, the stricter rule, so nothing has to be moved; the round trip is exercised
on a seeded lot by `tests/test_migrations_roundtrip.py`.

### The spot check that never reached a person

`settle_lot` created `settlement_spot` rows on `settled` objects with no cycle id, and triage defaults
to `review,annotate`, so no spot was ever served and nothing wrote `human_verdict`. Spot objects are
now stamped `provenance.flywheel.cycle_id = spot-{lot8}`, the settlement notification links to
`/review/grid?flywheel=spot-{lot8}&states=settled`, and `apply_review_batch` writes `correct` or
`incorrect` onto the spot when the verdict lands (`ApplyResult.spot_judged`). A human ruling on a
`settled` object still upgrades it to `accepted`, as `state_for` already did.

### The allocator

`settlement.py::expected_remaining(lot)` gives, per judging lot, the expected verdicts to a decision
(clamped to `cap_n - n`), minutes at the 10 verdicts per minute the estimate has always used, and
`value = population * OC(p_hat) / minutes`: objects a passed lot would settle per minute of a person's
time. `/autonomy/state` returns the ranked `worklist` and `verdict_minutes_open`; the autonomy page
draws each lot's llr between its two bounds. `p_hat` is smoothed with one pseudo-verdict at the
indifference midpoint `(p0+p1)/2`, because a Laplace prior made an unjudged lot look half defective
and priced it at zero.

### Measured

From the rule itself, evaluated at every `(defects, n)` up to the cap for each tier:

| far | cap (0106) | clean lot accepts at n | first whole increment | verdicts saved | straight defects to reject |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 110 | 87 | 100 | 9% | 5 |
| 0.02 | 283 | 222 | 225 | 20% | 5 |
| 0.01 | 562 | 447 | 450 | 20% | 5 |

A lot with exactly one defect never accepts early under the conjunction at any tier; it runs to the cap
and Wilson accepts it there, as it did before. Five consecutive defects reject at any tier after 5
verdicts, where the fixed draw would have needed the whole sample.

**The two live lots produce no replay number.** `253cacf5` (motorcycle, far 0.01, 562 drawn) and
`be7b3273` (traffic sign, far 0.05, 110 drawn) both have 0 verdicts, so there is nothing to replay
through the SPRT. They were planned under 0106 and stay `rule='wilson', cap_n=0`; the tally treats that
as the fixed rule. Re-planning them under the sequential rule is the operator's call, and the first
verdict-minute numbers will come from whichever lot is judged first.

Tests: `tests/test_sprt.py` (21), `tests/test_settlement.py` (16, including the calibration umbrella
`test_settlement_changes_no_calibration_input`), `tests/test_migrations_roundtrip.py` (1).
