# ADR-0001: An immutable prediction plane separate from annotations

Status: accepted. Date: 2026-07-27.

## Context

Evaluation drew model predictions from live `Object` rows filtered by machine `source`. Human review mutates
those rows in place (confirming a detection sets `source = "human"`), so a correct-and-confirmed detection left
the prediction population the moment it was confirmed. The harness scored the residue humans rejected, and the
published precision 0.034 / recall 0.018 was an artifact of destructive provenance. Every downstream that reads
a metric (the promotion gate, safety recall floors, drift, the flywheel budget) inherited the blindness.

## Decision

Predictions get their own append-only plane, `InferenceRun` + `Prediction`, that human review never writes to.
Inference writes `Prediction`; review writes `Object`; the two never share a row. Evaluation scores a named
`InferenceRun`, not corpus state. Inference runs at a low conf floor so the full score distribution is stored,
making a PR curve and a real AP computable. A run is keyed by `(model_version, gold_id, code_sha, params)` for
reproducibility.

## Alternatives considered

- **Patch the query** to exclude human rows and re-derive predictions from history. Rejected: the original
  confidence was already lost to the in-place mutation, so no PR curve is reconstructable, and the next review
  would corrupt the population again. The defect is structural and needs a structural fix.
- **A `model_version` column on `Object`.** Rejected: it conflates the model-owned and human-owned lifecycles on
  one row, which is exactly what caused the defect. Two lifecycles need two tables.
- **Store only aggregate metrics.** Rejected: an aggregate cannot be audited. Storing every prediction lets a
  confusion cell open to its real crops and lets the number be recomputed and trusted.

## Consequences

- Evaluation is auditable, reproducible, and inspectable; the metric is our own 101-point AP, not an opaque val
  pass. A second harness (the val-pass) cross-checks it and a divergence beyond epsilon blocks promotion.
- Historical predictions are only partially recoverable (`Review.before` has box and class, never conf), so a
  reconstructed run carries a null conf and the eval refuses AP for it.
- The invariant "no code path outside the inference writer mutates a `Prediction`" must be preserved in review.

See `docs/MEASUREMENT.md` for the full flow.
