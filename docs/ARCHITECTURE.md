# Architecture

LabeloxAV is a closed-loop data engine: it ingests fleet sensor data, auto-labels it, gates the labels on
confidence, routes the uncertain ones to human review, evaluates the model against a frozen yardstick, and
promotes a new model only when the evaluation clears a governance gate. This document is the map. For how the
evaluation number is produced and why it can be trusted, read `MEASUREMENT.md` first; it is the document that
would have caught the defect the prediction plane fixes.

## The spine

Everything hangs off three tables, in this order:

- **Session**: one recording from one vehicle (`vehicle_id`, time span, city, sensor set, ontology version).
- **Frame**: one camera image at one timestamp in a session (`img_uri`, `cam_id`, width/height, quality).
- **Object**: one annotation on one frame (`class_id` into the ontology, `bbox`, optional mask/keypoints/
  polyline/cuboid, `conf`, `state`, `source`, `provenance`, `version`).

An `Object` is human-owned once a person touches it. This is the invariant that the prediction plane exists to
protect: review mutates `Object` in place (confirming sets `source = "human"`), so `Object` cannot also be the
record of what the model predicted. Model output lives in its own plane.

## The two planes

- **Object** is the annotation plane. Review writes here.
- **Prediction** (`InferenceRun` + `Prediction`) is the model-output plane, append-only, that review never
  writes to. Inference writes it once per run at a low confidence floor so the full score distribution is
  captured and a real PR curve and AP are computable. A run is keyed by `(model_version, gold_id, code_sha,
  params)` for reproducibility.

They never share a row. Evaluation scores a named `InferenceRun`, not live corpus state. This is the core of
`MEASUREMENT.md` and the reason the published metric is auditable.

## The closed loop

1. **Ingest** (`services/ingest`): decode fleet video/MCAP into frames, run the quality gate, write frames and
   a per-frame PII anonymization audit (DPDPA).
2. **Auto-label** (`services/autolabel`): run the detection stack, write `Object` rows with a machine `source`
   and a `conf`. Provenance records which model paths agreed.
3. **Confidence gate** (`services/govern`): high-confidence agreed detections auto-accept; the uncertain and
   the safety-critical route to review.
4. **Review** (`services/api/routers/review.py`): a human confirms, reclassifies, adjusts geometry, or rejects.
   Every review writes a `Review` row (the before-state, for audit and reconstruction) and advances the object.
5. **Evaluate** (`services/verdyx/inference_run.py` then `services/analytics/evaluation.py`): score a
   registered model against a sealed gold set through the prediction plane; a second harness (the ultralytics
   val-pass in `services/govern/gold_eval.py`) cross-checks it and a divergence blocks promotion.
6. **Promote** (`services/govern/champion.py`): a pure gate decides whether a challenger beats the champion,
   holds the safety-class floors, and is neither reconstructed nor harness-divergent.
7. **Active learning / flywheel** (`services/activelearn`, `services/agent`): the gate's per-class deficits and
   the error detectors build the next review batch, closing the loop.

The hot path (Session to Frame to Object, the confidence gate, governance) is deliberately narrow and stable.
New capability is added around it, not through it.

## Domain packs

The engine is multi-domain. `core`, `services`, and `db` are the domain-neutral engine; a `DomainPack`
(`packs/base.py`, resolved through `packs/registry.py`) supplies the domain specifics (ontology, scene model,
ingest specialization). The AV pack and the Sec (security/surveillance) pack are the two concrete packs. An
import-linter contract (`.venv/bin/lint-imports`, run in CI) enforces the one structural rule that keeps this
honest: the engine core must never statically import a concrete pack; the only bridge is `packs.registry`.

## API and auth

The backend is FastAPI over async SQLAlchemy 2.0 (Postgres), with MinIO for objects and Redis/Redpanda for
queues and the event bus. The review UI reads and writes only through the API.

Auth is deny-by-default and fails closed (`services/api/main.py`):

- Every request, read or write, needs a signed v2 Bearer token unless its path is in a tiny public-read
  allowlist (only `/api/health`). A newly added route is gated automatically.
- Tokens are stateless (`services/api/auth_token.py`): `lbx2.<b64url(payload)>.<b64url(hmac)>` with an expiry
  and a `token_version`. Revocation is a version bump on the user row; expiry is enforced in the verifier.
- Role floors (annotator < reviewer < admin) apply by path. A startup backstop refuses to boot if any mounted
  read is public without a security-reviewed approval.

The frontend (`web`, Next.js App Router) rides the token on every request and rolls it before expiry through
`/auth/refresh`.

## Governance and measurement

The promotion gate reads metrics, never raw corpus state. It fails closed: a reconstructed run (predictions
backfilled from review history, no real confidence) or a harness-divergent run cannot promote. Safety classes
(vulnerable road users, animals) are held to tighter AP and recall floors than the default. The full flow and
the real current numbers are in `MEASUREMENT.md`; the design decision behind the prediction plane is
`adr/0001-prediction-plane.md`.

## CI

`.github/workflows/ci.yml` runs four things: a lint job (blocking secret scan, import contract, ruff on the
security/measurement modules, mypy `--strict` on a typed allowlist; informational ruff/mypy/bandit/pip-audit
over the rest), a web job (blocking tsc, vitest, and production build), and a test job that spins up real infra
and runs the suite against an isolated `labeloxav_test` database. The suite runs with auth on by default, so it
exercises the production posture.
