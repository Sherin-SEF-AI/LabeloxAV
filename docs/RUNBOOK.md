# Runbook

Operational procedures for LabeloxAV. Architecture is in `ARCHITECTURE.md`; the evaluation flow is in
`MEASUREMENT.md`. Commands assume the repo root and the project venv at `.venv`.

## Bring-up

```
make install          # create .venv and install dev deps (no GPU wheels)
make up               # start Postgres (postgis+pgvector), MinIO, Redis, Redpanda; wait for healthy
make migrate          # bring the database to the head Alembic schema
make health           # probe Postgres, Redis, MinIO and print per-dependency status
```

The backend runs without `--reload` (restart after backend edits). The frontend is `cd web && npm run dev`
(port 3000, which same-origin-proxies `/api` to the backend on 8000).

## Auth bootstrap

Auth is on by default. A fresh browser has no token, so:

- Locally, `POST /api/auth/dev-login` (env-guarded to `local`) mints an admin token to bootstrap the UI.
- The first user can also be created before any user exists (`POST /api/users` is allowed only while the user
  table is empty).
- Tokens expire (default 12h) and roll silently via `POST /api/auth/refresh`. To revoke a user's tokens, an
  admin calls `POST /api/users/{id}/revoke-tokens`, which bumps their `token_version` and invalidates every
  outstanding token for them.

If reads start returning 401 after an upgrade, that is the fail-closed default working: the caller needs a
token. Only the load-balancer probes are public: `/api/health` (liveness, always 200 with a status body)
and `/api/readyz` (readiness, 503 unless every dependency answers).

## The measurement and promotion flow

This is the sequence that produces a trustworthy metric. Do not read a number that skipped a step.

```
lbx-gold ...                                  # seal a gold set from human-verified frames (frozen object-id list)
lbx-infer --model <version> --gold <gold_id>  # run inference into the prediction plane at a low conf floor
```

Then score and gate through the API or `services/analytics/evaluation.py::evaluate_gold_patches(run_id=...)`:

- `GET /api/explore/gold/{gold_id}/provenance` splits the gold set into human-confirmed detections versus
  boxes drawn from scratch, so a genuine model miss is not confused with a confirmed hit.
- `services/govern/gold_eval.py` runs both harnesses (val-pass and prediction plane) and flags
  `harness_divergent` if they disagree beyond `govern.harness_reconcile_epsilon`.
- `services/govern/champion.py::champion_gate` decides promotion. It refuses a reconstructed or harness-
  divergent run outright and holds the safety-class AP and recall floors.

Reconstructed runs (from `scripts/backfill_prediction_from_review.py`) exist so historical frames are not
silently empty. They carry a null confidence, so the eval refuses AP for them and the gate refuses to promote
on them. Never promote on a reconstructed run.

## Common operations

- **Ingest a session**: `make ingest` (or the ingest CLI). Frames that fail the quality gate are rejected; a
  clean PII audit is written per accepted frame.
- **Auto-label**: `make label`, or `POST /api/autolabel/start` with a `session_id`.
- **Export a dataset**: `POST /api/datasets/export` (reviewer role). The DPDPA pre-sale gate refuses a slice
  that still contains un-redacted PII (returns 422 with the blocking sessions).
- **Re-detect the corpus**: `POST /api/autolabel/redetect-all`. This clears machine-source objects on each
  frame and re-runs; human-source objects are never touched.

## Tests

```
make test-unit        # fast pure-unit tier: no Postgres, MinIO, GPU, or Redis
make test             # full suite (needs infra up); runs with auth on, against labeloxav_test
```

The suite provisions and uses `labeloxav_test` and refuses to run against any database whose name does not
contain "test": the durable guard against the test-pollution that once corrupted the dev corpus. Markers `db`,
`gpu`, and `infra` select the tiers; `make test-unit` deselects all three.

The suite is not green, and the baseline is recorded in `tests/KNOWN_FAILURES.md`: a run that fails only
tests named there is at baseline, anything else is a regression. `tests/test_known_failures.py` keeps that
list from rotting (every name must still exist, every xfail must be documented, every category must state
what would fix it). Three categories: synthetic frames rejected by the ingest quality gate (xfailed), tests
that assert on corpus-wide statistics against a shared database and so depend on execution order, and tests
needing a local Ollama.

## Troubleshooting

- **`make health` shows a degraded dependency**: check `docker compose ps`; `docker compose logs <svc>`.
- **A migration will not apply**: confirm you are on head with `alembic history`; every migration in this repo
  has a working `downgrade`, so `alembic downgrade -1` is safe to step back.
- **The web build fails on a prerender error**: a client page that reads the query string via `useSearchParams`
  must wrap its body in a `<Suspense>` boundary (see the login and search pages for the pattern), or Next fails
  the static prerender.
- **A promotion will not go through**: read the gate's refusal reason. A reconstructed or harness-divergent run,
  a safety-class recall below floor, or a failure to beat the champion mAP each block it by design.
- **Cloud GPU seems stuck on**: a watchdog enforces the idle and max-session limits and teardown runs on
  shutdown, but you can force it with the cloud disconnect endpoint; `make` cloud targets manage the pod.
