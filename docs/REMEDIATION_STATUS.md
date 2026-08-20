# Audit remediation status

What the audits found, and what happened to it. Nothing is marked done unless a test fails without the fix.

This document was rewritten on 2026-08-18 from the code rather than edited, because the previous version
had stopped being usable. Its `## Closed since the last revision` heading was followed by three `###`
subsections of **open** work, so nine items appeared as both closed and open at once; the closed column was
the true one in every case that was checked. It also carried a suite size 1,300 tests behind reality. A
status document that contradicts itself is worse than none: it costs a reader time and then misleads them,
and this one is what the README points at.

**Suite as of this rewrite: 2,580 passing, 3 skipped, 4 xfailed** (`make test`, full infra). Plus two GPU
tests that fail only under memory pressure and pass in isolation - see `tests/KNOWN_FAILURES.md`.

---

## Fixed in the current round (2026-08-18)

Twenty-three commits on `feat/lidar-3d-module`. Grouped by what the defect actually was.

### Controls that existed and did not fire

| Gap | Fix | Test |
| --- | --- | --- |
| The DPDPA export gate checked that a `PiiAudit` row existed, not that a blur covered anything. 82.4% of frames holding an annotated person have zero faces redacted and every one has a row, so the gate passed all of them | Annotation-relative coverage predicate: a frame is suspect when its own annotations imply a redaction no stored region covers. Containment, not IoU - a real face has IoU ≈ 0.05 against the person box, so an IoU rule would refuse every frame. Ships **advisory** | `tests/test_dpdpa_coverage.py`, `tests/test_dpdpa_gate.py` |
| `POST /frames/{id}/objects` wrote caller-supplied `state`, defaulting to `accepted`, at the annotator floor. `review_policy.state_for` existed for exactly this and was only reachable from reviewer-floored paths | Clamped by role; unlisted states are a 400 rather than a 500 from the check constraint | `tests/test_phase1_guards.py` |
| The `?token=` exception was scoped to the `/api/events/` prefix, which holds `PATCH` and `DELETE /api/events/{id}` plus three JSON reads. A full-privilege token in a URL authorised row deletion | Scoped to the four exact SSE stream paths and to GET | same |
| `is_gated` took the newest `session_health` row from any writer, and SANYX writes to that table - so a SANYX pass after an inspector fail silently re-enabled auto-labeling | Scoped to the inspector's own `indexer_version`. There was no test for `is_gated` at all | same |
| `refuse_reason` - written after a run moved 1,047 buses into a bus shelter - was missing from `reconcile` (SigLIP zero-shot at conf 0.55) and `nl_edit`, which also did not exclude `source == "human"` | Applied at both. Not `cuboid_agent`: it writes `cuboid_3d`, never `class_id`, so there is no move to refuse | same |
| Gold sealing, calibration fitting, forced retrain and the SLO ledger sat at the annotator floor | Per-route `require_role`. Not prefix floors, which would have gated the reads too | `tests/test_route_auth.py`, `tests/test_auth.py` |
| Minting a token for another account left no trace | `record_activity` with the admin as actor | — |

### Things that reported success without doing anything

| Gap | Fix | Test |
| --- | --- | --- |
| `download_pii_models.py` logged failures and exited 0. A transient upstream 404 disarmed the redaction gate and CI's provisioning step passed | Exits non-zero; pinned to a commit SHA rather than `raw/main`; retries; rejects a 200 carrying an error page | CI |
| `make up`'s `grep -v healthy` also matched "un**healthy**", and a service with no healthcheck held the loop open forever, so it timed out every time and exited 0 | Gates on the four services it names; converges in under a second | — |
| A green suite could mean most of it never ran (~206 tests skip on a Redis ping) | Pass floor and skip ceiling in `pytest_sessionfinish` | — |
| `make backup` piped `pg_dump` into `gzip`, so a failed dump produced a valid 20-byte `.gz` and a green target. It also hardcoded the db name and *printed* the MinIO command | `scripts/backup.sh` + `scripts/restore.sh`: both halves together, refuses a half set | — |
| A component test was not failing - it was never collected. `environment: "node"` with an include of `*.test.ts` meant `.test.tsx` matched nothing | Two tiers (node + jsdom), coverage ratchet in CI | `LoadState`, `Toaster`, `PageShell` render tests |

### Failures that rendered as success

| Gap | Fix | Test |
| --- | --- | --- |
| Triage's loader was `try`/`finally` with no `catch`, so a dropped request rendered **"Queue is clear"** - telling an annotator their shift was over | `catch` + `LoadState` | — |
| `/analytics` had no error state at all: a dead backend rendered "no objects yet" over a 570k-object corpus | same | — |
| `/projects` swallowed its project-list fetch, so a manager saw an installation with no projects | same, plus a guard against new swallows | `web/lib/noswallow.test.ts` |
| Five review-queue verdicts were unhandled promises | One guarded helper | — |
| `humanizeError` fell through to `String(e)`, so a plain object rendered as `[object Object]` | Reads `detail`/`message`/`error`, falls back to JSON | `LoadState.test.tsx` |

### The test isolation guard

`_provision_test_db` refuses to run against a non-test database - and that refusal sat *after* an early
return taken when no test carried the `db` marker. 162 files touch the DB and 45 said so, so `make
test-unit` and any single-file run wrote rows with the guard never evaluated. This is the mechanism behind
the 1,730 fixture sessions purged from the real corpus on 2026-07-30.

Guard hoisted; 115 files marked; `tests/test_db_markers.py` keeps the set honest with a non-triviality
floor. **The unit tier is now real**: with Postgres stopped it was unrunnable and now completes 1,449
passed in 60s.

### Correctness

| Gap | Fix | Test |
| --- | --- | --- |
| `CLASS_HEIGHT_M` was keyed 0-based against a 1-based ontology. `pedestrian`, `rider` and `cattle` fell outside it entirely and silently returned `None`; id 5, commented "truck" at 3.2 m, is `delivery_rider_bike` | Keyed by name, resolved through the pack. Its old test asserted the table against itself | `tests/test_oraclyx_m14.py` |
| `frustum_indices` - the primary cuboid source - had no `calib` parameter, and the calibration *validator* measured reprojection against nominal config, so a drifted extrinsic passed its own check | Threaded at both | `tests/test_lidar_calib_threading.py` |
| Bulk review committed edits then stamped the batch, so dying between left it permanently un-revertible | One transaction | `tests/test_review_batch.py` |
| A promoted champion pointed at a build name, so two `loop-v1` builds a week apart were different corpora | Content fingerprint over the built labels | `tests/test_training_lineage.py` |

### Deployment

| Gap | Fix |
| --- | --- |
| The closed loop had no deployed driver: `govern-daemon` was a Makefile target nothing supervised | A service, with the workers behind profiles | `tests/test_deploy_composition.py` |
| A fresh install could not ingest - no PII weights, and the anonymizer refuses to construct without them | One-shot `pii-models` the API waits on |
| Job reaping ran only at API boot, so a dead task disabled auto-labeling until someone restarted | On the daemon's cadence |
| The API could not correctly run two workers: memory-only rate limiter (N workers = N× the budget), in-process MFA and OIDC state ((N−1)/N of sign-ins failed) | Redis token bucket in Lua; shared ephemeral store | `tests/test_ratelimit_shared.py` |
| CORS hardcoded to localhost | `LBX_CORS__ORIGINS` | `tests/test_deploy_composition.py` |

### Product

| Gap | Fix |
| --- | --- |
| "Confirm frame" was gated on `objects.length` while the reducer acts on `touched`, so it silently did nothing while still burning an undo slot - and never advanced | Gated on `touched`, advances, `A` shortcut too |
| `selectBy` never marked anything touched, so "select all low-confidence" then Confirm accepted **nothing** | Fixed |
| `saved` did not remap `tmp-` ids in `touched`, so a drawn object stopped counting as reviewed after the first autosave | Fixed |
| No `beforeunload`: a closing tab took unsaved edits with it | Armed while dirty |
| The editor mounted no chrome, so notifications and the palette were invisible during the longest activity | Bell + palette mounted |
| `ToastAction` was built with a written rationale and had zero call sites | Wired into bulk relabel |
| An issue notified the reviewer rota, so the annotator whose label it was about was the only participant never told | Also addressed to them, plus a `mine` filter and an inbox |
| No focus ring anywhere, hover-only tooltips, no skip link | One `:focus-visible` system, focusable tooltips, skip link + `main` landmark |

---

## Corrections to the audits

Findings that were wrong. Recorded because acting on them would have caused damage.

| Claimed | Actually |
| --- | --- |
| `main` is 527 commits behind; the mainline is a stale feature branch | Local `main` was a stale *ref*. `origin/main` was current and green, with work merged continuously through PRs #20–29 |
| `services/agent/weather.py` is an orphan with zero references - delete it | Imported by `services/agent/fleet_dispatch.py` at two call sites. Deleting it would have broken the fleet dispatcher |
| The timeline and multicam routes have no back link of any kind | Both mount `BackButton`. The real gap was the fallback going to `/` on a deep link |
| `acceptAll` rubber-stamps every object | Already gated on `touched`, with a comment explaining why. The live defects were different and narrower |
| The PII URL moved upstream | It 404'd transiently and resolves again. Worse, not better: a re-run would have gone green and buried it |
| `cuboid_agent` needs the `refuse_reason` guard | It never writes `class_id`. There is no move to refuse |

---

## Open

### Blocked on hardware or a paid resource

| Gap | What it needs |
| --- | --- |
| No learned 3D detector in the loop | OpenPCDet or Pointcept and a CUDA device. The KITTI builder is complete; `train()` refuses by name rather than training a 2D model on 3D labels. Also needs real LiDAR - the running path lifts cuboids from monocular depth and inherits its scale error |
| 3D semantic segmentation is cuboid membership plus a ground plane | PTv3 inference, plus the runtime above |
| Multi-GPU / distributed training | The worker holds a Postgres advisory lock as a global GPU mutex by design. Changing that needs the scheduling model redesigned and more than one GPU to test on |
| TensorRT, LiteRT, Hailo export | A device toolchain. The capability gate refuses them by name rather than fabricating a result |
| Cloud training/autolabel/relabel park in `queued-cloud` | A provisioned pod. The data-movement contract is implemented and tested |

### Needs a product decision

| Gap | The decision |
| --- | --- |
| Single-tenant by construction. `session.project_id` exists (migration 0092) and `services/api/tenancy.py` provides the scope helpers - **with zero production callers**. `User` has no project binding, so scope is not expressible from a request, and both ingest paths write `project_id = NULL` | Whether to be multi-tenant, and whether to enforce in the session layer or with Postgres row-level security |
| Authorization is three global roles by path prefix, with no per-project or per-object ACL | What the permission model should be |

### Outstanding work

| Gap | Note |
| --- | --- |
| The DPDPA coverage gate is **advisory** | Run `POST /api/agent/reanalyze/all`, watch `coverage_gaps` reach zero in the `dpdpa.gate` log, then set `LBX_PII__COVERAGE_GATE=enforcing`. The whole suite already passes under enforcing |
| Nothing measures the *realized* precision of the auto-accepted subset against humans | The gate sits at P(correct) ≈ 0.45 and its isotonic curve is fit against a VLM judging the detector, because human review is 223 accepts / 6 rejects. The control-sample machinery exists; the gate does not feed it |
| No e2e / browser tests | Playwright is on disk and invoked by nothing. The sign-in → annotate → review → export journey has no automated coverage |
| No visual regression on the canvas | For a Konva annotation tool this is the highest-value missing test type |
| 68 of 79 API routers are never imported by a test | The authz matrix now covers the route *table*; the handlers themselves largely are not exercised over HTTP |
| No API versioning; 3 of 674 routes declare `response_model` | The generated OpenAPI is almost entirely untyped, so no client can be generated from it |
| No soft delete | Zero `deleted_at` columns; all 17 DELETE routes are hard deletes. DPDPA erasure is separate and real |
| `web/lib/i18n.ts` has four locales and one consumer | The annotator surface is hardcoded English |
| Shadow/canary serving | `services/verdyx/shadow.py` is a correct accumulator with no producer |
| ~28 of 48 LiDAR endpoints have no UI; ontology repair is API-only | The UI can grow the vocabulary but not repair it |
| 45 swallowed `.catch(() => {})` remain | Baselined in `web/lib/noswallow.test.ts` so the count cannot grow. The baseline was measured, not adjudicated |
| `DomainPack.scene_model` is golden-frozen and consumed by nothing | Wire it or remove it; an unused frozen surface reads as done |
| No operational docs | INCIDENT_RESPONSE, DATA_RETENTION (a DPDPA obligation with working code and no operator doc), USER_GUIDE, TESTING, LICENSE |
