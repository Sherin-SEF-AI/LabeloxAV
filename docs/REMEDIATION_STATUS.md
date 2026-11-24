# Audit remediation status

What the audits found, and what happened to it. Nothing is marked done unless a test fails without the fix.

This document was rewritten on 2026-08-18 from the code rather than edited, because the previous version
had stopped being usable. Its `## Closed since the last revision` heading was followed by three `###`
subsections of **open** work, so nine items appeared as both closed and open at once; the closed column was
the true one in every case that was checked. It also carried a suite size 1,300 tests behind reality. A
status document that contradicts itself is worse than none: it costs a reader time and then misleads them,
and this one is what the README points at.

**Suite as of 2026-11-24: 4,038 passing, 6 skipped, 4 xfailed** (web: 1,249 passing; browser: 8 passing, `make e2e-editor`). As of the 2026-08-18 rewrite it was 2,580 passing, 3 skipped, 4 xfailed (`make test`, full infra). Plus two GPU
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

## Fixed in the 2026-11-24 round

A whole-codebase pass: four read-only audits, then the fixes. Uncommitted on `feat/complete-gaps` at the
time of writing. Each row has a test that fails without the fix unless it says otherwise.

### Controls that existed and did not fire

| Gap | Fix | Test |
| --- | --- | --- |
| An import request could name any directory or archive the API can read (`Path(source_uri)`, no allowlist). Reachable by a reviewer and by an outside workforce holding a return key | `LBX_PATHS__IMPORT_ROOTS` allowlist, checked after symlinks resolve; empty means no local paths. A cap on what an archive may expand to | `tests/test_import_source_guard.py` |
| Session erasure deleted the frame images and nothing else. The uploaded source video (the unredacted copy), every mask, raster and point cloud stayed in the bucket under an erasure certificate | Every `*_uri` column on a row tied to the session, found from the schema, plus the session's own key prefixes; shared source media kept until the last session using it goes | `tests/test_retention_erasure.py` |
| A recording health-gate override, cancel-any-job, replacing a session's calibration, a flywheel dispatch and corpus-wide attribute writes sat at the annotator floor, unattributed | Write-only reviewer floors (a prefix floor would have gated the reads too), and the override, the cancel and calibration changes record who did them | `tests/test_route_auth.py` |
| Two simultaneous auto-label starts both found the GPU idle; re-detection checked training only; the PII backfill checked nothing and left no row | A transaction-scoped lock around check and insert; the backfill has a run row and the same guards | `tests/test_gpu_start_race.py` |
| The frame lock did not exist: the second person on a frame edited freely and met a 409 at save | Presence and an edit lock (`services/collab/presence.py`), enforced on the editor's saves by the server (423), with reviewer take-over | `tests/test_presence_*.py`, `web/e2e/presence.spec.ts` |
| The reviewer floor on `/api/objects` covered every route under it, so an annotator could draw a box and then could not move, relabel or delete it, and the clamp in `review_policy` written for exactly those requests never ran | The routes that edit one object are open to annotators, and `services/object_access.py` decides per object: their own or an unreviewed machine label, not ruled on by anyone else, not accepted | `tests/test_annotator_object_access.py`, `web/e2e/editor.spec.ts` |
| `/api/datasets/blob` confined itself with `"/datasets/" in uri` | Resolved to a key in our bucket and required to start with the prefix | `tests/test_ops_endpoints.py` |

### Things that reported success without doing anything

| Gap | Fix | Test |
| --- | --- | --- |
| `pip-audit --strict \|\| true` could not fail (it tries to look the project up on PyPI). The pins had 159 advisories across 21 packages by the time anyone looked | `scripts/audit_deps.py`: fails on any advisory not in `configs/pip_audit_baseline.txt`. **The 159 are baselined, not fixed** | CI |
| Ruff blocked on nine paths; the other 135 findings were reported and ignored | All fixed; the whole tree gates | CI |
| Bandit ended in `\|\| true` | Blocking at high severity (clean); medium and low are printed | CI |
| `POST /training/{id}/cancel` cancelled the job and answered 500 (`MissingGreenlet`) | Refresh before building the response | `tests/test_training_cancel_route.py` |
| The evidence panel stored a review response as the object it was showing; the response has no frame or image, so the image went blank and the footer threw | The response is merged onto the object; `api.review` is declared with the server's real type | `web/components/inspect/mergeReview.test.ts` |
| `/api/health`, an unauthenticated probe, created the bucket when it was missing | A read-only probe, off the event loop | `tests/test_ops_endpoints.py` |
| Nine import adapters read a track id and the importer ignored it; a 3D box rode in provenance and was never written to its column; a real nuScenes release imported with zero objects | Tracks and cuboids are written; the nuScenes adapter reads the standard tables; LIDAR_TOP keyframes are stored | `tests/test_import_tracks_and_cuboids.py`, `tests/test_nuscenes_standard_import.py` |

### Failure modes

| Gap | Fix | Test |
| --- | --- | --- |
| `eval_patch.prediction_id` and `.frame_id` (39M rows, cascading) and 49 other foreign keys had no index: the 0122 incident waiting to repeat on the next erasure | Migration 0123, and a test that fails on the next unindexed foreign key | `tests/test_fk_indexes.py` |
| An import or export in flight when the API restarted was failed ten minutes later and had to be started again, which for an import made a second session | A per-job advisory lock says exactly whether a job's process is alive; the next start continues it (an import into its own session, skipping stored frames) | `tests/test_job_recovery.py` |
| Thirty-odd handlers did blocking storage reads and image decodes on the event loop of a one-process server | `services/api/offload.py`; a source scan fails on the next one | `tests/test_event_loop_blocking.py` |
| `/metrics` ran a GROUP BY over the object table per scrape, for anyone who could reach it | Cached for `LBX_METRICS__DB_GAUGE_TTL_S`; optional bearer token | `tests/test_ops_endpoints.py` |
| 60 `limit` parameters had no upper bound (one defaulted to 200,000) | Bounded with `Annotated[int, Query(le=...)]`, which keeps the default an int for handlers called from Python | suite |
| No request id on log lines | `X-Request-ID` in, out, and bound to every log line of the request | `tests/test_request_id.py` |
| The same checkpoint was loaded up to three times in one process (SAM by the editor, the describe tool and Path B), and nothing could free a model somebody else had loaded | `core/model_manager.py`: one instance per model, leases, eviction on demand, one VRAM probe | `tests/test_model_manager.py` |
| No token or cost accounting for two wired cloud providers | A daily ledger (0125), every model call through `services/llm/ledger.post`, an optional daily cap | `tests/test_llm_ledger.py` |
| Containers ran as root; the infrastructure had no restart policy; backups ran when somebody remembered | An entrypoint that hands the volumes over and drops to an unprivileged user with no capabilities; `restart: unless-stopped` on the four infra services in the deployment overlay; `labeloxav backup-schedule` | `tests/test_deploy_composition.py`, `tests/test_backup_schedule.py`. The entrypoint was exercised against a root-owned volume in a container; **the full application image was not rebuilt** (the build did not finish on this network) |

### Measured, and not changed

Recorded because each was proposed as a win and is not one here (docs/MEASUREMENT.md, "Where auto-label
time goes"). Batching the detectors across frames is no faster and changes 7% to 12% of detections at a
batch of 4 to 8; `gpu.batch_size` exists and defaults to 1. A smaller segmenter input is 1.5x faster at a
mean mask IoU of 0.80. TensorRT for the detectors could at most recover 24 ms of a 300 ms frame.

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
| Multi-GPU: done in code, untested on hardware | The worker lock, the GPU slot and the training lease are per card since 2026-11-24 (`python -m services.gpu_worker --device cuda:1`), covered by tests against two device names on one database. It has not been run on a machine with two cards. Distributed training of ONE job across cards is still not attempted |
| TensorRT, LiteRT, Hailo export | A device toolchain. The capability gate refuses them by name rather than fabricating a result. For auto-label throughput specifically TensorRT is not worth wiring: measured 2026-11-24, the detectors are 24 ms of a 300 ms frame and segmentation is the rest (docs/MEASUREMENT.md) |
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
| Browser tests cover the editor and the frame lock, not the whole journey | `make e2e-editor` (in CI) drives the editor against a throwaway database: nudge, undo, cross-frame paste, multi-delete, view-only for the second person, the server refusing their save, take-over. The sign-in → review → export journey is still only the `make e2e` smoke, which needs a deployed stack and is not in CI |
| No visual regression on the canvas | For a Konva annotation tool this is the highest-value missing test type |
| 68 of 79 API routers are never imported by a test | The authz matrix now covers the route *table*; the handlers themselves largely are not exercised over HTTP |
| No API versioning; 90 of 787 operations declare what they return | Up from 41. The web app's types are generated from the schema (`web/lib/api.generated.ts`, `make clients`) and an undeclared response is typed `unknown`. The 51 typed in 2026-11 are the ones the web app calls most; `tests/test_response_contracts.py` proves no field is dropped. The other 697 are untyped |
| No soft delete | Zero `deleted_at` columns; all 17 DELETE routes are hard deletes. DPDPA erasure is separate and real |
| The shell is translated, page bodies are not | 357 keys in four locales cover the sidebar, menus, command palette, page titles, common buttons and the home page. Page bodies, the editor and format names are English. The Hindi, Kannada and Tamil were machine-written and no native speaker has reviewed them |
| Shadow/canary serving | `services/verdyx/shadow.py` is a correct accumulator with no producer |
| ~28 of 48 LiDAR endpoints have no UI; ontology repair is API-only | The UI can grow the vocabulary but not repair it |
| 45 swallowed `.catch(() => {})` remain | Baselined in `web/lib/noswallow.test.ts` so the count cannot grow. The baseline was measured, not adjudicated |
| `DomainPack.scene_model` is golden-frozen and consumed by nothing | Wire it or remove it; an unused frozen surface reads as done |
| No operational docs | INCIDENT_RESPONSE, DATA_RETENTION (a DPDPA obligation with working code and no operator doc), USER_GUIDE, TESTING, LICENSE |
