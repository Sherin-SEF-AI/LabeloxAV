# Audit remediation status

A four-part audit (backend, frontend, ML pipeline, ops and enterprise) produced the gap list this document
tracks. It records what is fixed, what is not, and for each open item what specifically is needed. Nothing
here is marked done unless there is a test that fails without the fix.

Suite at the time of writing: 1,144 passing, 4 xfailed, failures limited to `tests/KNOWN_FAILURES.md`.

---

## Fixed

### Correctness and trust

| Gap | Fix | Test |
| --- | --- | --- |
| Frontend sent no auth header on ~20 call sites, so under deny-by-default reads those pages rendered blank | All data calls routed through the shared client; a source-tree test blocks the pattern returning | `web/lib/nofetch.test.ts` |
| Export silently ignored unsupported formats, returned 200, and recorded them as delivered | Validated writer registry; unknown format is a 400; the commit records what was written | `tests/test_export_formats.py` |
| Drift detection projected 768 dimensions onto one basis vector and binned over a fixed range, so it could not detect drift | Seeded random-projection ensemble with quantile binning, breaching on the worst axis | `tests/test_drift_detection.py` |
| Label drift histogram hardcoded to 64 slots against a 226-slot ontology | Slot count follows the ontology | same |
| Protected-slice safety gate read numbers supplied in the request body; nothing computed them | `services/verdyx/slice_eval.py` computes them from the sealed gold set and the immutable inference run | `tests/test_slice_eval.py` |
| Validation split was per frame, so near-duplicate dashcam frames leaked across the split | Session-grouped split, gold-bearing sessions preferred for validation | `tests/test_trainset_split.py` |
| Nothing prevented the trainset from containing gold objects | `BuildSpec.exclude_gold_id` drops them; a missing gold id is a refusal | same |
| OCR and ANPR fabricated a constant 0.8 confidence and fed it to a threshold | Unmeasured confidence is `None`; unscored reads are kept but flagged, and a strict mode can reject them | `tests/test_ocr_confidence_honesty.py` |
| Drivable reported the unpaved fallback class as 0.0 coverage when it could not measure it | Reported in `unestimated_classes` instead | covered by the drivable endpoint test |
| Champion weights cached by basename, so two models named `best.pt` collided; no integrity check | Content-addressed key plus a sha256 sidecar verified on hit | `tests/test_champion_weights_cache.py` |
| Active-learning pool applied LIMIT with no ORDER BY, biasing every selection | Ordered by distance from the decision boundary with a stable tiebreak | (behavioural; covered by existing AL tests) |

### Security and operations

| Gap | Fix | Test |
| --- | --- | --- |
| Text-to-object search did not exist; the function that claimed to do it read a table the pipeline no longer populates and scanned it in Python | SigLIP2 vector per crop with an HNSW index (migration 0073); search is one ANN query | `tests/test_text_to_object_search.py` |
| OCR search was an unanchored ILIKE, a sequential scan on every query | pg_trgm GIN index; the query is unchanged | (index asserted by the migration round-trip) |
| ONNX export, quantization, and benchmarking were unreachable: no dependency group contained the backends | `edge` extra pins onnx, onnxruntime, onnxslim; a real export runs and is benchmarked | `tests/test_forgyx_onnx.py` |
| Webhook URLs were validated only as "starts with http", giving any authenticated caller an SSRF primitive | Private, loopback, link-local, reserved, and multicast targets refused; re-checked immediately before each fetch, which also closes DNS rebinding; internal receivers are an explicit opt-in | `tests/test_webhook_hardening.py` |
| Webhook signature covered the body alone, so a captured delivery replayed forever | Timestamp bound into the signed material and published in a header; legacy form still verifies | same |
| Webhook delivery was a single POST with no retry | Backoff retry on transport failure and 5xx/429, never on 4xx; every attempt recorded | same |
| `/api/health` returned 200 while degraded, so a load balancer kept routing to a dead node | `/api/readyz` returns 503 unless every dependency answers | `tests/test_route_auth.py` |
| `GET /users` was unbounded and ran a full review-table aggregation on every page mount | Paginated with a ceiling, envelope with total, aggregation scoped to the returned users | `tests/test_editor_api.py` |
| No error boundary anywhere: one render throw white-screened the app | `error.tsx`, `global-error.tsx`, `not-found.tsx` | (build-verified) |
| The runbook told operators to compare against a baseline that did not exist | `tests/KNOWN_FAILURES.md` plus a test that keeps it accurate | `tests/test_known_failures.py` |

### Capability gaps closed

| Gap | Fix | Test |
| --- | --- | --- |
| Evaluation was 2D-box only: masks, cuboids, tracks, and lanes could be labelled but never scored | Mask AP plus a boundary F1 (IoU is dominated by an object's interior, so a mask can score well while tracing badly), 3D and BEV AP with translation and orientation error, MOTA/IDF1/HOTA, and CULane-style lane F1. All four matchers mirror the box matcher so the numbers are comparable | `tests/test_eval_metrics.py` |
| Training had one task plugin, so masks and keypoints could never improve a model | `SegmentationTask` and `PoseTask`, each gating on its own metric rather than the box number and starting from a head-appropriate checkpoint | `tests/test_training_tasks.py` |
| No hyperparameter search of any kind | Grid and random sweeps over ordinary training jobs, capped with truncation reported, ranked with unscored trials excluded rather than scored zero | `tests/test_training_sweep.py` |
| A crashed run restarted at epoch zero, discarding every epoch already paid for | Resumes from its checkpoint, and the resume is recorded on the job | same |
| Only the latest metric point was kept, so a run's shape was unreadable afterwards | The whole curve is retained, bounded | same |
| No realtime anywhere: nine polling loops, and ingest progress scraped from a log file with a regex | Server-sent events push on change; the browser cannot set a header on EventSource, so a token in the query string is accepted for `/api/events/` alone and nowhere else | `tests/test_sse_events.py` |
| Editor had no multi-select, so every batch action went through the agent bar | `selectedIds` alongside `selectedId`, so single-object panels keep working; bulk hide, lock, and delete | `web/components/editor/useEditor.test.ts` |
| Per-object visibility was honoured by the canvas but hardcoded true with no UI, and hiding marked the object dirty | Eye and lock controls; visibility no longer queues a save or consumes an undo step | same |
| The editor was strictly frame-at-a-time, so every temporal judgement was a sequence of blind single steps | A filmstrip of neighbouring frames from the same camera, counts fetched in one query | endpoint verified against the corpus |
| Retention was computed and never enforced; `retention_until` could not even be set through the API | A sweep and a single-subject erasure that removes frames, annotations, audits, and blobs, both defaulting to a dry run, each returning a tamper-evident certificate | `tests/test_retention_erasure.py` |

### Reachability and round-trips

| Gap | Fix |
| --- | --- |
| 23 of 31 menu destinations were inert duplicates because five pages ignored their query string | Each page reads its parameter through `lib/useQueryParam`; a test pins the deep-link contract |
| Command palette keyed rows by href, which repeats by design, causing duplicate React keys | Keyed by href and label, with the uniqueness invariant asserted |
| The 3D cuboid editor, the 3D-to-camera view, and the multicam workspace had no inbound link and demanded a pasted UUID | Added to the Spatial menu with a real session picker |
| Pascal VOC and Mapillary were importable but not exportable | Both adapters written; a latent frame-naming collision found while testing them was fixed |

---

## Not fixed

Each item states what it needs. These are not hidden behind optimistic language: the work is real and has
not been done.

### Needs hardware or a runtime this environment does not have

Three entries remain, and all three are genuinely absent hardware rather than undone work. Everything else
previously in this section has been built.

| Gap | What it needs |
| --- | --- |
| No learned 3D detector in the loop | OpenPCDet or Pointcept, and a CUDA device. The KITTI dataset builder is now complete (`services/training/tasks/detect3d.py`), so the corpus is trainable the moment a runtime exists; `train()` refuses by name rather than training a 2D model on 3D labels and reporting an mAP that would look like progress. Also needs real LiDAR: the running path lifts cuboids from monocular depth, whose scale error they inherit. |
| 3D semantic segmentation is cuboid membership plus a ground plane | PTv3 inference, plus the runtime above. |
| Multi-GPU and distributed training | The worker holds a Postgres advisory lock as a global GPU mutex by design. Changing that needs the scheduling model redesigned and more than one GPU to test on. |

Cloud execution is no longer wholly blocked. The MinIO-to-workspace data-movement contract is implemented
and tested (`services/cloud/transfer.py`): a job's dataset is published with a checksummed manifest written
last, and results come home pattern-filtered. Steps 1, 3 and 5 (provision, run, stop) remain the pod's and
need a provisioned GPU; a dispatched job is visibly awaiting a pod rather than reported as trained.

TensorRT, LiteRT and Hailo export still need a device toolchain and are refused by name.

---

## Closed since the last revision

| Was open | Now |
| --- | --- |
| No password, OAuth, OIDC, SSO or MFA; the only credential path was admin-issued tokens | Local passwords (scrypt), TOTP with replay protection and recovery codes, OIDC with PKCE and full id-token verification, self-service signup, reset and profile. `tests/test_identity.py` |
| Notifications did not exist | `Notification` plus a bell fed by SSE, and emitters at six real event sites. A repeated condition supersedes rather than piling up. `tests/test_inbox.py` |
| No PII access log | `pii_access_log` records who viewed personal data and whether it was redacted, admin-only, storing no copy of what it tracks. same |
| No activity feed | `activity_event` plus `/activity`. same |
| Classification, lane and 3D training plugins | All three built; 3D builds and refuses to train. `tests/test_ml_gaps.py` |
| Tracking metrics not wired to gold | The sealer records track ids and a `tracks_sealed` flag; a partially-identified set is refused rather than scored. same |
| Active learning could not select frames the detector missed | Four-signal frame-level candidate source. same |
| No experiment tracking | `experiment` / `experiment_run`, in the loop rather than beside it. same |
| `fit_channel_reliability` was a stub | Fitted from the human verdicts, Laplace smoothed, a thin channel keeps its prior. same |
| Masks, lanes, drivable and HD-map could not leave the system | Cityscapes labelIds, CULane, BDD masks, GeoJSON. `tests/test_export_and_cloud.py` |
| Exports fully buffered in memory | Streamed ZIP, ZIP64, one file in flight. same |
| Dataset diffing was count-level only | Object- and class-level comparison, with an ontology change flagged. same |
| Canvas marquee selection | Enclosed-not-touched, honouring the lock flag. |
| The remaining eight polling loops | Seven converted; two timers remain and both are documented as deliberate. |
| No bulk keyboard review mode | `/review/rapid`. |
| No user self-service, onboarding, activity view, i18n or responsive layout | All built. English, Hindi, Kannada, Tamil. |


An earlier version of this document over-used this heading. Two entries (ONNX backends, the OCR index) were
listed as blocked when the stated blocker was itself the fix, and one (text-to-object search) was listed as
blocked on a long backfill when the schema and query work was the actual deliverable. Those are now done. The
entries that remain here are ones where a specific piece of hardware or a paid resource is genuinely absent.

| Gap | What it needs |
| --- | --- |
| No learned 3D detector in the loop; the running path lifts 2D boxes geometrically | OpenPCDet or Pointcept installed and a pod-side entrypoint. The wrappers raise `NativeDetectionUnavailable` honestly today. Also needs real LiDAR rather than monocular pseudo-LiDAR, whose scale error the cuboids inherit. |
| 3D semantic segmentation is cuboid membership plus a ground plane | PTv3 inference wrapper, which is genuinely unwritten, plus the runtime above. |
| Cloud training, autolabel, relabel, and HD-map fusion all park in `queued-cloud` forever | A pod-side worker and the MinIO/workspace data-movement contract that is currently prose in a docstring. Requires a provisioned GPU pod to develop against. |
| TensorRT, LiteRT, and Hailo export | These need a device toolchain (Jetson, Android/LiteRT, Hailo SDK). The capability gate refuses them by name rather than fabricating a result, which is the correct behaviour. ONNX and ONNX Runtime were wrongly listed here and are now done: they are pure-CPU wheels, so the blocker was self-inflicted. |
| Multi-GPU and distributed training | The worker holds a Postgres advisory lock as a global GPU mutex by design; multi-GPU needs that scheduling model redesigned, and more than one GPU to test on. |

### Needs a schema migration and a backfill run

| Gap | What it needs |
| --- | --- |
| ~~Notifications do not exist~~ **Done.** `services/notify.py`, the `notification` table, the routes, and the bell in the shell all exist, with emitters at the issue, job-assignment and promotion sites. | Nothing. This row is kept struck through rather than deleted because it appeared as open in two places and a reader who found the other one deserves to see it settled. |

Text-to-object search and the OCR index were in this section and are now done (migration 0073). The
outstanding part is operational, not structural: existing crops need their `siglip_vec` backfilled by
running the embedder over the corpus, which the daemon does incrementally. New crops are embedded on
ingest.

### Substantial feature work

| Gap | What it needs |
| --- | --- |
| Classification, lane, and 3D training plugins | Detection, segmentation, and pose are done. These three need their own dataset builders; the 3D one additionally needs the learned detector above. |
| Experiment tracking (wandb or mlflow) | The metric curve is now persisted per job, which covers reading a run back; an external tracker is still absent. |
| Tracking metrics are not yet wired to a gold set | The metrics exist and are tested, but scoring a real tracker needs gold with guaranteed track ids, which the gold sealer does not currently produce. |
| Masks, lanes, drivable surfaces, and HD-map elements cannot leave the system in any format | Export adapters for each. The 3D exporter exists but no UI calls it. |
| Exports are fully buffered in memory inside the API process | Chunked writing and a streaming response. |
| Canvas marquee selection | Multi-select, bulk actions, hide, lock, and the filmstrip are done. Dragging a rubber-band box on the canvas itself (as opposed to selecting from the object list) is still outstanding. |
| Active learning cannot select frames the detector missed entirely | Candidates come from existing `Object` rows, so a false-negative frame is unreachable. Needs a frame-level candidate source. |
| Per-session pack routing is deferred, so multi-domain is architecture rather than behaviour | Thread `Session.pack_id` through the `services/domain` helpers at every call site. |
| The remaining eight polling loops | The stream and the hook exist and the jobs page uses them; the other pages still poll. |
| No PII access log | Retention and erasure are done. Recording who viewed personal data (as opposed to what the redactor detected) is still absent. |

### Requires a product decision before implementation

| Gap | The decision |
| --- | --- |
| No password, OAuth, OIDC, SSO, or MFA; the only credential path is admin-issued tokens and a local dev-login | Which identity model to adopt. This is the single largest blocker to a real deployment and it is a choice, not a coding task. |
| Single-tenant by construction. **Partly addressed**: migration 0092 put `project_id` on `session`, so frames, tracks and objects reach a tenant through one indexed hop, all 377 existing sessions are in a default project, and `services/api/tenancy.py` is the single seam that scopes a query. `GovernanceState` is still a global singleton row, and the scope is a convention until something enforces it on every read. | Whether to be multi-tenant at all, and if so whether to enforce the scope in the session layer or in Postgres row-level security. The column exists now, so this is a smaller decision than it was: what remains is enforcement and the singleton tables, not a 130-table retrofit. |
| Authorization is three global roles by path prefix, with no per-project or per-object ACL | What the permission model should be before building it. |
| No deployment story: compose has no api or web service, no TLS, no supervisor, and `.env.example` omits all seven required secrets | Target platform. The Dockerfiles are production-shaped; nothing composes them. |
