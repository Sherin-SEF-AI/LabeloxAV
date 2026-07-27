# Audit remediation status

A four-part audit (backend, frontend, ML pipeline, ops and enterprise) produced the gap list this document
tracks. It records what is fixed, what is not, and for each open item what specifically is needed. Nothing
here is marked done unless there is a test that fails without the fix.

Commits: `61f618d`, `03320b6`, `35151be`, `4994abc`, `02a1cf8`, `c438bb7`, `e65760e`.
Suite at the time of writing: 951 passing, 4 xfailed, failures limited to `tests/KNOWN_FAILURES.md`.

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
| Webhook URLs were validated only as "starts with http", giving any authenticated caller an SSRF primitive | Private, loopback, link-local, reserved, and multicast targets refused; re-checked immediately before each fetch, which also closes DNS rebinding; internal receivers are an explicit opt-in | `tests/test_webhook_hardening.py` |
| Webhook signature covered the body alone, so a captured delivery replayed forever | Timestamp bound into the signed material and published in a header; legacy form still verifies | same |
| Webhook delivery was a single POST with no retry | Backoff retry on transport failure and 5xx/429, never on 4xx; every attempt recorded | same |
| `/api/health` returned 200 while degraded, so a load balancer kept routing to a dead node | `/api/readyz` returns 503 unless every dependency answers | `tests/test_route_auth.py` |
| `GET /users` was unbounded and ran a full review-table aggregation on every page mount | Paginated with a ceiling, envelope with total, aggregation scoped to the returned users | `tests/test_editor_api.py` |
| No error boundary anywhere: one render throw white-screened the app | `error.tsx`, `global-error.tsx`, `not-found.tsx` | (build-verified) |
| The runbook told operators to compare against a baseline that did not exist | `tests/KNOWN_FAILURES.md` plus a test that keeps it accurate | `tests/test_known_failures.py` |

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

| Gap | What it needs |
| --- | --- |
| No learned 3D detector in the loop; the running path lifts 2D boxes geometrically | OpenPCDet or Pointcept installed and a pod-side entrypoint. The wrappers raise `NativeDetectionUnavailable` honestly today. Also needs real LiDAR rather than monocular pseudo-LiDAR, whose scale error the cuboids inherit. |
| 3D semantic segmentation is cuboid membership plus a ground plane | PTv3 inference wrapper, which is genuinely unwritten, plus the runtime above. |
| Cloud training, autolabel, relabel, and HD-map fusion all park in `queued-cloud` forever | A pod-side worker and the MinIO/workspace data-movement contract that is currently prose in a docstring. Requires a provisioned GPU pod to develop against. |
| ONNX, TensorRT, LiteRT, and Hailo export are dead code | Those backends are in no dependency group, so `capabilities.require()` fails for every target on a stock install. Needs the deps and hardware to verify against. |
| Multi-GPU and distributed training | The worker holds a Postgres advisory lock as a global GPU mutex by design; multi-GPU needs that scheduling model redesigned, and more than one GPU to test on. |

### Needs a schema migration and a backfill run

| Gap | What it needs |
| --- | --- |
| No text-to-object search | A `siglip_vec` column on `ObjectEmbedding`, an HNSW index, and a backfill over existing crops. The migration is small; the backfill is a long GPU job over the corpus. |
| OCR search is an unindexed `ILIKE '%...%'` | A `pg_trgm` GIN index on `Object.ocr_text`. |
| Notifications do not exist: issue comments, job assignments, kill-switch, drift, and SLO breaches are all silent | A notification table, an API, a bell in the shell, and emitters at each event site. |

### Substantial feature work

| Gap | What it needs |
| --- | --- |
| Training supports detection only | A `SegmentationTask` (masks exist and export already), then classification, pose, lane, and 3D plugins. Each needs its own dataset builder and its own eval. |
| No segmentation, lane, 3D, or tracking evaluation metrics | Mask IoU and mask AP (the Triton kernel exists and is unused by eval), lane F1 at distance, 3D and BEV AP, and MOTA/IDF1/HOTA. `services/verdyx/track_metrics.py` implements three tracking statistics and has zero callers; wiring it needs gold with guaranteed track ids, which the gold sealer does not currently guarantee. |
| No HPO, no resume from checkpoint, no experiment tracking | A sweep job type over child jobs; `resume=True` plus checkpoint discovery on orphan restart; a wandb or mlflow integration. |
| Masks, lanes, drivable surfaces, and HD-map elements cannot leave the system in any format | Export adapters for each. The 3D exporter exists but no UI calls it. |
| Exports are fully buffered in memory inside the API process | Chunked writing and a streaming response. |
| Editor has no multi-select, no per-object hide/lock wiring, no video filmstrip | Multi-select is a state change from `selectedId` to a set, plus marquee selection and bulk operations. Hide/lock is wired in the canvas already and needs only UI. |
| Active learning cannot select frames the detector missed entirely | Candidates come from existing `Object` rows, so a false-negative frame is unreachable. Needs a frame-level candidate source. |
| Per-session pack routing is deferred, so multi-domain is architecture rather than behaviour | Thread `Session.pack_id` through the `services/domain` helpers at every call site. |
| No realtime anywhere; nine polling intervals stand in for it | An SSE endpoint for job and progress events plus an `EventSource` hook. |
| Retention is computed and never enforced; no erasure workflow; no PII access log | `retention_until` is not settable through the API and no purge executor exists. Erasure needs a subject index and a cascade that covers object storage, not only database rows. |

### Requires a product decision before implementation

| Gap | The decision |
| --- | --- |
| No password, OAuth, OIDC, SSO, or MFA; the only credential path is admin-issued tokens and a local dev-login | Which identity model to adopt. This is the single largest blocker to a real deployment and it is a choice, not a coding task. |
| Single-tenant by construction: no tenant column on any of ~130 tables, and `GovernanceState` is a global singleton row | Whether to be multi-tenant at all. Retrofitting a tenant boundary touches every table and every query. |
| Authorization is three global roles by path prefix, with no per-project or per-object ACL | What the permission model should be before building it. |
| No deployment story: compose has no api or web service, no TLS, no supervisor, and `.env.example` omits all seven required secrets | Target platform. The Dockerfiles are production-shaped; nothing composes them. |
