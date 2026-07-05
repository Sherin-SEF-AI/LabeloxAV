# LabeloxAV Data Engine, M0 proposal: module tree + kernel schema

Status: proposal for review. No implementation code until approved (M0 is a structural gate).

## 0. The operating decision

LabeloxAV is the host platform. The six subsystems (SANYX, CALYX, SIEVYX, ORACLYX, VERDYX, FORGYX)
plus the existing annotation core become seven **Platforms**: distinct, navigable UIs behind a platform
launcher, all sharing one backend spine, one kernel, one storage layer, and one design system.

This is adopt-in-place, not the greenfield `labeloxav/kernel/ + planes/` tree. Reconnaissance of the repo
found that 5.5 of the 6 planes and 12 of the 14 spine entities already exist. The existing `core/` + `db/`
already function as the kernel; the existing `services/<domain>` groups already function as the plane
backends; the FastAPI gateway already mounts every plane's router. Rebuilding that as a fresh monorepo would
orphan a live platform. The section 5 layout in the spec is honored as a **logical** map, realized over the
existing tree via a platform registry, not a physical relocation.

Net-new work is concentrated, not spread: FORGYX (the only large greenfield plane), SANYX formalization,
`core/geometry.py` consolidation, three additive tables (migration 0049), the platform UI shell, and the
flywheel DAG (M9). Everything else is wire + verify + surface.

## 1. Kernel: spec layout mapped to the existing repo

| spec `kernel/` | realized as | status |
|---|---|---|
| `models/` | `db/models.py` (68 tables, the spine) + Pydantic in `core/schemas` | EXISTS |
| `mcap_io/` | `services/ingest/reader_mcap.py`, `services/lidar/ingest/mcap_pc.py`, `web/lib/inspector/mcap.ts` | EXISTS |
| `geometry/` | SE(3)/reprojection in `services/lidar/aggregate/register.py`, `services/calibration/extrinsics_check.py` | EXTEND: consolidate into `core/geometry.py`, add explicit SLERP + back-projection with tests |
| `ontology/` | `services/autolabel/ontology.py`, `ontology/*.yaml`, `OntologyVersion`/`OntologyClass` | EXISTS |
| `storage/` | `core/storage.py` (MinIO, content hashing, presign) | EXISTS |
| `db/` | `db/session.py`, `db/migrations/` (at 0048) | EXISTS |
| `jobs/` | `services/training/jobs.py` + `worker.py` (in-process asyncio; `*_job` tables) | EXTEND: wrap behind a `core/jobs` interface; keep asyncio backend, leave a Celery/Ray adapter seam |
| `vectorindex/` | `core/embeddings.py` (pgvector + HNSW) | EXTEND: put behind a swappable interface; keep pgvector, leave a Qdrant adapter seam |
| `telemetry/` | `core/logging.py` (structlog) | EXTEND: add a metrics hook |
| `config/` | `core/config.py` (Pydantic v2, `LBX_` env) | EXISTS |

## 2. Concrete module tree (logical platforms over the existing tree)

```
core/                         KERNEL. config, storage, embeddings(vectorindex), logging(telemetry)
  geometry.py                 NEW. SE(3), SLERP, multi-cam back-projection, epipolar (consolidated + tested)
  jobs/                       NEW (thin). queue interface over the existing asyncio worker; Celery/Ray seam
db/                           KERNEL. models.py (spine) + migrations (0049 adds benchmark/deployment/evaluation)
services/                     PLATFORM BACKENDS (existing, grouped by platform via the registry)
  ingest/ inspector/ adverse/            -> SANYX   + NEW services/sanyx/ (dropped-frame detector, quarantine gate)
  calibration/ lidar/calib/              -> CALYX
  intelligence/ curation/ activelearn/   -> SIEVYX
  autolabel/ multicam/ temporal/ recall/ -> LABELOX (core)
  lidar/ hdmap/                          -> ORACLYX + EXTEND (explicit consensus flag, distillation export)
  training/ govern/                      -> VERDYX  + EXTEND (slice as first-class, explicit evaluation rows)
  export/                                -> FORGYX  + NEW services/forgyx/ (onnx, ptq/qat, trt/litert/hailo, benchmark)
  api/                        GATEWAY. routers mounted today under domain names; add platform-name aliases
platforms/registry.py         NEW. single source of truth: platform id, label, backend prefix, gates, state hook
orchestration/dag.py          NEW (M9). the flywheel gate DAG
web/
  app/(launcher)/             NEW. platform home: tiles per platform with live state
  app/labelox|sanyx|calyx|sievyx|oraclyx|verdyx|forgyx/   route groups (existing pages regrouped + NEW forgyx)
  app/sessions/[id]/          NEW. cross-platform session stitch (health, calib, tags, labels, pseudo-GT, eval, deploy)
  platforms/registry.ts       NEW. mirrors platforms/registry.py; drives launcher, switcher, session stitch
  components/shell/           EXTEND PageShell -> PlatformShell + PlatformSwitcher
```

## 3. Platform registry (the one new abstraction)

A declarative registry, mirrored backend (`platforms/registry.py`) and frontend (`web/platforms/registry.ts`),
is the spine of both the UI and the plane grouping. One entry per platform:

```
{ id: "sanyx", label: "SANYX", role: "ingest QA",
  backend_prefix: "/sanyx", legacy_prefixes: ["/inspector","/adverse"],
  routes: ["/sanyx","/sanyx/session/[id]","/sanyx/quarantine"],
  gate: "health",            // this platform can block downstream progression
  state_hook: sanyxSummary } // launcher tile live state (quarantine count, etc.)
```

The launcher, the left-rail switcher, the flywheel DAG, and lineage queries all read the registry, so the whole
system speaks "platform" without any code being physically moved.

## 4. Kernel data-model schema (the spine)

12 of 14 spine entities already exist. The proposal is **one additive migration, `0049`**, nothing rewritten.

### 4a. Existing entities (keep, minor extend)

| spine entity | existing table | note |
|---|---|---|
| Session | `session` | keep |
| Sample | `frame` | keep (clip = session + t range) |
| Annotation | `object`, `object_3d`, `review` | keep |
| Release | `dataset_commit`, `gold_set` | keep (content-hashed + parent lineage already) |
| Model | `model_run`, `model_registry` | keep |
| HealthReport | `session_health` | EXTEND: per-check sub-scores + evidence uris + {pass,degraded,quarantine} |
| CalibrationState | `calibration_validation` (+ lidar) | EXTEND: SE(3) drift delta per pair + rig drift-history view |
| Embedding | `frame_embedding`, `object_embedding` | keep |
| ScenarioTags | `frame.scene` JSONB | keep informal; VERDYX reads it directly |
| Slice | `curation_slice` | EXTEND: add `protected` bool + `kind` in {curation,eval}; reuse for VERDYX |

### 4b. Migration 0049 (new tables)

```
evaluation            # lift eval out of model_registry.gold_metrics into first-class rows
  eval_id (uuid pk), model_version (fk model_registry), release_commit (fk dataset_commit),
  gold_id (fk gold_set, null), per_slice (jsonb: slice_id -> {map,precision,recall,confusion}),
  failure_clusters (jsonb), aggregate (jsonb), verdict (str: promote|reject|needs_review),
  challenger_of (str, null), created_at

benchmark             # per (model, target) latency/accuracy/power - the true gap
  benchmark_id (uuid pk), model_version (fk), target (str: sentrixai_litert|agx_orin_trt|orin_nano_trt|pi_hailo),
  latency_ms (jsonb: {p50,p95,p99}), throughput_fps (float), power_w (float, null),
  accuracy_ref (uuid fk evaluation, null), per_layer_uri (text, null), pareto_rank (int, null),
  artifact_uri (text), created_at

deployment            # deployed/exported artifact per target, with lineage to release + verdict
  deployment_id (uuid pk), model_version (fk), target (str), artifact_uri (text), export_format (str),
  release_commit (fk dataset_commit), verdict_ref (uuid fk evaluation, null), benchmark_ref (uuid fk benchmark, null),
  status (str: built|verified|blocked|deployed|retired), notes (text), created_at

pseudo_label          # ORACLYX consensus, side table keyed to the fused object (no object rewrite)
  object_id (uuid pk, fk object), consensus (bool), consensus_score (float),
  voters (jsonb: path -> {agree,conf}), fusion_run_id (str, null), created_at
```

Rationale: `evaluation`, `benchmark`, `deployment` complete the Model side of the spine (VERDYX + FORGYX).
`pseudo_label` makes ORACLYX consensus explicit without touching the hot `object` table (it already carries
`source="fused"` + `provenance`). No existing column changes except the two EXTENDs in 4a, both additive.

## 5. Milestone effort, reframed against what exists

| M | plane | reality |
|---|---|---|
| M0 | kernel + scaffold | EXTEND: `core/geometry.py`, `core/jobs` iface, migration 0049, platform registry + launcher shell |
| M1 | SANYX | EXTEND: dropped-frame detector, GPS/IMU isolation, quarantine gate; UI regroup (health/adverse exist) |
| M2 | CALYX | mostly EXISTS: add SE(3) drift-delta + rig history view + reprojection overlays |
| M3 | SIEVYX | EXISTS: regroup UI (embeddings/nlsearch/dedup/activelearn all present) |
| M4 | Labelox core (gate) | wire existing autolabel/multicam/recall onto registry + queue; low code, high care |
| M5 | ORACLYX | EXTEND: consensus flag (pseudo_label) + distillation export; fusion exists (fuse3d/register/hdmap) |
| M6 | release | mostly EXISTS: `dataset_commit` is content-hashed + lineage; add reproducibility test + registry UI |
| M7 | VERDYX (gate) | EXISTS: slice-as-first-class + `evaluation` rows + slice matrix UI; champion gate present |
| M8 | FORGYX | NEW: the real build (onnx, ptq/qat, trt/litert/hailo, benchmark matrix, dual gate, Pareto UI) |
| M9 | flywheel | NEW: `orchestration/dag.py` + lineage query + cross-platform session view |

## 6. Stack + open items (recommendations)

- Jobs: keep in-process asyncio behind `core/jobs`; add Celery/Ray only if a real scale need appears (spec: "pragmatic beats enterprise-scale"). RECOMMEND keep.
- Vector index: keep pgvector behind `core/vectorindex`; Qdrant adapter is a seam, not now. RECOMMEND keep.
- Route prefixes: add platform-name aliases (`/sanyx`, `/calyx`, ...) alongside existing domain routers; non-breaking. RECOMMEND yes.
- CLAUDE.md: a repo CLAUDE.md already governs the build; fold this brief in additively, do not overwrite. RECOMMEND additive.

## 7. What M0 delivers once approved (the reviewable artifact set)

1. `db/migrations/0049_data_engine_spine.py` + the four new models in `db/models.py` (additive).
2. `core/geometry.py` with SE(3), SLERP, back-projection, epipolar + tests promoted from the calibration/lidar code.
3. `core/jobs/` and `core/vectorindex/` thin interfaces wrapping the current backends.
4. `platforms/registry.py` + `web/platforms/registry.ts` defining the seven platforms.
5. `web/app/(launcher)/` platform home + `PlatformShell`/`PlatformSwitcher` over the existing PageShell.
6. Backend platform-name route aliases mounted in `services/api`.
7. Acceptance: ingest one real CHRONYX MCAP session (path exists), see it in the launcher and the cross-platform
   session view, with SANYX/CALYX state surfaced. `make` target for the demo.

No plane logic is built in M0. It establishes the platform model + the spine additions only, so nothing that
works today changes behavior.
