# AV_ASSUMPTIONS - where LabeloxAV bakes in autonomous driving

SEC-M0 deliverable. A walk of every plane and organ *as implemented*, with the exact places each hardcodes an
autonomous-driving / moving-camera assumption. Evidence is cited as `file:line`. This document is ground
truth: anything in later milestones that conflicts with a finding here is resolved in favour of the finding.

Method: four parallel read-only audits over `services/*`, `core/*`, `db/models.py`, `core/config.py`, and
`ontology/labelox_in_v0.yaml`. No code was changed.

## TL;DR - the shape of the coupling

The engine is **more separable than it looks**. Three facts frame the whole refactor:

1. **The kernels are already domain-neutral.** The SE(3) projection kernel takes an arbitrary `T_cam_world`
   (`core/accel/projection.py:160`), confusion/slice aggregation is parameterised by `n_classes`
   (`core/accel/slices.py:13`), dataset paths are generic and content-addressed
   (`services/ingest/run.py:40`, `core/storage.py:75`), there is **no hardcoded road-scene pixel augmentation**
   to remove, and privacy is an engine-level fail-closed organ (`services/anonymize/`). None of these fights
   the refactor.

2. **The breakage surface is concentrated, not diffuse.** AV-specificity clusters in four buckets: (a)
   code-baked ontology facts, (b) the moving-camera / ego-motion math, (c) the safety/strata policy, (d)
   AV-shaped privacy targets. Each is a named, finite set of sites, listed below.

3. **The deepest assumption is in the schema, not a plane:** `session = a moving-vehicle drive keyed by
   `vehicle_id`, with a per-frame `ego_speed` (`db/models.py:74`, `db/models.py:105`). Every rig-level
   aggregation and every dynamics/calibration metric is downstream of this. It is a data-model decision, and
   it is upstream of all six planes.

The `SceneModel` fork the mission predicts (SEC-M2) is real and lands exactly at the calibration seam
(`services/calibration/resolve.py`). The ontology and prompt forks are shallow (config/string swaps). The
governance-safety fork is a one-literal-in-six-files consolidation. The privacy plane already exists and needs
pluggable targets, not a rewrite.

---

## 0. The data-model assumption (upstream of every plane)

| Assumption | Evidence |
|---|---|
| A session is a moving-vehicle drive, identified by a vehicle | `Session.vehicle_id` `db/models.py:74`; keyed everywhere: `services/sanyx/trends.py:66`, `services/calyx/run.py:43`, `services/calibration/report.py:61` |
| Every frame carries an ego speed (CAN-derived) | `Frame.ego_speed` `db/models.py:105`; consumed by dynamics `services/dynamics/compute.py:78`, autolabel `services/autolabel/runner.py:165` |
| A rig is one camera set per vehicle | `CameraRig` keyed on `vehicle_id` `services/calibration/report.py:61`; `RigSettings` `core/config.py:427` |

**Refactor implication.** A static-camera pack keys state by a stable `camera_id` (SEC-M2), not `vehicle_id`,
and has no ego speed. The engine already degrades when the moving-camera signals are absent (SANYX
renormalises with no inertial channels `services/sanyx/run.py:107`; CALYX has honest-null paths for "no road
lines / no overlapping cameras" `services/calibration/estimate.py:128`, `extrinsics_check.py:74`) - so the
schema fields become **optional, pack-populated** rather than removed. `Session`/`Frame` stay; `vehicle_id`
and `ego_speed` become nullable inputs a pack's `SceneModel` may or may not provide.

---

## 1. SANYX - ingest quality assurance (`services/sanyx/`)

**As implemented.** Six ingest-QA checks over a session → a 0-100 score + `{pass, degraded, quarantine}`
decision → a `SessionHealth` row → a quarantine gate downstream planes consult. Orchestrated by
`services/sanyx/run.py:run_health`; checks in `services/sanyx/checks.py`; scoring `services/sanyx/score.py`;
gate `services/sanyx/gate.py`; rig-trend maintenance `services/sanyx/trends.py`; on-vehicle subset
`services/sanyx/lite.py`.

**I/O.** In: `Frame`, `Session`, `SessionIndex.topics`, raw MCAP. Out: `SessionHealth` `db/models.py:1610`,
`SanyxRigAlert` `db/models.py:1788`. Config: `SanyxSettings` `core/config.py:665`. Routes:
`services/api/routers/sanyx.py`.

**AV assumptions.**
- Hardcoded rig-sensor limits as function defaults, not config: `check_imu(accel_limit=156.9, gyro_limit=34.9)`
  - Xsens MTi-3 full-scale - `services/sanyx/checks.py:167`; GNSS 2D-fix semantics `checks.py:145`.
- AV sensor-topic vocabulary + ROS NavSat decode: `services/sanyx/inertial.py:22` (`imu/mti/xsens`, `gnss/gps/
  ublox`, `pps/stm32`), `inertial.py:86`.
- AV fault vocabulary + remediation baked in code: `services/sanyx/rootcause.py:14` (`loose_gmsl2_connector`,
  `gps_urban_canyon_multipath`, `imu_thermal_drift`, "reseat the GMSL2 coax", "route timing").
- Rig components + `cam_ft` default + live-drive framing: `services/sanyx/trends.py:23`, `services/sanyx/lite.py:32`.

**Split.** *Reusable:* the image/timestamp checks (`check_dropped_frames`, `check_exposure`,
`check_lens_contamination` `checks.py:79`), scoring/decision fold `score.py`, gate `gate.py`, streaming
accumulator `stream.py`, trend-fit math `trends.py:30`. *AV-specific → pack:* the inertial check family
(`check_time_sync/imu/gps`) and `inertial.py`, the root-cause signature table, `lite.py`, `vehicle_id` keying.
A static-camera pack supplies the image checks + scoring/gate unchanged and swaps the signature table.

## 2. CALYX - calibration (`services/calyx/` + `services/calibration/`) - the deep AV fork

**As implemented.** `services/calyx/` monitors calibration drift over time and recovers it; `services/
calibration/` is the calibration seam (intrinsics/extrinsics resolve, validation, estimation, import). Drift is
an **SE(3) delta between visual odometry and IMU/GNSS dead-reckoning** - only meaningful for a moving camera.

**I/O.** In: `Frame`, `Object.rig_track_id`+`bbox`, `Calibration`. Out: `CameraCalibration` `db/models.py:917`,
`CalibrationValidation` `db/models.py:895`, `CalibrationOverride` `db/models.py:1808`. Core geometry:
`core/geometry.py` (SE(3), slerp). Config: `CalyxSettings` `core/config.py:705`, `RigSettings` `:427`,
`SpatialSettings` `:446`, `LensIntrinsics` `:417`. Routes: `services/api/routers/{calyx,calibration}.py`.

**AV assumptions.**
- Ego-motion is the premise of drift: `estimate_extrinsic_drift(cam_points vs inertial_points)`
  `services/calyx/drift.py:47`; `temporal_consistency` flags steps "vehicle dynamics" disallow `drift.py:66`.
- The ego-frame projection convention: `_R_OPT2EGO` and `Calibration.R() = Rz(-yaw)@Ry@Rx@_R_OPT2EGO`
  `services/calibration/resolve.py:22,49`; `nominal_calibration()` pulls per-camera mount yaw + mount height
  `resolve.py:65`. **This seam is the SceneModel boundary.**
- Camera-to-IMU / camera-to-GNSS extrinsics by motion residual `services/calibration/extrinsics.py:32` - needs
  a moving platform.
- Road-scene calibration: focal/pitch from lane lines + horizon `services/calyx/targetless.py:15`,
  `services/calibration/estimate.py:52` (drops the sky, keeps road-line VPs).
- AV rig/dataset config: STURDeCAM31 lens pair + 4-camera surround yaws `core/config.py:432`; STM32/IMU-rate
  time-sync `services/calibration/timesync.py:15`; KITTI/nuScenes importers `services/calibration/import_calib.py:25`.

**Split.** *Reusable:* the geometry kernels - `core/geometry.py`, `rigid_align`/Kabsch `drift.py:20`, Sampson
`epipolar_residual` `extrinsics.py:15`, confidence-weighted `fuse_calibrations` `consensus.py:11`,
`uncertainty.py`; the calibration store/precedence machinery `store.py:51`; generic intrinsics (ChArUco,
implied-HFOV) `intrinsics.py:19`. *AV-specific → pack:* the ego-frame projection convention (`_R_OPT2EGO`,
`Calibration.R()/nominal_calibration`), the ego-motion drift estimation, the road-scene estimators
(targetless, road-line VP), the ego-sensor extrinsics + STM32 time-sync, `RigSettings`/IPM `SpatialSettings`,
the AV-benchmark importers, and `vehicle_id` rig keying. A static-camera pack reimplements `Calibration.R()/t()`
and `nominal_calibration()` (or makes `_R_OPT2EGO`/yaw/height pack-provided) behind the `resolve.py` seam;
everything else stays.

## 3. SIEVYX - curation / mining (`services/sievyx/`)

**As implemented.** Pure embedding-space curation functions + a thin API: long-tail cluster discovery
(`discovery.py`), failure-cluster mining (`failure_mining.py`), core-set batch selection (`batch.py`),
label-queue composition (`composition.py`), ODD coverage gaps (`odd.py`), maneuver classification
(`maneuver.py`). No models/GPU; operates over embeddings/trajectories/counts handed in by the router.

**AV assumptions.** Concentrated in **one module**: `services/sievyx/maneuver.py:13` (`MANEUVERS` road taxonomy:
`cut_in, unprotected_turn, u_turn, lane_change, jaywalk, straight`), `maneuver.py:43` (road-driving angle/metre
thresholds in a BEV/ego frame). `odd.py:2` intends illumination×weather×road-type×density cells (the *code* is
dict-driven and generic; the *cell schema* is AV). Everything else - discovery, failure-mining, batch,
composition, ODD mechanics - carries **no drive-session assumption** and is cleanly reusable.

**Split.** *Reusable:* everything except `maneuver.py`. *AV-specific → pack:* the maneuver taxonomy + thresholds
(a Sec pack's "events" - perimeter_breach, tailgate - are the analogue), and the ODD cell dimensions (become
pack strata, see §6).

## 4. ORACLYX - offline pseudo-truth (`services/oraclyx/`)

**As implemented.** Offline consensus + pseudo-GT calibration. `consensus.vote()` fuses the ORACLYX detection
against the three auto-label paths, auto-accepts on agreement, routes disagreement to humans; `run.py`
persists to `PseudoLabel` and sets `Object.state`; `uncertainty.py` ranks the disagreement queue;
`tracks4d.py` stitches consensus into 4D tracks; `mono_depth.py`/`radar.py` add depth/velocity priors.

**I/O.** Out: `PseudoLabel` `db/models.py:1770`. No dedicated config block - tolerances are function defaults
(`consensus.vote(iou_tol=0.5, min_agree=3)` `consensus.py:26`). Depth checkpoint pinned to an outdoor-driving
model `core/config.py:572`. Routes: `services/api/routers/oraclyx.py`.

**AV assumptions.**
- **Latent bug + AV coupling:** `mono_depth.CLASS_HEIGHT_M` `services/oraclyx/mono_depth.py:15` maps class-ids
  0-8 to real-world heights - and the ids **do not match the governed ontology** (this map's `0:pedestrian,
  4:car` vs the YAML's `1:motorcycle, 11:sedan`). Wrong for AV today; a landmine for any pack.
- Ground-plane depth prior assumes a forward camera above a road plane `mono_depth.py:28`; radar fusion assumes
  ego-motion in km/h `radar.py:37`.

**Split.** *Reusable:* `consensus.vote()`, `uncertainty.py`, `tracks4d.py`, `run.export_distillation()` - all
domain-agnostic (opaque `{path, class_id, bbox, conf}` dicts). *AV-specific → pack:* `mono_depth`, `radar`, the
outdoor depth model. ORACLYX also needs a config block (tolerances are currently un-externalised defaults).

## 5. The auto-label orchestration (`services/autolabel/`)

**As implemented.** `StagedRunner`/`autolabel_session` owns the 16 GB GPU lifecycle: Stage 1 = YOLO (Path A) +
open-vocab YOLO-World/SAM (Path B) co-resident; Stage 2 = duty-cycled VLM (Path C) over the uncertain subset.
Per-frame: fuse → drop stuff → ego-hood drop → quality-review → gate → VLM under budget → re-gate → persist.
Fusion (`fusion.py`) clusters/votes/reconciles/dedupes; gate (`gate.py`) routes by calibrated confidence.

**AV assumptions - five swappable surfaces.**
1. **Ontology artifact + the in-code stuff set.** `STUFF_NAMES`/`STUFF_L0` panoptic split baked as frozensets
   `services/autolabel/ontology.py:31` (should live in the YAML); `supported_core` 40-name allow-list
   `core/config.py:396`.
2. **The VLM prompt.** Hardcoded *"You are labeling an object cropped from an Indian road scene for an
   autonomous-driving dataset"* `services/autolabel/paths/path_c_vlm.py:64` - no config seam today.
3. **Anchor/synonym/map constants.** `CROSS_ANCHORS` (23 India road actors) `path_c_qwen3vl.py:186`,
   `COCO_TO_ONTOLOGY` `path_a_yolo26.py:21`, `_OPENVOCAB_SYNONYMS` `path_b_sam3.py:29`.
4. **Gate policy.** `_SAFETY_L1 = {"vru","animal"}` `gate.py:20`; `is_rare = india or l1=="fallback"` `gate.py:23`.
5. **Quality-reviewer road-plane rules.** `_GROUND/_VEHICLE/_VRU/_OVERHEAD` sets + horizon/containment checks
   `services/autolabel/quality_reviewer.py:20`; per-superclass `size_bounds` `core/config.py:242`.

Plus the **ego-hood mask** in the loop (`services/autolabel/runner.py:317`, "the car labeling its own bonnet")
- a moving-vehicle assumption that a pack must disable (already a safe no-op when `vehicle_id`/mask absent).

**Split.** *Reusable engine plumbing:* `runner.py` staging + VRAM guard, `fusion.py` clustering/voting/dedupe,
`gate.py` state machine, `calibrate.py`/`isotonic.py`, `paths/base.py` `RawDetection` contract, and the Path
B/C load/infer/vote machinery. Class handling is **dict/name-based and class-count-agnostic** throughout
(`fusion._vote` builds a dynamic weights dict). *AV-specific → the five surfaces above.* This is the SEC-M5
target: reuse the plumbing under a Sec `AutoLabelProfile` that supplies those five.

## 6. VERDYX + FORGYX + governance (eval, edge deploy, promotion)

**VERDYX** (`services/verdyx/`). Per-slice eval + champion-challenger slice verdict + safety-weighted recall.
`verdict.py`/`stats.py`/`shadow.py` are domain-agnostic (opaque slice-id strings). AV: `CRITICAL_CLASSES =
{0,1,2,3,8}` fixed ids `services/verdyx/safety_recall.py:10`; TTC/near-miss weighting `safety_recall.py:22`
(pure collision constructs); default `protected_slices = ["pedestrian_night","autorickshaw_glare"]`
`run.py:20`. **Latent bug:** `run.py:20` reads `GovernSettings.protected_slices`, which **does not exist** in
`core/config.py` - so the AV fallback is *always* used.

**FORGYX** (`services/forgyx/`). Capability-gated edge export/quantize/compile + dual (latency+accuracy) gate +
signed rollout. `gate.py`/`packaging.py` are fully generic (opaque `target` strings). All AV coupling is in
four **target registries**: `_BACKENDS` (agx_orin, orin_nano, sentrixai, pi_hailo) `capabilities.py:18`,
`TARGET_THERMAL` `thermal.py:11`, `TARGET_BUDGET_MS` `cooptimize.py:11`, `_FORMAT` `run.py:19`, plus Jetson
`tegrastats` parsing `benchmark.py:44`. CCTV targets (NVR SoCs, Ambarella, Axis ARTPEC, x86) just slot in as a
per-pack hardware profile.

**Governance** (`services/govern/champion.py` + `services/recall/gate.py` + `services/training/eval.py`). The
`champion_gate` control flow (beats-mAP ∧ Safe-mIoU-not-regressed ∧ safety-recall floor ∧ no-regress, fail-
closed) is **reusable and pack-agnostic**. The AV parts are the *predicates* it calls:
- **`{"vru","animal"}` is the safety definition, copy-pasted in six+ files:** `champion.py:26`,
  `recall/gate.py:12`, `training/safe_miou.py:13`, `autolabel/gate.py:20`, `dynamics/compute.py:33`,
  `flywheel/signals.py:26`. Consolidate to one pack-scoped `safety_l1` predicate.
- **`get_ontology()` is a single global India-AV ontology**, `@lru_cache(maxsize=1)`
  `services/autolabel/ontology.py:225` - one taxonomy per process. Governance (and everything) needs a pack
  dimension on this.
- **Safe-mIoU affinity cost tree** is AV road semantics (`pedestrian-vs-pole` = max cost)
  `services/training/safe_miou.py:21`; a Sec pack needs its own affinity.
- **Strata dimensions** `("weather","time_of_day","road_type","density")` hardcoded in
  `services/curation/slices.py:15`, `services/explore/facets.py:28`, consumed by SIEVYX ODD.

**Split.** *Reusable:* champion-gate flow, recall-gate structure, VERDYX verdict/stats/shadow, FORGYX gate/
packaging/rollout, `gold_eval` common-gold re-scoring, `gold.py` sealing mechanism, all JSONB DB models.
*Per-pack:* the safety-class predicate, the ontology, the Safe-mIoU affinity, the strata dimensions, the
critical-class list + TTC, the FORGYX target registries, the gold-set selection defaults.

---

## 7. Cross-cutting facts (good news for the refactor)

- **Dataset paths are generic + content-addressed.** `frames/{session}/{cam}/{ts}.jpg`
  `services/ingest/run.py:40`; exports `datasets/{name}/{commit}` `services/export/dataset.py:219`; training
  `scratch/training/{name}` `dataset_builder.py:115`; content-addressed store `core/storage.py:75`. Only
  external-anchor helpers are AV-named (`scripts/idd_to_yolo.py`, `map_region="bangalore"` `core/config.py:454`)
  - all in defaults/comments. **Keep as-is.**
- **Tensor shapes are runtime-derived** from `len(names)`/`n_classes` (`core/accel/slices.py:13`,
  `training/eval.py:82`, `dataset_builder.py:132`) - already pack-adaptive. The only fixed-integer couplings
  are the two latent bugs (`CRITICAL_CLASSES {0,1,2,3,8}`, `CLASS_HEIGHT_M` ids 0-8) and the single-ontology
  singleton.
- **No road-scene pixel augmentation to remove.** Training passes no augmentation kwargs → Ultralytics defaults
  (`services/training/tasks/detection.py:79`, `finetune.py:26`). The AV coupling is in *sampling* (class-balance
  caps `dataset_builder.py:35`, `cities=["BLR"]` scoping) - a non-road pack may want to disable `fliplr` for
  text/plate work, so augmentation becomes a pack-overridable set.
- **Privacy is already an engine-level, fail-closed, two-gate organ.** Ingest Gate A blurs faces/plates **in
  place before JPEG encode** (`services/ingest/run.py:120`, "no clean frame reaches storage") and writes a
  `PiiAudit` row; the export gate refuses any clip with un-redacted face/plate/speech
  (`services/export/dataset.py:168`, `services/anonymize/compliance.py:34`). Fail-loud if a required detector is
  missing (`anonymizer.py:48`); fail-closed if an audit row is absent. **AV-typed** (face/plate/speech, DPDPA)
  and speech detection is still an unwired seam (`services/anonymize/speech.py:14`). SEC-M6 formalises this as a
  `PrivacyPlane` interface with pluggable redaction targets + legal regimes - the mechanism and chokepoints are
  already right.

---

## 8. Latent bugs the audit surfaced (fix during the refactor, not before)

All four are now **resolved**. They are kept here with their fixes rather than deleted, because the list was
the record of what the audit found and a reader who arrives at it deserves to see how each one ended.

1. ~~**`CLASS_HEIGHT_M` ids don't match the ontology**~~ - **Fixed 2026-08-18.** The table is keyed by name
   (`CLASS_HEIGHT_M_BY_NAME`) and resolved through the ontology at call time. The damage was worse than the
   entry recorded: id 0 was dead, `pedestrian`/`rider`/`cattle` fell outside the table and silently returned
   `None` so the size prior was off for every VRU, and id 5 - commented "truck", 3.2 m - is actually
   `delivery_rider_bike`. Four tests in `test_oraclyx_m14.py`, including one that pins every name resolving.
   This one outlived #2-#4 specifically because its test asserted the table against itself.
2. ~~**`CRITICAL_CLASSES = {0,1,2,3,8}` is 0-based**~~ - Fixed; resolved by name via
   `services/domain.py:critical_class_ids`.
3. ~~**`GovernSettings.protected_slices` doesn't exist**~~ - Fixed; `core/config.py` carries it and
   `services/verdyx/run.py` reads it.
4. ~~**`{"vru","animal"}` is triplicated+**~~ - Fixed; `services/domain.py` is the single seam.

The SEC-M0 rule said these get resolved when the relevant surface moves behind the pack interface. #2-#4
went that way. #1 did not need to: keying the table by name fixes the ontology mismatch without adding a
pack surface, and adding one would have changed the frozen pack digest for a table no other domain has a
use for yet. If a second domain ever needs metric size priors, this is the natural thing to move.

---

## 9. Deviations from the mission's assumptions

- The mission names `services/govern/gate.py`; the gate logic actually lives in `services/govern/champion.py`
  fed by `services/recall/gate.py` + `services/training/eval.py`. Resolved in favour of the code.
- The mission's `PrivacyPlane | None` (None for AV) is slightly off: AV *does* run a privacy organ today
  (`services/anonymize/`) - it's not None. The refactor should make AV's privacy plane a real (face/plate)
  instance and Sec's a superset (adds retention/consent/rights), rather than treating AV as None.
- Strata are assumed configurable; today they're hardcoded in three modules (§6). The interface must add a
  `strata_dimensions` surface, not just consume an existing one.
</content>
