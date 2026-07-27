# PACK_INTERFACE - the DomainPack contract

SEC-M0 deliverable. A proposed `DomainPack` Protocol fitted to the findings in
[AV_ASSUMPTIONS.md](AV_ASSUMPTIONS.md). This is a design document, not code; it is the contract SEC-M1+ builds
against. Every surface below exists because a specific AV assumption was found baked into engine code, and the
surface is where that assumption becomes pack-supplied data.

## Design rules (from what the audit found)

1. **The engine core imports nothing from packs.** Enforced in CI by import-linter (SEC-M1). The core keeps
   the kernels the audit found already neutral (SE(3), slice math, storage, fusion state machine, gate control
   flow) and calls into a pack only through the Protocol below.
2. **A pack is data + a small amount of code.** The bulk of AV-specificity the audit found is *data wearing a
   code costume*: `STUFF_NAMES` frozensets, the `{"vru","animal"}` literal, `CROSS_ANCHORS`, `supported_core`,
   the target registries. Those move to pack config (`pack.yaml` + governed ontology YAML). Only the genuinely
   behavioural forks (the `SceneModel`, the privacy detectors) are pack *code*.
3. **Reuse the existing config system.** Pack settings are a namespaced extension of `core/config.py`
   (`get_settings().pack(<id>)`), not a parallel mechanism. AV's current defaults (`OntologySettings`,
   `RigSettings`, `SpatialSettings`, `SanyxSettings`, ...) become the AV pack's config block verbatim, proving
   parity.
4. **AV moves behind the interface unchanged.** The AV pack is authored first and must reproduce every current
   metric (the golden regression harness, SEC-M1, is the gate). If a surface can't express what AV does today,
   the surface is wrong.

## The Protocol

```python
# packs/base.py  (engine core; a pack implements this)
from typing import Protocol, runtime_checkable

@runtime_checkable
class DomainPack(Protocol):
    manifest: PackManifest                     # identity, version, capabilities, feature flags
    ontology: OntologySpec                      # the taxonomy + the facts currently baked in code
    scene_model: SceneModelFactory              # moving-camera vs static-camera pose/geometry source
    autolabel_profile: AutoLabelProfile         # prompt, anchors, maps, gate policy for the 3 paths
    safety_policy: SafetyPolicy                  # replaces the copy-pasted {"vru","animal"}
    eval_strata: EvalStrataSpec                  # slice dimensions + critical classes + protected slices
    ingestion_adapters: list[IngestionAdapter]  # how raw captures become Session/Frame
    quality_profile: QualityProfile              # SANYX checks + rootcause table for this domain
    privacy_plane: PrivacyPlane                  # redaction targets + legal regime (never None; see note)
    forge_targets: list[ForgeTarget]            # edge hardware profiles (replaces the AV silicon registries)
```

The names track repo conventions: `*Settings`/`*Spec` for config-shaped data, `*Factory` where the pack
returns a constructed object, `*Profile` for a bundle of policy. Each is detailed below with the exact finding
it dissolves.

---

### `manifest: PackManifest`

Pack identity and the on/off switches for optional planes.

```python
@dataclass(frozen=True)
class PackManifest:
    id: str                       # "av", "sec"
    version: str                  # semver; travels into ModelRun.provenance
    display_name: str
    capabilities: frozenset[str]  # {"moving_camera","dynamics","depth","inertial_qa","privacy_export",...}
    default_scene_model: str      # "moving_camera" | "static_camera"
```

`capabilities` is the honest feature gate the audit kept bumping into: the engine already degrades gracefully
when a signal is absent (SANYX with no inertial channels, CALYX with no overlapping cameras). `capabilities`
makes that explicit instead of implicit, so a plane can skip cleanly (`"dynamics" not in pack.capabilities` for
Sec) rather than running IPM math on a static camera and producing garbage.

**Dissolves:** the implicit "every pack has ego-motion" premise behind ORACLYX depth, SIEVYX maneuver, and
`services/dynamics/compute.py`.

### `ontology: OntologySpec`

The single biggest finding: ontology *facts* live in code, and there is one global ontology per process.

```python
@dataclass(frozen=True)
class OntologySpec:
    yaml_path: str                        # the governed, versioned taxonomy (already exists)
    stuff_names: frozenset[str]           # was services/autolabel/ontology.py:31 STUFF_NAMES
    supported_core: frozenset[str]        # was core/config.py:396 OntologySettings.supported_core
    superclass_map: dict[str, str]        # l1/l0 semantics used by quality_reviewer road-plane rules
    size_priors: dict[str, SizeBound]     # was core/config.py:242 size_bounds
    fallback_policy: FallbackPolicy        # CUSTOM_ID_BASE, fallback_ids, is_fallback (ontology.py:22,108)
```

Two structural fixes ride along:

- **`get_ontology()` gains a pack dimension.** The `@lru_cache(maxsize=1)` singleton
  (`services/autolabel/ontology.py:225`) becomes `get_ontology(pack_id)` (cache keyed by pack + version). This
  is the one change that touches many call sites (the finding lists `finetune.py:54`, `dataset_builder.py:56`,
  and every plane) so it is done first under SEC-M1 with the golden harness proving AV is byte-identical.
- **`stuff_names` moves from a code frozenset into the YAML** (a `role: thing|stuff` per class), so the
  panoptic split is governed data. `superclass_map` externalises the `_GROUND/_VEHICLE/_VRU/_OVERHEAD` sets the
  quality reviewer keys on.

**Dissolves:** `ontology.py:31` STUFF_NAMES, `core/config.py:396` supported_core, `quality_reviewer.py:20`
road-plane l1 sets, the single-ontology singleton. **Also fixes latent bug** #1/#2 (the id-based
`CLASS_HEIGHT_M` and `CRITICAL_CLASSES` become name-based lookups through the spec, so they can't drift from
the YAML).

### `scene_model: SceneModelFactory` - the deep fork (SEC-M2)

The audit located the moving-vs-static boundary precisely at `services/calibration/resolve.py`. This factory is
that boundary.

```python
class SceneModelFactory(Protocol):
    def build(self, session: Session, calib: Calibration) -> SceneModel: ...

class SceneModel(Protocol):
    def camera_pose(self, frame: Frame) -> SE3: ...        # world<-camera for this frame
    def R(self) -> np.ndarray: ...                          # camera rotation convention
    def ground_plane(self) -> Plane | None: ...            # for depth/IPM priors; None if not applicable
    def is_static(self) -> bool: ...
```

- `MovingCameraSceneModel` (AV) wraps today's `Calibration.R() = Rz(-yaw)@Ry@Rx@_R_OPT2EGO`
  (`resolve.py:49`), the SLERP ego-pose composition (`core/accel/propagate.py`), and the road ground plane. It
  reads `Session.vehicle_id` / `Frame.ego_speed`.
- `StaticCameraSceneModel` (Sec) holds a fixed per-camera extrinsic (no ego-motion, no SLERP), keys state by a
  stable `camera_id`, and derives a scene background prior instead of a road ground plane. `camera_pose()` is
  constant across frames; `ground_plane()` may be None.

Everything downstream of the pose source (CALYX drift, ORACLYX depth, `dynamics/compute.py`, the ego-hood mask
in `autolabel/runner.py:317`) asks the `SceneModel` instead of assuming ego-motion. Where a pack's SceneModel
reports `is_static()`, those planes either skip (guarded by `manifest.capabilities`) or use the static path.

**Dissolves:** `resolve.py` ego-frame convention, `calyx/drift.py` VO-vs-IMU estimation, `mono_depth.py:28`
road-plane prior, `radar.py` ego-motion, the ego-hood mask, and the `Session.vehicle_id`/`Frame.ego_speed`
hard dependency (they become inputs the moving-camera SceneModel consumes and the static one ignores).

### `autolabel_profile: AutoLabelProfile` (SEC-M5)

Bundles the five swappable auto-label surfaces the audit named, over the reusable runner/fusion/gate plumbing.

```python
@dataclass(frozen=True)
class AutoLabelProfile:
    vlm_prompt_template: str            # was hardcoded "Indian road scene" path_c_qwen3vl.py:64
    cross_anchors: list[str]            # was CROSS_ANCHORS path_c_qwen3vl.py:186
    coco_map: dict[int, str]            # was COCO_TO_ONTOLOGY path_a_yolo26.py:21
    openvocab_synonyms: dict[str, list[str]]   # was _OPENVOCAB_SYNONYMS path_b_sam3.py:29
    gate_policy: GatePolicy             # safety_auto_accept + is_rare rules, gate.py:20
    disable_ego_hood_mask: bool         # static packs set True
```

`vlm_prompt_template` is the single highest-value seam (there is *no* config path for it today). It is a
template the pack fills, e.g. AV's "object cropped from an Indian road scene for an autonomous-driving dataset"
vs Sec's "object cropped from an Indian CCTV/security camera scene". The runner, VRAM guard, fusion voting, and
calibration/isotonic scaling are engine core and untouched.

**Dissolves:** `path_c_qwen3vl.py:64` prompt, `:186` anchors, `path_a_yolo26.py:21` COCO map,
`path_b_sam3.py:29` synonyms, `autolabel/gate.py:20` policy.

### `safety_policy: SafetyPolicy`

The `{"vru","animal"}` literal appears in six+ files (`champion.py:26`, `recall/gate.py:12`,
`safe_miou.py:13`, `autolabel/gate.py:20`, `dynamics/compute.py:33`, `flywheel/signals.py:26`). One pack-scoped
definition replaces all of them.

```python
class SafetyPolicy(Protocol):
    def is_safety_class(self, class_name: str) -> bool: ...   # AV: l1 in {"vru","animal"}
    def affinity_cost(self, a: str, b: str) -> float: ...     # AV: the safe_miou.py:21 road cost tree
    def critical_classes(self) -> frozenset[str]: ...         # by NAME, not id (fixes CRITICAL_CLASSES bug)
```

The `champion_gate` control flow (`services/govern/champion.py`) stays exactly as-is; it calls
`pack.safety_policy.is_safety_class(...)` instead of testing a module-level set. AV's policy returns the current
behaviour verbatim, so the golden harness sees no change. Sec's policy defines its own safety set (e.g.
`{"person","weapon"}`) and its own affinity.

**Dissolves:** the six copies of `{"vru","animal"}` (finding #4), the Safe-mIoU affinity tree
(`safe_miou.py:21`), and `CRITICAL_CLASSES` (fixes latent bug #2 by making it name-based).

### `eval_strata: EvalStrataSpec` (SEC-M3/M4)

Replaces the hardcoded scene axes and protected slices.

```python
@dataclass(frozen=True)
class EvalStrataSpec:
    dimensions: list[StratumDimension]   # AV: weather,time_of_day,road_type,density (curation/slices.py:15)
    protected_slices: list[str]          # AV: ["pedestrian_night","autorickshaw_glare"] (verdyx/run.py:20)
```

Two fixes ride along: the **missing config field** (latent bug #3, `GovernSettings.protected_slices` doesn't
exist, so this spec becomes the real source), and the strata dimensions currently duplicated across
`curation/slices.py:15`, `explore/facets.py:28`, and consumed by SIEVYX ODD. VERDYX verdict/stats/shadow and
`core/accel/slices.py` stay generic (they already take opaque slice-id strings). Sec's dimensions are e.g.
`camera_zone, time_of_day, occupancy, event_type`.

**Dissolves:** `verdyx/run.py:20` default + missing-field bug, `curation/slices.py:15` / `explore/facets.py:28`
strata literals, SIEVYX `odd.py` cell schema, and `sievyx/maneuver.py:13` (the maneuver taxonomy becomes one
`StratumDimension` / event vocabulary the pack supplies).

### `ingestion_adapters: list[IngestionAdapter]` (SEC-M2)

How a domain's raw captures become `Session`/`Frame`. The dataset paths are already generic
(`frames/{session}/{cam}/{ts}.jpg`), so this is about *populating* the schema, not repathing it.

```python
class IngestionAdapter(Protocol):
    def can_handle(self, source: IngestSource) -> bool: ...
    def sessions(self, source: IngestSource) -> Iterator[SessionDraft]: ...  # sets vehicle_id XOR camera_id
```

AV's adapter fills `vehicle_id` + `ego_speed` from MCAP/CAN; Sec's adapter fills `camera_id` and leaves
ego-motion fields null. `Session`/`Frame` schema is unchanged (the AV-only fields become nullable, per
AV_ASSUMPTIONS §0).

### `quality_profile: QualityProfile` (SEC-M2)

SANYX's reusable image checks stay in core; the domain-specific parts become pack data.

```python
@dataclass(frozen=True)
class QualityProfile:
    image_checks: list[str]              # reused core checks: dropped/exposure/lens (checks.py:79)
    sensor_checks: list[SensorCheck]     # AV: imu/gps/time_sync with MTi-3/NEO-F9P limits (checks.py:167)
    rootcause_signatures: list[Signature]  # was rootcause.py:14 fault table
    score_weights: ScoreWeights
```

AV supplies the inertial checks + the GMSL2/GPS/IMU fault signatures; Sec supplies CCTV signatures (stream
dropout, IR-cut flip, tampering) and no inertial checks. The scoring/gate/streaming machinery is core.

**Dissolves:** `sanyx/checks.py:167` hardcoded sensor limits, `sanyx/inertial.py` sensor vocab,
`sanyx/rootcause.py:14` AV fault table, `sanyx/lite.py`.

### `privacy_plane: PrivacyPlane` (SEC-M6)

The audit's correction to the mission stands: AV is **not** `None` here. The privacy organ already exists,
fail-closed, at two gates. The interface generalises its targets and legal regime.

```python
class PrivacyPlane(Protocol):
    redaction_targets: list[RedactionTarget]   # AV: face, plate. Sec: face, plate, + ...
    legal_regime: LegalRegime                   # DPDPA today; config not code
    def ingest_gate(self, frame: Frame) -> RedactionResult: ...   # Gate A: blur before storage
    def export_gate(self, clip: Clip) -> ExportDecision: ...      # refuse un-redacted export
```

AV instantiates it with face+plate detectors and the DPDPA regime (exactly today's behaviour, wiring at
`ingest/run.py:120` and `export/dataset.py:168` unchanged). Sec adds targets and can carry retention/consent
metadata as *config*, not new code paths. The unwired speech seam (`anonymize/speech.py:14`) becomes an
optional `RedactionTarget` a pack may enable.

**Dissolves:** the AV-typed hardcoding in `services/anonymize/` (face/plate/DPDPA as the only shape), and the
mission's incorrect `PrivacyPlane | None`.

### `forge_targets: list[ForgeTarget]` (SEC-M9)

Replaces the four AV silicon registries with a per-pack hardware profile list.

```python
@dataclass(frozen=True)
class ForgeTarget:
    name: str                    # opaque string FORGYX gate/packaging already accept
    backend: str                 # was _BACKENDS capabilities.py:18
    thermal_envelope: Thermal    # was TARGET_THERMAL thermal.py:11
    latency_budget_ms: float     # was TARGET_BUDGET_MS cooptimize.py:11
    export_format: str           # was _FORMAT run.py:19
    telemetry_parser: str        # AV: tegrastats; Sec: NVR/Ambarella/ARTPEC
```

`gate.py`/`packaging.py`/rollout are already fully generic (opaque target strings), so only the registries move.
AV registers Jetson/Sentrix targets; Sec registers CCTV SoCs and x86.

**Dissolves:** `capabilities.py:18` `_BACKENDS`, `thermal.py:11`, `cooptimize.py:11`, `run.py:19` `_FORMAT`,
`benchmark.py:44` tegrastats.

---

## Registry and discovery

```python
# packs/registry.py (engine core)
def load_packs() -> dict[str, DomainPack]: ...   # discover packs/*/pack.yaml, validate manifest, register
def get_pack(pack_id: str) -> DomainPack: ...
```

Packs live in `packs/<id>/` with a `pack.yaml` (manifest + config) and Python modules for the behavioural
surfaces (`scene_model.py`, `privacy.py`). Discovery mirrors the existing platform registry
(`web/platforms/registry.ts` has the front-end analogue; this is the backend twin). Multiple packs load in one
process; the ontology cache and every plane take a `pack_id`. A `Session.pack_id` column (nullable, defaulting
to `"av"` in the backfill) routes a session to its pack.

## What stays in the engine core (unchanged)

Per AV_ASSUMPTIONS §7, these are already neutral and do **not** get a pack seam:

- The SE(3)/geometry kernels (`core/geometry.py`, `core/accel/projection.py`, Kabsch, Sampson, consensus).
- Slice/confusion aggregation (`core/accel/slices.py`), parameterised by `n_classes`.
- The content-addressed store and all dataset path conventions (`core/storage.py`, `ingest/run.py:40`).
- The auto-label runner/fusion/gate *plumbing* (only the five profile surfaces move).
- The `champion_gate` / recall-gate / VERDYX-verdict *control flow* (only the predicates move).
- FORGYX gate/packaging/rollout (only the target registries move).
- The privacy *mechanism* and its two chokepoints (only targets + regime move).
- Every JSONB DB model (Evaluation, ModelRegistry, Deployment, PseudoLabel, ...) already domain-agnostic.

## Build order this contract implies

1. **SEC-M1** - golden regression harness + import-linter; `get_ontology(pack_id)`; author the **AV pack**
   against this contract and prove byte-parity. No behaviour change.
2. **SEC-M2** - `SceneModelFactory` + `StaticCameraSceneModel`; ingestion adapter + quality profile for Sec.
3. **SEC-M3/M4** - `OntologySpec`/`SafetyPolicy`/`EvalStrataSpec` for the Sec taxonomy; eval strata.
4. **SEC-M5** - `AutoLabelProfile` for Sec (CCTV prompt, anchors, gate policy).
5. **SEC-M6** - `PrivacyPlane` generalisation (retention/consent/rights as config).
6. **SEC-M7/M8/M9** - ANPR-India, the Sentigon webhook, `ForgeTarget` CCTV profiles.

Each milestone announces its one-paragraph plan first. The AV pack (SEC-M1) is the parity proof: if it can't
reproduce today's metrics through this contract, the contract is wrong and gets revised before Sec begins.
</content>
