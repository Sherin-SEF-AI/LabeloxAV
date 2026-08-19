"""The DomainPack contract.

This is the code form of docs/PACK_INTERFACE.md. It is pure contract: only stdlib + typing at runtime, so
the engine core and the base-only CI job can import it without pulling a pack or any ML dependency. Concrete
packs live under packs/<id>/ and are reached only through packs.registry.

Two kinds of surface live here:

* Data surfaces (dataclasses) - the values the audit found baked into engine code (ontology facts, the safety
  l1 set, the VLM prompt, the forge silicon registries, the eval strata). A pack supplies these as data.
* Behavioural surfaces (Protocols) - the genuine forks the audit found (the moving-vs-static scene model, the
  ingestion adapter, the privacy gates). SEC-M1 defines them; the AV pack fills the ones already consumed
  (ontology, safety) and leaves scene_model / ingestion_adapters for SEC-M2 (typed Optional, not stubbed).

See docs/AV_ASSUMPTIONS.md for the file:line origin of every value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Engine types used only for annotating the behavioural protocols. Under TYPE_CHECKING so importing the
    # contract never imports the engine (keeps the one-way dependency and avoids import cycles). With
    # `from __future__ import annotations` every annotation below is a string and is never evaluated at
    # runtime, so the contract module stays stdlib-only.
    from collections.abc import Iterator

    import numpy as np

    from core.geometry import Plane
    from services.autolabel.ontology import Ontology
    from services.calibration.resolve import Calibration


# --------------------------------------------------------------------------------------------------------
# Small value types
# --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SizeBound:
    """Per-superclass instance-box area bounds as a fraction of frame area (autolabel quality reviewer)."""
    min_area_frac: float
    max_area_frac: float


@dataclass(frozen=True)
class GatePolicy:
    """The confidence-gate policy surfaces the audit found AV-specific (services/autolabel/gate.py:20).

    safety_l1: l1 superclasses that must never auto-accept without the safety path.
    rare_needs_review: a rare class (india or fallback) requires human/VLM confirmation before it counts.
    """
    safety_l1: frozenset[str]
    rare_needs_review: bool = True


@dataclass(frozen=True)
class StratumDimension:
    """One eval-slice axis. Empty `values` means the axis is open-ended / discovered at runtime."""
    name: str
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensorCheck:
    """A SANYX inertial/timing check with its limits (was hardcoded in services/sanyx/checks.py:167)."""
    name: str
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RootCauseSignature:
    """A SANYX fault signature + remedy (was the AV fault table in services/sanyx/rootcause.py:14)."""
    name: str
    remedy: str


@dataclass(frozen=True)
class RedactionTarget:
    """A privacy redaction target (face, plate, speech, ...). `detector` names the engine detector to use."""
    name: str
    detector: str


# --------------------------------------------------------------------------------------------------------
# Data surfaces
# --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PackManifest:
    """Pack identity and the capability switches planes consult to skip cleanly on an absent signal."""
    id: str
    version: str
    display_name: str
    capabilities: frozenset[str]
    default_scene_model: str  # "moving_camera" | "static_camera"


@dataclass(frozen=True)
class OntologySpec:
    """The taxonomy plus the ontology facts the audit found in code rather than in the governed YAML.

    yaml_path is the governed, versioned artifact (unchanged for AV). stuff_names / stuff_l0 externalise the
    panoptic split that was a frozenset in services/autolabel/ontology.py:31. supported_core externalises
    core/config.py:396. superclass_map / size_priors are consumed when the quality reviewer moves behind the
    pack in SEC-M5; they are None until then (honest incremental fill, not a stub).
    """
    yaml_path: str
    stuff_names: frozenset[str]
    stuff_l0: frozenset[str]
    supported_core: frozenset[str]
    custom_id_base: int
    superclass_map: Mapping[str, str] | None = None       # SEC-M5
    size_priors: Mapping[str, SizeBound] | None = None     # SEC-M5


@dataclass(frozen=True)
class AutoLabelProfile:
    """The five swappable auto-label surfaces (docs/AV_ASSUMPTIONS.md section 5), consumed in SEC-M5.

    vlm_prompt_template is the domain preamble hardcoded at path_c_qwen3vl.py:64. cross_anchors,
    coco_map, openvocab_synonyms mirror the path constants. gate_policy is the gate's safety/rare rule.
    disable_ego_hood_mask is True for a static-camera pack (no moving-vehicle bonnet to mask).
    """
    vlm_prompt_template: str
    cross_anchors: tuple[str, ...]
    coco_map: Mapping[str, str]
    openvocab_synonyms: Mapping[str, tuple[str, ...]]
    gate_policy: GatePolicy
    disable_ego_hood_mask: bool = False


@dataclass(frozen=True)
class EvalStrataSpec:
    """Eval slice axes + protected slices + the safety-critical class list, by name (fixes the id-based
    CRITICAL_CLASSES in services/verdyx/safety_recall.py:10). Consumed in SEC-M3/M4.
    """
    dimensions: tuple[StratumDimension, ...]
    protected_slices: tuple[str, ...]
    critical_class_names: frozenset[str]


@dataclass(frozen=True)
class QualityProfile:
    """SANYX profile: the reused image checks + the domain sensor checks + the fault table. Consumed SEC-M2."""
    image_checks: tuple[str, ...]
    sensor_checks: tuple[SensorCheck, ...]
    rootcause_signatures: tuple[RootCauseSignature, ...]


@dataclass(frozen=True)
class ForgeTarget:
    """One edge hardware profile, replacing the AV silicon registries (services/forgyx/*). FORGYX gate and
    packaging already accept opaque `name` strings; only these registries were AV-shaped.
    """
    name: str
    backend: str
    backend_modules: tuple[str, ...]
    throttle_temp_c: float
    power_ceiling_w: float
    latency_budget_ms: float
    export_format: str


@dataclass(frozen=True)
class ConfusionClique:
    """A set of classes a model genuinely confuses, and what confusing them costs.

    A clique is not "classes that look alike": it is a set where the model's probability mass moves
    between the members, so a labelling budget spent inside it buys a decision boundary rather than more
    examples of one class. `cost` is the price of confusing two members, in [0, 1], and it is what makes
    a scooter called a motorcycle cheap and a pedestrian called a bollard expensive.

    `crosses_safety` records that at least one member is a safety class and at least one is not, which is
    the case where a confusion inside an apparently tidy clique is the most expensive kind of error.
    """
    name: str
    class_names: tuple[str, ...]
    cost: float
    crosses_safety: bool = False


@dataclass(frozen=True)
class CliqueSpec:
    """The confusion cliques of a domain, and the cost of confusing classes across them.

    Held by the pack because which classes a model confuses is a fact about the world it works in: a
    scooter and a motorcycle are one decision on an Indian road and two unrelated objects in a warehouse.
    """
    cliques: tuple[ConfusionClique, ...]
    # Cost of confusing two classes that are not in a clique together. Higher than any within-clique cost
    # by construction: a confusion the ontology did not anticipate is worse than one it did.
    cross_clique_cost: float = 1.0

    def by_name(self, name: str) -> ConfusionClique | None:
        for c in self.cliques:
            if c.name == name:
                return c
        return None

    def clique_of(self, class_name: str) -> ConfusionClique | None:
        for c in self.cliques:
            if class_name in c.class_names:
                return c
        return None

    def pair_cost(self, a: str, b: str) -> float:
        """Cost of confusing a for b. Zero for a class with itself, the clique cost within one, else cross."""
        if a == b:
            return 0.0
        ca, cb = self.clique_of(a), self.clique_of(b)
        if ca is not None and ca is cb:
            return ca.cost
        return self.cross_clique_cost


@dataclass(frozen=True)
class RelationSpec:
    """The relationship vocabulary, and which ordered class pairs can stand in which relation.

    Two disjoint vocabularies are live in the engine today and neither knows about the other:
    services/api/routers/objects.py validates {rider_of, towed_by, part_of, member_of, occludes} on the
    editor path, while services/intelligence/scene_graph.py inserts {occluded_by, following,
    crossing_in_front_of, parked_near} directly and never passes that validation. `occludes` and
    `occluded_by` are the same fact in opposite directions and both are stored, so a query for one silently
    misses half the corpus.

    `kinds` is the union any writer may use. `overlap_pairs` maps an ordered (subject l1, object l1) pair
    to the relation an overlapping pair of those superclasses probably stands in, which is what lets
    relationship-aware NMS keep a rider and their motorcycle apart and say why. `inverse` records the pairs
    that are one fact in two directions, so a reader can normalise rather than guess.
    """
    kinds: frozenset[str]
    overlap_pairs: Mapping[tuple[str, str], str]
    inverse: Mapping[str, str] = field(default_factory=dict)

    def relation_for_l1(self, subject_l1: str, object_l1: str) -> str | None:
        return self.overlap_pairs.get((subject_l1, object_l1))

    def canonical(self, kind: str) -> str:
        """The direction this engine stores. An inverse maps to its canonical form; anything else is itself."""
        return self.inverse.get(kind, kind)


@dataclass(frozen=True)
class PrivacyPlaneSpec:
    """The redaction targets + legal regime for a pack. The behavioural PrivacyPlane gates (ingest/export)
    already exist as the engine anonymize organ; SEC-M6 binds them to these targets. AV is never None here
    (it runs face+plate redaction today), correcting the mission's `PrivacyPlane | None`.
    """
    redaction_targets: tuple[RedactionTarget, ...]
    legal_regime: str


# --------------------------------------------------------------------------------------------------------
# Behavioural surfaces (Protocols)
# --------------------------------------------------------------------------------------------------------

@runtime_checkable
class SafetyPolicy(Protocol):
    """Replaces the `{"vru","animal"}` literal copy-pasted across six+ files (AV_ASSUMPTIONS section 6)."""
    def is_safety_class(self, class_name: str) -> bool: ...
    def affinity_cost(self, a_id: int, b_id: int) -> float: ...
    def critical_class_names(self) -> frozenset[str]: ...
    def accept_far_bound(self, class_name: str) -> float: ...


@runtime_checkable
class SceneModel(Protocol):
    """Per-camera geometry. MovingCameraSceneModel (AV) vs StaticCameraSceneModel (Sec); the fork the audit
    located at services/calibration/resolve.py. `reference` is the ego frame for a moving camera and the fixed
    world frame for a static camera - is_static() tells them apart.
    """
    def is_static(self) -> bool: ...
    def rotation(self) -> np.ndarray: ...                                          # Calibration.R(): cam=(ref-t)@R
    def extrinsic(self) -> np.ndarray: ...                                          # SE(3) reference<-camera
    def camera_pose(self, world_from_reference: np.ndarray | None = None) -> np.ndarray: ...  # SE(3) world<-camera
    def ground_plane(self) -> Plane | None: ...


@runtime_checkable
class SceneModelFactory(Protocol):
    """Builds a SceneModel from a resolved per-camera Calibration."""
    def build(self, calib: Calibration) -> SceneModel: ...


@dataclass(frozen=True)
class IngestSource:
    """A pack-agnostic handle to raw captures: a video file, an image directory, an RTSP URL, an MCAP, ..."""
    media_type: str
    uri: str
    cam_id: str
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameDraft:
    """One decoded frame on its way to a Frame row. ego_speed is None for a static camera."""
    ts_ns: int
    cam_id: str
    image_bgr: np.ndarray
    ego_speed: float | None = None
    lat: float | None = None
    lon: float | None = None


@dataclass(frozen=True)
class SessionDraft:
    """Session-level metadata for a capture. vehicle_id is None for a static-camera pack (no ego vehicle);
    pack_id routes the session to its pack; cam_id is the stable camera identity."""
    pack_id: str
    cam_id: str
    vehicle_id: str | None = None
    city: str | None = None
    route: str | None = None


@runtime_checkable
class IngestionAdapter(Protocol):
    """Turns a domain's raw captures into a SessionDraft + a stream of FrameDrafts. The engine writes the rows;
    the adapter only knows how to decode this domain's sources and which schema fields it populates."""
    media_types: frozenset[str]
    def can_handle(self, source: IngestSource) -> bool: ...
    def read(self, source: IngestSource) -> tuple[SessionDraft, Iterator[FrameDraft]]: ...


@runtime_checkable
class PrivacyPlane(Protocol):
    """The two privacy chokepoints (blur-before-storage; refuse-un-redacted-export). Bound in SEC-M6."""
    def ingest_gate(self, frame: object) -> object: ...
    def export_gate(self, clip: object) -> object: ...


@dataclass(frozen=True)
class Crossing:
    """One spatial rule firing, before the engine decides whether it becomes an incident.

    The shape lives in the contract rather than in the pack because the engine reads every field of it to
    build an incident. A pack decides *when* a rule fires; what a firing looks like is engine vocabulary.
    """
    zone_id: str
    zone_name: str
    rule: str
    track_id: str | None
    class_name: str
    ts_ns: int
    severity: str
    detail: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class ZonePolicy(Protocol):
    """Spatial rules over a camera's field of view: what a zone may say, and when a track violates it.

    Geometry is domain knowledge. Whether a loitering track in a doorway is an event at all has no meaning in
    the AV pack and is the core of the security one, so the engine owns zone storage and incident raising
    while the pack owns the predicate.
    """
    def validate(self, kind: str, rule: str, points: Sequence, dwell_seconds: float | None) -> None:
        """Raise ValueError if a zone definition is not one this pack can evaluate."""
        ...

    def evaluate_track(self, zone: Mapping[str, object],
                       samples: Sequence[Mapping[str, object]]) -> Sequence[Crossing]: ...


@runtime_checkable
class StreamSource(Protocol):
    """Sampling a live camera into a session.

    A live stream has no end, so every method here is bounded by construction. `unavailable_error` is part of
    the surface because the engine has to distinguish an unreachable camera from its own failure, and it
    cannot do that by catching a pack-private exception type.
    """
    unavailable_error: type[Exception]

    def sampling_policy(self, **overrides: object) -> object:
        """A pack-defined policy object, built from engine-supplied overrides. Opaque to the engine: it is
        carried straight back into `ingest` and never inspected."""
        ...

    async def ingest(self, url: str, camera_id: str, *, city: str | None = None,
                     policy: object | None = None, max_frames: int | None = None,
                     max_seconds: float | None = None, pack_id: str | None = None) -> dict: ...


# --------------------------------------------------------------------------------------------------------
# The pack
# --------------------------------------------------------------------------------------------------------

@runtime_checkable
class DomainPack(Protocol):
    """The contract a concrete pack satisfies. Attribute-only Protocol: any object exposing these fields is a
    DomainPack (runtime_checkable checks presence). scene_model / ingestion_adapters are Optional / empty until
    SEC-M2 wires the scene-model fork; every other surface is real in SEC-M1.
    """
    manifest: PackManifest
    ontology: OntologySpec
    safety_policy: SafetyPolicy
    autolabel_profile: AutoLabelProfile
    eval_strata: EvalStrataSpec
    quality_profile: QualityProfile
    forge_targets: Sequence[ForgeTarget]
    privacy: PrivacyPlaneSpec
    relations: RelationSpec | None
    cliques: CliqueSpec | None
    scene_model: SceneModelFactory | None
    ingestion_adapters: Sequence[IngestionAdapter]
    # Optional because they are genuinely domain-specific rather than merely unfinished: an AV pack has no
    # fixed zones to police and no camera to open. A pack that does not fill them is complete, and the engine
    # refuses the corresponding route rather than pretending the capability exists.
    zone_policy: ZonePolicy | None
    stream_source: StreamSource | None


@dataclass(frozen=True)
class Pack:
    """Concrete DomainPack container. A pack module builds one of these and exposes it as `PACK`. Using a
    frozen dataclass (rather than each pack hand-rolling attributes) gives every pack the same shape and lets
    the registry validate structurally.
    """
    manifest: PackManifest
    ontology: OntologySpec
    safety_policy: SafetyPolicy
    autolabel_profile: AutoLabelProfile
    eval_strata: EvalStrataSpec
    quality_profile: QualityProfile
    forge_targets: tuple[ForgeTarget, ...]
    privacy: PrivacyPlaneSpec
    # Optional: a domain with no meaningful object-to-object relations (a static camera counting entries)
    # is complete without one, and the engine proposes no edges rather than inventing a vocabulary for it.
    relations: RelationSpec | None = None
    # Optional: a domain whose classes are not confusable in structured groups gets uniform sampling
    # rather than an invented grouping.
    cliques: CliqueSpec | None = None
    scene_model: SceneModelFactory | None = None            # SEC-M2
    ingestion_adapters: tuple[IngestionAdapter, ...] = ()   # SEC-M2
    zone_policy: ZonePolicy | None = None                   # static-camera domains only
    stream_source: StreamSource | None = None               # static-camera domains only


def superclass_affinity_cost(onto: Ontology, a_id: int, b_id: int, safety_l1: frozenset[str]) -> float:
    """Generic safety cost of confusing class a for class b, in [0, 1], parameterised by a pack's safety l1
    set. Mirrors the AV safe-mIoU cost tree (services/training/safe_miou.py) but with the safety superclasses
    supplied rather than the hardcoded `{"vru","animal"}`, so a non-AV pack gets a real, safety-aware affinity.
    """
    BACKGROUND_ID = -1
    if a_id == b_id:
        return 0.0
    if a_id == BACKGROUND_ID or b_id == BACKGROUND_ID:
        real = b_id if a_id == BACKGROUND_ID else a_id
        try:
            return 1.0 if onto.by_id(real).l1 in safety_l1 else 0.5
        except KeyError:
            return 0.5
    ca, cb = onto.by_id(a_id), onto.by_id(b_id)
    a_safety, b_safety = ca.l1 in safety_l1, cb.l1 in safety_l1
    if a_safety != b_safety:
        cost = 1.0                      # a safety-critical class confused with anything non-safety
    elif ca.l1 == cb.l1:
        cost = 0.2
    elif ca.l0 == cb.l0:
        cost = 0.5
    else:
        cost = 0.8
    if ca.india or cb.india or ca.l1 == "fallback" or cb.l1 == "fallback":
        cost = min(1.0, cost + 0.1)
    return cost


#: Fallback auto-accept false-accept bounds, used by any pack that does not state its own. Deliberately
#: strict: a pack that has not thought about what a wrong auto-accept costs it should get the cautious
#: answer, because the failure mode of the loose one is a wrong label entering the corpus unseen.
DEFAULT_FAR_BOUNDS: Mapping[str, float] = MappingProxyType({
    "critical": 0.01,   # the classes whose confusion the safety gate already refuses to automate
    "safety": 0.02,     # the rest of the safety superclasses
    "default": 0.05,    # everything else
})


def make_safety_policy(onto: Ontology, safety_l1: frozenset[str], critical_names: frozenset[str],
                       affinity: Callable[[Ontology, int, int], float] | None = None,
                       far_bounds: Mapping[str, float] | None = None) -> SafetyPolicy:
    """Build the standard ontology-backed SafetyPolicy. Kept in the contract module (not a specific pack) so
    every pack that defines safety by an l1 superclass set gets the identical, tested implementation.

    `affinity` computes the safety cost of confusing two class ids. It defaults to the AV safe-mIoU cost, so
    the AV number is byte-identical to today's governance; a non-AV pack passes its own (e.g. a partial of
    superclass_affinity_cost bound to its safety_l1), since the AV cost hardcodes the VRU/animal semantics.

    `far_bounds` are the auto-accept false-accept bounds by tier (critical / safety / default). Missing
    keys fall back to DEFAULT_FAR_BOUNDS, so a pack states only what it disagrees with.
    """
    if affinity is None:
        from services.training.safe_miou import affinity_cost
        affinity_fn: Callable[[Ontology, int, int], float] = affinity_cost
    else:
        affinity_fn = affinity
    bounds = {**DEFAULT_FAR_BOUNDS, **(far_bounds or {})}

    class _OntologySafetyPolicy:
        def is_safety_class(self, class_name: str) -> bool:
            try:
                return onto.by_name(class_name).l1 in safety_l1
            except Exception:  # noqa: BLE001 - an unknown name is simply not a safety class
                return False

        def affinity_cost(self, a_id: int, b_id: int) -> float:
            return affinity_fn(onto, a_id, b_id)

        def critical_class_names(self) -> frozenset[str]:
            return critical_names

        def accept_far_bound(self, class_name: str) -> float:
            """How often an auto-accepted box of this class may be wrong, in [0, 1].

            The alpha that core/accel/np_threshold.py fits an operating point against. It belongs to the
            pack because the cost of a wrong auto-accept is a domain judgement and not a model property: a
            mislabelled pedestrian and a mislabelled bollard are not equally expensive, and one bound
            across the ontology is wrong for at least one of them.

            The tiering is generic (critical, then safety, then everything else) and the numbers come from
            the pack, so a domain that defines safety differently gets a correct policy without new code
            here.
            """
            if class_name in critical_names:
                return bounds["critical"]
            try:
                safety = onto.by_name(class_name).l1 in safety_l1
            except Exception:  # noqa: BLE001 - an unknown name gets the cautious bound, not the loose one
                return bounds["critical"]
            return bounds["safety"] if safety else bounds["default"]

    return _OntologySafetyPolicy()
