"""The AV domain pack: LabeloxAV's original behaviour, authored against the DomainPack contract.

Every surface here is built from the engine's own values (imported, not re-declared) so the pack is a faithful
mirror and byte-parity holds by construction. The golden regression harness (tests/test_golden_av_pack.py)
freezes these values; any future change to AV behaviour trips it. See docs/AV_ASSUMPTIONS.md for the file:line
origin of each value.

Import direction: this module imports the engine (allowed). The engine never imports this module statically;
it reaches it only through packs.registry (enforced by .importlinter).
"""

from __future__ import annotations

from core.config import get_settings
from packs.av.data.india_cities import ALIASES, CITIES, NON_INDIA
from packs.av.scene_model import MovingCameraSceneModelFactory
from packs.base import (
    AttributeSpec,
    AutoLabelProfile,
    ClassTree,
    CliqueSpec,
    ConfusionClique,
    ContextSpec,
    EvalStrataSpec,
    ForgeTarget,
    GatePolicy,
    MotionModelSpec,
    OntologySpec,
    Pack,
    PackManifest,
    PrivacyPlaneSpec,
    QualityProfile,
    RedactionTarget,
    RegionSpec,
    RelationSpec,
    RootCauseSignature,
    SensorCheck,
    StratumDimension,
    TrackEventSpec,
    TrackEventType,
    make_safety_policy,
)

# Single source of truth: import the AV values that live in engine code today (all import-light modules).
from services.autolabel.ontology import (
    CUSTOM_ID_BASE,
    STUFF_L0,
    STUFF_NAMES,
    get_ontology,
)
from services.curation.slices import _SCENE_AXES
from services.forgyx.capabilities import _BACKENDS
from services.forgyx.cooptimize import TARGET_BUDGET_MS
from services.forgyx.run import _FORMAT
from services.forgyx.thermal import TARGET_THERMAL
from services.sanyx.rootcause import REMEDIATION

# The l1 superclasses that define a safety-critical class. This is the `{"vru","animal"}` literal the audit
# found copy-pasted across six+ files (govern/champion.py:26, recall/gate.py:12, training/safe_miou.py:13,
# autolabel/gate.py:20, dynamics/compute.py:33, flywheel/signals.py:26). Consolidated here; SEC-M3/M4 point
# those call sites at pack.safety_policy.
AV_SAFETY_L1 = frozenset({"vru", "animal"})

# The safety-critical classes, by NAME. Fixes latent bug #2: services/verdyx/safety_recall.py:10 lists them
# as 0-based ids {0,1,2,3,8}, which disagree with the governed 1-based YAML ids. Consumed in SEC-M4.
#
# `pothole` and `open_manhole` are here because a hole in the road is a VRU problem. Every other member is
# something a vehicle might hit; these two are things a two-wheeler falls into, in a country where
# two-wheelers are most of the traffic and the rider has no cage. A missed pedestrian and a missed open
# manhole put the same person in hospital.
#
# Both join wholesale rather than only when severe. This set is keyed by class name, so "a severe pothole is
# critical" is not expressible in it, and the two ways to resolve that are to include every pothole or to
# include none. Recall on a shallow pothole costing a little extra sampling is the cheaper mistake, and
# severity stays available as the `hazard_severity` attribute for anything that wants to weight by it.
AV_CRITICAL_CLASSES = frozenset({"pedestrian", "rider", "motorcycle", "cycle", "cattle",
                                 "pothole", "open_manhole"})

# The domain preamble hardcoded at services/autolabel/paths/path_c_vlm.py:64 (no config seam today). The
# parity test asserts the live prompt still begins with this; SEC-M5 makes path_c read it from here.
AV_VLM_PROMPT_TEMPLATE = (
    "You are labeling an object cropped from an Indian road scene for an autonomous-driving "
    "dataset. Identify the object and read its attributes."
)

# The auto-label path values now live here (the pack owns them); the three paths read them at runtime from
# pack.autolabel_profile. Moved verbatim from services/autolabel/paths/* so AV behaviour is byte-identical.

# Cross-superclass road actors always offered to the VLM so it can fix gross mislabels across superclasses.
AV_CROSS_ANCHORS = (
    "autorickshaw", "e_auto", "e_rickshaw", "motorcycle", "scooter", "cycle",
    "pedestrian", "rider", "cyclist", "sedan", "suv", "hatchback", "pickup",
    "truck", "lcv", "bus", "tempo", "water_tanker", "cattle", "dog",
    "push_cart", "vendor_handcart", "street_vendor",
)

# COCO class name -> ontology class name; unmapped COCO detections fold to object_fallback (never dropped).
AV_COCO_TO_ONTOLOGY = {
    "person": "pedestrian",
    "bicycle": "cycle",
    "motorcycle": "motorcycle",
    "car": "sedan",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_signal",
    "stop sign": "traffic_sign",
    "fire hydrant": "pole",
    "bench": "object_fallback",
    "cat": "object_fallback",
    "dog": "dog",
    "horse": "cattle",
    "sheep": "goat",
    "cow": "cattle",
    "elephant": "object_fallback",
    "bird": "object_fallback",
}

# Open-vocab synonyms mapped back to one ontology class, so long-tail rural classes get proposed.
#
# A class with no entry here is prompted with its own name and the underscores swapped for spaces, which for
# signs meant the single phrase "traffic sign" covering 21 distinct Indian designs, from a red octagon to a
# green destination board. The phrases below name the designs the corpus actually contains.
#
# Signs and advertising are kept lexically apart on purpose. While `hoarding` was stuff the autolabeller
# dropped it and its contents were labelled `traffic_sign`; now that it is a thing, the two classes would
# compete for the same detections if their prompts both said "sign". So the sign phrases never say
# "billboard" or "advertisement", and the advertising phrases never say "sign". Tests hold both directions.
AV_OPENVOCAB_SYNONYMS = {
    "cattle": ("cattle", "cow", "buffalo", "bull", "ox"),
    "traffic_sign": (
        "traffic sign", "road sign", "a red and white triangular warning sign",
        "a circular speed limit sign", "a red octagonal stop sign",
        "a blue circular mandatory direction sign", "a green destination board with place names",
    ),
    "hoarding": (
        "advertising hoarding", "billboard", "a large commercial advertisement board",
        "a shop name board", "a fuel station price display",
    ),
    "traffic_signal": ("traffic signal", "traffic light", "a signal head with red amber green lamps"),
}


def _forge_targets() -> tuple[ForgeTarget, ...]:
    """Reassemble the AV silicon registries (services/forgyx/*) into ForgeTargets. The four deployment
    targets are those with a full thermal+latency profile; onnx/onnxruntime are intermediate backends."""
    out: list[ForgeTarget] = []
    for name in sorted(TARGET_BUDGET_MS):
        thermal = TARGET_THERMAL[name]
        out.append(ForgeTarget(
            name=name,
            backend=name,
            backend_modules=tuple(_BACKENDS.get(name, ())),
            throttle_temp_c=float(thermal["throttle_temp_c"]),
            power_ceiling_w=float(thermal["power_ceiling_w"]),
            latency_budget_ms=float(TARGET_BUDGET_MS[name]),
            export_format=_FORMAT.get(name, name),
        ))
    return tuple(out)


def _quality_profile() -> QualityProfile:
    """SANYX profile: the reused image checks + the AV inertial checks (limits from checks.py:167) + the AV
    fault table (services/sanyx/rootcause.py REMEDIATION)."""
    return QualityProfile(
        image_checks=("dropped_frames", "exposure", "lens_contamination"),
        sensor_checks=(
            SensorCheck(name="time_sync"),
            SensorCheck(name="gps"),
            # Xsens MTi-3 full-scale limits (services/sanyx/checks.py:167 defaults).
            SensorCheck(name="imu", params={"accel_limit": 156.9, "gyro_limit": 34.9}),
        ),
        rootcause_signatures=tuple(
            RootCauseSignature(name=k, remedy=v) for k, v in sorted(REMEDIATION.items())
        ),
    )


def _motion_models() -> MotionModelSpec:
    """Which infrastructure sits on the road surface and which is above it.

    The split matters because the ground homography is exact for the first group and confidently wrong for
    the second. A gantry warped by the ground plane moves in the opposite direction to the truth, and a
    propagated box that is wrong in a plausible-looking way costs an annotator more than no box at all.

    Everything not named here is "moving", which refuses to propagate. That is the safe default: a vehicle
    or a person mistakenly treated as static would have its box placed by ego motion alone.
    """
    return MotionModelSpec(
        static_ground=frozenset({
            "cone", "barrier", "barricade_line", "construction_barrier", "crash_barrier",
            "median_barrier", "temp_barricade", "guardrail", "fence", "sandbag", "tar_drum",
            "hume_pipe", "debris", "garbage_pile", "excavation_pit", "waterlogging", "fallen_tree",
            "electric_pole", "light_pole", "traffic_pole", "signal_pole", "cctv_pole", "pole",
            "metro_pillar", "flyover_pillar", "transformer", "postbox", "milestone", "km_stone",
            "tree", "vegetation", "shrine", "telephone_booth",
        }),
        static_elevated=frozenset({
            "traffic_signal", "pedestrian_signal", "traffic_sign", "chevron_sign", "street_light",
            "overhead_water_tank", "hoarding", "foot_overbridge", "speed_camera",
        }),
    )


def _region() -> RegionSpec:
    """Bengaluru is in this corpus 373 times under two spellings. That is what this exists to fix.

    Ordered from largest so a coverage report renders strata in a meaningful order. The classification is
    the Census of India 2011 urban-agglomeration one, cited in packs/av/data/india_cities.py.
    """
    def resolve(s: str) -> tuple[str, str, str] | None:
        key = ALIASES.get(s, s)
        c = CITIES.get(key)
        return None if c is None else (c.name, c.state, c.urban_class)

    return RegionSpec(resolve=resolve, classes=("megacity", "million_plus", "class_1", "other"),
                      keys=lambda: frozenset(CITIES) | frozenset(ALIASES),
                      outside=NON_INDIA.get)


def _track_events() -> TrackEventSpec:
    """What a track did, and over which frames.

    The vocabulary is the sixteen words already governed in services/intelligence/intent.py, plus seven
    India spans that vocabulary has no way to say. Reusing the existing spellings is deliberate: `cut_in`
    means one thing in this system, and a second vocabulary that spelled it `cutting_in` would split every
    query that ever asks for it. tests/test_track_events.py asserts the superset relation holds, so an edit
    to either list that breaks it fails rather than quietly forking the two.

    The definitions are the interface. An annotator reads one of these at the moment they are deciding
    whether a span starts, so each says what must be visible for it to begin and what ends it. A vague
    definition does not produce fewer labels, it produces labels two people disagree about silently.

    Only two are proposable, and both are marked so here rather than left to be inferred. The rest are
    human-labeled and stay that way until there is a signal that can see them.
    """
    V, R, A = "vehicle", "vru", "any"
    return TrackEventSpec(types=(
        # ---- the governed intents, now with extent ----
        TrackEventType("cut_in", "Starts when the vehicle crosses the lane line into the ego lane with less "
                       "than about two seconds of gap; ends when it is fully within the lane.", V),
        TrackEventType("hard_brake", "Starts at the first frame of visible rapid deceleration (brake lights, "
                       "pitch, closing gap); ends when the speed is steady again.", V, proposable=True),
        TrackEventType("u_turn", "Starts when the vehicle begins turning back on itself; ends when it is "
                       "travelling in the opposite direction.", V),
        TrackEventType("overtaking", "Starts when the vehicle leaves its lane to pass another; ends when it "
                       "has returned to a lane ahead of the vehicle it passed.", V),
        TrackEventType("wrong_side", "The whole span during which the vehicle travels against the flow of "
                       "the carriageway it is on.", V),
        TrackEventType("parking", "Starts when the vehicle begins its final manoeuvre into a stopped "
                       "position; ends when it is stationary and no longer manoeuvring.", V),
        TrackEventType("changing_lane", "Starts when the vehicle begins to cross a lane line; ends when it "
                       "is fully in the new lane. Use cut_in instead when the target lane is the ego lane.", V),
        TrackEventType("merging", "Starts when the vehicle begins to join the carriageway from a slip road, "
                       "service lane or side road; ends when it is in a through lane.", V),
        TrackEventType("yielding", "Starts when the vehicle slows or stops to let another road user pass; "
                       "ends when it resumes.", V),
        TrackEventType("crossing", "Starts when the person or animal enters the carriageway; ends when they "
                       "leave it. Use this whether or not there is a marked crossing.", A),
        TrackEventType("waiting", "The span during which the person is stationary at the edge of the "
                       "carriageway, oriented towards it, before crossing.", R),
        TrackEventType("running", "The span during which the person moves at a run rather than a walk.", R),
        TrackEventType("jaywalking", "Crossing away from a marked crossing while one is visible in frame. "
                       "Use crossing when no crossing is visible.", R),
        TrackEventType("looking_at_vehicle", "The span during which the person's head is turned towards the "
                       "ego vehicle. Label only when the face or head orientation is actually visible.", R),
        TrackEventType("hesitating", "Starts when the person stops or reverses direction part way across; "
                       "ends when they commit to a direction.", R),
        TrackEventType("entering_lane", "Starts when the person or animal crosses the lane edge into a "
                       "running lane; ends when they are clear of it.", A),
        # ---- India spans the intent vocabulary cannot say ----
        TrackEventType("lane_splitting", "The span during which a two-wheeler travels along the gap between "
                       "two lanes of slower or stopped traffic rather than within a lane.", V),
        TrackEventType("overtaking_on_left", "Overtaking on the near side. Separate from overtaking because "
                       "traffic keeps left here, so passing on the left is the manoeuvre with no mirror "
                       "covering it.", V),
        TrackEventType("stopping_in_live_lane", "Starts when the vehicle comes to a stop in a running lane "
                       "without pulling off the carriageway; ends when it moves off. This is what a bus or "
                       "an auto does at a stop, and it is a planning problem rather than a violation.", V,
                       proposable=True),
        TrackEventType("blocking_intersection", "Starts when the vehicle enters a junction it cannot clear "
                       "and stops within it; ends when it leaves the junction.", V),
        TrackEventType("reversing_on_carriageway", "The span during which the vehicle moves backwards along "
                       "a running lane, as opposed to reversing into a parking position.", V),
        TrackEventType("encroaching_opposing_lane", "Starts when the vehicle crosses the centre into "
                       "oncoming traffic while continuing in its own direction; ends when it returns. Use "
                       "wrong_side when it stays there.", V),
        TrackEventType("weaving", "Repeated lateral movement across lane lines without settling, over at "
                       "least three crossings.", V),
    ))


def _cliques() -> CliqueSpec:
    """The confusions an Indian-road detector actually makes, and what each costs.

    Grouped by what a label buys rather than by what looks alike. Inside a clique the model's probability
    mass moves between the members, so a labelled example draws a boundary; across cliques it does not,
    so a label mostly adds one more example of something already learned.

    Costs are the safe-mIoU affinity semantics reused: 0.2 within a superclass, 1.0 across a safety
    boundary. pedestrians_vs_riders is the expensive one and it is expensive for a specific reason: a
    person on a motorcycle and a person walking need different predictions from a planner, and the
    classes are visually nearly identical from behind.
    """
    return CliqueSpec(
        cliques=(
            ConfusionClique("two_wheelers", ("motorcycle", "scooter", "moped", "cycle"), cost=0.2),
            ConfusionClique("three_wheelers", ("autorickshaw", "e_rickshaw", "cycle_rickshaw"), cost=0.2),
            ConfusionClique("livestock", ("cattle", "buffalo", "goat", "dog", "pig"), cost=0.2),
            # The most common confusion in the corpus, and it had no clique. Judging 80 crops per class
            # against a VLM: `sedan` confirms at 0.220 and its rejections propose suv 12 times, hatchback 9
            # and minivan 2; `mpv` at 0.794 proposes suv 9 and hatchback 5; `suv` at 0.742 proposes minivan
            # 6, app_cab 3 and hatchback 3. Every one of those is a body-style call on a passenger car, and
            # the YAML already says so - 94.3% of four-wheelers are labelled `sedan`, which is why
            # `body_style` exists as an attribute. Cheap because a planner treats them identically.
            ConfusionClique("four_wheelers",
                            ("sedan", "hatchback", "suv", "mpv", "minivan", "pickup", "jeep",
                             "taxi", "app_cab", "luxury_sedan", "vintage_car"), cost=0.2),
            # Every member is a person; what differs is what they are doing, which is exactly what a
            # planner needs and what a detector gets wrong from behind.
            ConfusionClique("pedestrians_vs_riders",
                            ("pedestrian", "rider", "scooter_with_rider", "delivery_rider_bike",
                             "child", "person_carrying_load"),
                            cost=1.0, crosses_safety=True),
            ConfusionClique("heavy_vehicles",
                            ("bus", "truck", "tractor", "water_tanker", "petrol_tanker",
                             "container_truck", "multi_axle_trailer",
                                              # Two bodies on one chassis, and the
                                              # failure is a box around one of them. It joins this
                                              # clique rather than getting an `articulated` one of its
                                              # own, because the confusion it is actually in is with
                                              # `tractor`, which is already here, and because a class
                                              # may belong to at most one clique.
                                              "tractor_trolley"), cost=0.2),
            ConfusionClique("carts", ("bullock_cart", "push_cart", "vendor_handcart", "cargo_bike"),
                            cost=0.2),
        ),
        cross_clique_cost=1.0,
    )


def _context() -> ContextSpec:
    """What a person may record about the whole frame, beyond what the ingest classifier already guessed.

    The engine writes density, weather, road_type and time_of_day into `Frame.scene` at ingest with a
    confidence per axis. These are the axes a machine does not reliably call and that change what the frame
    is worth: a monsoon downpour and a dust haze both cut visibility but fail differently, an unlit road at
    night is a different problem from a mixed-lit one, and a market street or a procession is a density the
    density classifier has no word for.

    Nothing here is about an object. `waterlogging` is the state of the road surface across the frame, not
    of any one puddle; the `waterlogged` attribute on a pothole is the separate, per-object fact.
    """
    enum = lambda *v: AttributeSpec(type="enum", values=tuple(v))   # noqa: E731 - a table reads better
    return ContextSpec(attributes={
        "monsoon_intensity": enum("none", "light", "heavy"),
        "haze": enum("none", "light", "dense"),
        "dust": AttributeSpec(type="bool"),
        "glare_oncoming": AttributeSpec(type="bool"),
        "festival_or_procession": AttributeSpec(type="bool"),
        "market_street": AttributeSpec(type="bool"),
        "night_lighting": enum("well_lit", "mixed", "unlit", "na_daytime"),
        "waterlogging": enum("none", "partial", "flooded"),
    })


def _relations() -> RelationSpec:
    """The AV relationship vocabulary, unified across the two that were live and disjoint.

    The editor validated {rider_of, towed_by, part_of, member_of, occludes}; the scene-graph proposer wrote
    {occluded_by, following, crossing_in_front_of, parked_near} straight past that validation. `kinds` is
    the union, so a writer on either path is checkable against one list.

    `occludes` and `occluded_by` are the same fact in opposite directions and both were being stored, which
    means a query for either silently missed half the corpus. `occludes` is kept as canonical because it is
    the one the editor offers and the one a person draws.

    `overlap_pairs` is keyed on l1 superclasses rather than leaf classes, because the relation is a property
    of the kind of thing: every VRU heavily overlapping a two-wheeler is a rider on it, whatever the leaf
    class of either. This is what lets relationship-aware NMS keep a rider and their motorcycle as two
    objects instead of merging them, and say which relation justified it.
    """
    return RelationSpec(
        kinds=frozenset({"rider_of", "towed_by", "part_of", "member_of", "occludes",
                         "occluded_by", "following", "crossing_in_front_of", "parked_near",
                         # towing is the active side of towed_by; pulling is an animal or a person drawing
                         # a cart, which is not a tow and does not move like one; herding is a person
                         # driving livestock without touching any of it.
                         "towing", "pulling", "herding"}),
        overlap_pairs={
            # A person on a two-wheeler or three-wheeler. The India case the editor was built around.
            ("vru", "two_wheeler"): "rider_of",
            ("vru", "three_wheeler"): "rider_of",
            # An animal drawing a cart, and a person pushing one: both are a VRU or animal in contact with
            # a vehicle that is not carrying them.
            ("animal", "cart"): "towed_by",
            ("vru", "cart"): "towed_by",
            # A person inside a four-wheeler, which overlaps heavily and is emphatically not one object.
            ("vru", "four_wheeler"): "part_of",
            ("vru", "heavy_vehicle"): "part_of",
        },
        inverse={"occluded_by": "occludes", "towed_by": "towing"},
    )


def _build() -> Pack:
    settings = get_settings()
    onto = get_ontology("av")

    # Fail loud if a critical class name is not in the governed ontology (guards against the id/name drift
    # that produced latent bug #2 in the first place).
    for name in AV_CRITICAL_CLASSES:
        if not onto.has_name(name):
            raise RuntimeError(f"AV critical class '{name}' is not in ontology {onto.version}")

    manifest = PackManifest(
        id="av",
        version="0.1.0",
        display_name="Autonomous Driving (India)",
        capabilities=frozenset({
            "moving_camera", "ego_motion", "dynamics", "mono_depth", "inertial_qa", "privacy_redaction",
        }),
        default_scene_model="moving_camera",
    )

    ontology = OntologySpec(
        yaml_path=str(settings.ontology_abspath()),
        stuff_names=STUFF_NAMES,
        stuff_l0=STUFF_L0,
        supported_core=frozenset(settings.ontology.supported_core),
        custom_id_base=CUSTOM_ID_BASE,
    )

    autolabel = AutoLabelProfile(
        vlm_prompt_template=AV_VLM_PROMPT_TEMPLATE,
        cross_anchors=AV_CROSS_ANCHORS,
        coco_map=dict(AV_COCO_TO_ONTOLOGY),
        openvocab_synonyms=dict(AV_OPENVOCAB_SYNONYMS),
        gate_policy=GatePolicy(safety_l1=AV_SAFETY_L1, rare_needs_review=True),
        disable_ego_hood_mask=False,
    )

    eval_strata = EvalStrataSpec(
        dimensions=tuple(StratumDimension(name=axis) for axis in _SCENE_AXES),
        protected_slices=("pedestrian_night", "autorickshaw_glare"),
        critical_class_names=AV_CRITICAL_CLASSES,
    )

    privacy = PrivacyPlaneSpec(
        redaction_targets=(
            RedactionTarget(name="face", detector="face"),
            RedactionTarget(name="plate", detector="plate"),
            # Text within a vehicle: the plate the plate detector missed. It is still a plate, it is
            # still legible, and the release attestation says the frame was redacted. Constrained by a
            # vehicle prior in services/anonymize/text_regions.py, because redacting every shop sign and
            # hoarding in an Indian street scene destroys the frame and protects nobody.
            RedactionTarget(name="text", detector="text"),
        ),
        legal_regime="DPDPA",
    )

    return Pack(
        manifest=manifest,
        ontology=ontology,
        safety_policy=make_safety_policy(onto, AV_SAFETY_L1, AV_CRITICAL_CLASSES),
        autolabel_profile=autolabel,
        eval_strata=eval_strata,
        quality_profile=_quality_profile(),
        forge_targets=_forge_targets(),
        privacy=privacy,
        relations=_relations(),
        cliques=_cliques(),
        # leaf -> l1 -> l0 -> root. The two governed levels made explicit, plus the root, so the gap
        # between "found a two-wheeler" and "named the right two-wheeler" is readable.
        class_tree=ClassTree(level_names=("leaf", "l1", "l0", "root")),
        motion_models=_motion_models(),
        context=_context(),
        track_events=_track_events(),
        region=_region(),
        scene_model=MovingCameraSceneModelFactory(),
        # The MCAP/CAN ingestion already lives in services/ingest (it fills vehicle_id + ego_speed); the AV
        # pack does not re-wrap it as an adapter yet. Sec ships the first IngestionAdapter (packs/sec).
        ingestion_adapters=(),
    )


PACK = _build()
