"""ASAM OpenSCENARIO 1.2 export: a mined event, handed over as something a simulator can run.

The map half of sim handover already exists. `services/hdmap/export.py` emits OpenDRIVE and Lanelet2 from
the fused HD map, so a customer can already load the road. What they cannot do is load what happened on it.
A cut-in at an unprotected right turn with an autorickshaw is the thing an AV planning team pays a multiple
for, and it has been sitting in `timeline_event` and `object_dynamics` with no way out.

**What this produces is a replay, and the header says so.** Turning an observed cut-in into a parameterised
scenario (vary the gap, vary the approach speed, sweep the aggression) requires assumptions the recording
does not contain: nothing in the data says which parts of the manoeuvre were incidental and which were
essential. Emitting a parameterised scenario from a single observation would be inventing the parameters and
presenting them as measurements. So the trajectory is written out faithfully as a FollowTrajectoryAction and
the parameters a customer is likely to want to sweep are declared in ParameterDeclarations, ready to be
varied by someone who knows what they mean.

**The coordinates are ego-relative and monocular.** `object_dynamics` positions come from flat-road IPM on a
single camera, whose error grows with the square of distance and which assumes the road is a plane. That is
adequate for reconstructing a manoeuvre and inadequate for anything claiming survey accuracy, so the
scenario carries the method and the caveat in its description rather than presenting derived metres as
though they were measured ones.

The document validates against the 1.2 schema for the subset it uses: FileHeader, ParameterDeclarations,
RoadNetwork, Entities, and a Storyboard with Init plus one Story.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from xml.dom import minidom

from core.logging import get_logger

log = get_logger("export.openscenario")

OSC_MAJOR, OSC_MINOR = 1, 2

# Ontology class to OpenSCENARIO vehicleCategory / pedestrianCategory. The categories are a short closed
# list in the standard, so the India-specific classes have to be mapped onto the nearest thing a simulator
# understands: an autorickshaw is not in the standard and is closest to a car in dynamics and footprint.
# Recorded explicitly rather than defaulted, because a silent fallback to "car" for a cow is the kind of
# thing that survives into a customer's regression suite.
VEHICLE_CATEGORY = {
    "sedan": "car", "hatchback": "car", "suv": "car", "taxi": "car", "car": "car",
    "autorickshaw": "car", "e_auto": "car", "tempo": "van", "van": "van",
    "bus": "bus", "minibus": "bus", "truck": "truck", "tractor": "truck", "trailer": "trailer",
    "motorcycle": "motorbike", "scooter": "motorbike", "rider": "motorbike",
    "bicycle": "bicycle", "cycle": "bicycle", "cycle_rickshaw": "bicycle",
}
PEDESTRIAN_CATEGORY = {"pedestrian": "pedestrian", "child": "pedestrian"}
ANIMAL_CLASSES = {"cattle", "cow", "buffalo", "dog", "goat", "animal"}


@dataclass
class TrajectoryPoint:
    """One observed pose of an actor, in the ego frame at the scenario's start.

    x is forward from ego, y is left. Both come from `object_dynamics`, so both inherit its monocular
    flat-road assumption.
    """
    t_s: float
    x: float
    y: float
    heading_rad: float = 0.0
    speed_mps: float | None = None


@dataclass
class Actor:
    name: str
    class_name: str
    track_id: str | None = None
    points: list[TrajectoryPoint] = field(default_factory=list)


def category_for(class_name: str) -> tuple[str, str]:
    """(entity_kind, category) for an ontology class.

    entity_kind is which OpenSCENARIO element to emit: Vehicle, Pedestrian or MiscObject. Animals are
    MiscObject because the standard has no animal entity, and a cow modelled as a car would behave like one
    in any simulator that reads the category.
    """
    n = (class_name or "").lower()
    if n in PEDESTRIAN_CATEGORY:
        return "Pedestrian", PEDESTRIAN_CATEGORY[n]
    if n in ANIMAL_CLASSES:
        return "MiscObject", "obstacle"
    return "Vehicle", VEHICLE_CATEGORY.get(n, "car")


def _speed_from_points(points: list[TrajectoryPoint]) -> float:
    """Initial speed in m/s: the recorded one if present, else differenced from the first two poses.

    Differencing is the fallback rather than the default because a recorded speed carries the ego-speed
    correction and a differenced one does not; using it when a real value exists would throw that away.
    """
    if points and points[0].speed_mps is not None:
        return max(0.0, float(points[0].speed_mps))
    if len(points) >= 2:
        dt = points[1].t_s - points[0].t_s
        if dt > 1e-6:
            d = math.hypot(points[1].x - points[0].x, points[1].y - points[0].y)
            return round(d / dt, 3)
    return 0.0


def _sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        if v is not None:
            el.set(k, str(v))
    return el


def _entity(entities: ET.Element, actor: Actor) -> None:
    obj = _sub(entities, "ScenarioObject", name=actor.name)
    kind, category = category_for(actor.class_name)

    if kind == "Vehicle":
        veh = _sub(obj, "Vehicle", name=actor.class_name, vehicleCategory=category)
        _bounding_box(veh, actor.class_name)
        _sub(veh, "Performance", maxSpeed=70, maxAcceleration=5, maxDeceleration=9)
        axles = _sub(veh, "Axles")
        _sub(axles, "FrontAxle", maxSteering=0.5, wheelDiameter=0.6, trackWidth=1.6,
             positionX=2.8, positionZ=0.3)
        _sub(axles, "RearAxle", maxSteering=0.0, wheelDiameter=0.6, trackWidth=1.6,
             positionX=0.0, positionZ=0.3)
    elif kind == "Pedestrian":
        ped = _sub(obj, "Pedestrian", name=actor.class_name, pedestrianCategory=category, mass=80)
        _bounding_box(ped, actor.class_name)
    else:
        misc = _sub(obj, "MiscObject", name=actor.class_name, miscObjectCategory=category, mass=400)
        _bounding_box(misc, actor.class_name)


# Nominal footprints in metres. Approximate on purpose and marked as such: the corpus records 2D image
# boxes, not physical extents, so any figure here is a class-level prior rather than a measurement of the
# object in this recording.
_FOOTPRINT = {
    "pedestrian": (0.6, 0.6, 1.7), "child": (0.5, 0.5, 1.2),
    "motorcycle": (2.0, 0.8, 1.4), "scooter": (1.8, 0.7, 1.3), "rider": (2.0, 0.8, 1.6),
    "bicycle": (1.7, 0.6, 1.4), "cycle": (1.7, 0.6, 1.4), "cycle_rickshaw": (2.6, 1.0, 1.6),
    "autorickshaw": (2.6, 1.4, 1.7), "e_auto": (2.6, 1.4, 1.7),
    "bus": (11.0, 2.6, 3.2), "minibus": (7.0, 2.3, 2.8), "truck": (8.0, 2.5, 3.0),
    "cattle": (2.2, 0.8, 1.5), "cow": (2.2, 0.8, 1.5), "buffalo": (2.4, 0.9, 1.5),
    "dog": (0.9, 0.3, 0.6),
}


def _bounding_box(parent: ET.Element, class_name: str) -> None:
    length, width, height = _FOOTPRINT.get((class_name or "").lower(), (4.5, 1.8, 1.5))
    bb = _sub(parent, "BoundingBox")
    _sub(bb, "Center", x=round(length / 2, 2), y=0, z=round(height / 2, 2))
    _sub(bb, "Dimensions", width=width, length=length, height=height)


def build_scenario(*, name: str, actors: list[Actor], ego: Actor | None = None,
                   road_network_file: str = "map.xodr", description: str = "",
                   author: str = "LabeloxAV", date: str = "1970-01-01T00:00:00") -> str:
    """One OpenSCENARIO 1.2 document for a recorded event.

    `date` is a parameter rather than read from the clock so the same inputs produce the same bytes. An
    export that differs on every run cannot be content-addressed, and every other export path here seals a
    commit id over its contents.
    """
    root = ET.Element("OpenSCENARIO")

    caveat = ("Recorded replay, not a parameterised scenario: actor poses are observed, not synthesised. "
              "Positions are ego-relative and derived by flat-road monocular IPM, so lateral and "
              "longitudinal error grow with distance and the road is assumed planar.")
    _sub(root, "FileHeader", revMajor=OSC_MAJOR, revMinor=OSC_MINOR, date=date,
         description=f"{description} {caveat}".strip(), author=author)

    # Declared so a customer can sweep what they understand, rather than this file pretending to know which
    # quantities were incidental to the manoeuvre.
    params = _sub(root, "ParameterDeclarations")
    _sub(params, "ParameterDeclaration", name="EgoInitialSpeed", parameterType="double",
         value=_speed_from_points(ego.points) if ego else 0.0)
    _sub(params, "ParameterDeclaration", name="TimeScale", parameterType="double", value=1.0)

    _sub(root, "CatalogLocations")
    road = _sub(root, "RoadNetwork")
    _sub(road, "LogicFile", filepath=road_network_file)

    entities = _sub(root, "Entities")
    all_actors = ([ego] if ego else []) + list(actors)
    for a in all_actors:
        _entity(entities, a)

    storyboard = _sub(root, "Storyboard")
    init = _sub(storyboard, "Init")
    init_actions = _sub(init, "Actions")
    for a in all_actors:
        _init_private(init_actions, a)

    story = _sub(storyboard, "Story", name=f"{name}_story")
    act = _sub(story, "Act", name=f"{name}_act")
    for a in all_actors:
        if len(a.points) >= 2:
            _maneuver_group(act, a)
    _act_start_trigger(act)

    stop = _sub(storyboard, "StopTrigger")
    cond_group = _sub(_sub(stop, "ConditionGroup"), "Condition", name="end_of_recording",
                      delay=0, conditionEdge="rising")
    by_value = _sub(cond_group, "ByValueCondition")
    _sub(by_value, "SimulationTimeCondition", value=round(_duration(all_actors), 3), rule="greaterThan")

    return _pretty(root)


def _duration(actors: list[Actor]) -> float:
    return max((a.points[-1].t_s for a in actors if a.points), default=0.0)


def _init_private(actions: ET.Element, actor: Actor) -> None:
    private = _sub(actions, "Private", entityRef=actor.name)
    p0 = actor.points[0] if actor.points else TrajectoryPoint(0.0, 0.0, 0.0)

    teleport = _sub(_sub(private, "PrivateAction"), "TeleportAction")
    _sub(_sub(teleport, "Position"), "WorldPosition",
         x=round(p0.x, 3), y=round(p0.y, 3), z=0, h=round(p0.heading_rad, 4))

    speed_action = _sub(_sub(_sub(private, "PrivateAction"), "LongitudinalAction"), "SpeedAction")
    _sub(speed_action, "SpeedActionDynamics", dynamicsShape="step", dynamicsDimension="time", value=0)
    target = _sub(speed_action, "SpeedActionTarget")
    _sub(target, "AbsoluteTargetSpeed", value=round(_speed_from_points(actor.points), 3))


def _maneuver_group(act: ET.Element, actor: Actor) -> None:
    """One actor's observed path, as a FollowTrajectoryAction.

    followingMode is "position" rather than "follow": the recording says where the actor was, not what
    throttle and steering produced it, and asking a simulator to reproduce those poses dynamically would
    substitute its vehicle model's opinion for the observation.
    """
    group = _sub(act, "ManeuverGroup", maximumExecutionCount=1, name=f"{actor.name}_group")
    actors_el = _sub(group, "Actors", selectTriggeringEntities="false")
    _sub(actors_el, "EntityRef", entityRef=actor.name)

    maneuver = _sub(group, "Maneuver", name=f"{actor.name}_replay")
    event = _sub(maneuver, "Event", name=f"{actor.name}_follow", priority="overwrite",
                 maximumExecutionCount=1)
    action = _sub(event, "Action", name=f"{actor.name}_trajectory")
    routing = _sub(_sub(action, "PrivateAction"), "RoutingAction")
    follow = _sub(routing, "FollowTrajectoryAction")

    traj = _sub(follow, "Trajectory", name=f"{actor.name}_path", closed="false")
    _sub(traj, "ParameterDeclarations")
    shape = _sub(traj, "Shape")
    polyline = _sub(shape, "Polyline")
    for p in actor.points:
        vertex = _sub(polyline, "Vertex", time=round(p.t_s, 3))
        _sub(_sub(vertex, "Position"), "WorldPosition",
             x=round(p.x, 3), y=round(p.y, 3), z=0, h=round(p.heading_rad, 4))

    # Relative timing, so a scenario dropped into a longer sequence starts when its act does rather than at
    # the wall-clock offset it happened to be recorded at.
    _sub(_sub(follow, "TimeReference"), "Timing",
         domainAbsoluteRelative="relative", scale=1.0, offset=0.0)
    _sub(follow, "TrajectoryFollowingMode", followingMode="position")

    start = _sub(event, "StartTrigger")
    cond = _sub(_sub(start, "ConditionGroup"), "Condition", name=f"{actor.name}_start",
                delay=0, conditionEdge="rising")
    _sub(_sub(cond, "ByValueCondition"), "SimulationTimeCondition", value=0, rule="greaterThan")


def _act_start_trigger(act: ET.Element) -> None:
    start = _sub(act, "StartTrigger")
    cond = _sub(_sub(start, "ConditionGroup"), "Condition", name="act_start", delay=0,
                conditionEdge="rising")
    _sub(_sub(cond, "ByValueCondition"), "SimulationTimeCondition", value=0, rule="greaterThan")


def _pretty(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")
