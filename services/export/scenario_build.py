"""Turning a recorded event into an OpenSCENARIO document: reading the corpus, deciding what belongs.

The XML lives in adapter_openscenario.py, which is pure. This is the part that has to make judgements about
real data, and the judgements are what make the output honest or misleading.

**Which actors belong in the scenario.** Everything visible is the wrong answer: a busy Bangalore junction
has forty tracked objects and a scenario with forty actors is not reproducible, it is a traffic simulation
with the interesting part buried. Only tracks that are actually near the ego during the window are included,
by a distance gate, and the count is reported so a customer can see what was left out.

**Where the trajectory comes from.** `object_dynamics` carries distance_m and lateral_m per object per
frame, from flat-road monocular IPM. Those are the only positions this corpus has, so they are what the
scenario uses, and the resulting document says so in its own header rather than presenting derived metres
as measurements.

**What a track with two points is worth.** Nothing: a FollowTrajectoryAction needs a path, and two poses
several hundred milliseconds apart describe a line segment rather than a manoeuvre. Such tracks are dropped
and counted, not padded out with interpolation that would invent the shape of the thing being sold.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectDynamics, OntologyClass, TimelineEvent
from db.models import Session as DbSession
from services.export.adapter_openscenario import Actor, TrajectoryPoint, build_scenario

log = get_logger("export.scenario_build")

# How near an actor has to come to the ego to be part of the story. Beyond this it is scenery: present in
# the recording, irrelevant to what happened, and expensive to reproduce.
DEFAULT_NEAR_M = 60.0

# Fewer poses than this is a line segment, not a manoeuvre.
MIN_POSES = 3

# The ontology L1 groups whose members can move. Everything else the corpus tracks is roadside furniture:
# poles, hoardings, street lights, signs, buildings, barriers.
#
# This is not a nicety. Without it a twelve-second window of one Bangalore session produced a scenario with
# 316 actors, among them advertisement_board, cctv_pole and street_light, each carrying a
# FollowTrajectoryAction describing how a lamp post appeared to move as the ego drove past it. That document
# is enormous, meaningless, and would look authoritative to a customer.
#
# Chosen by L1 rather than by l0="object" because that l0 also holds `custom` (electric_post, fly_over,
# buildings, side_wall) which is static, and `fallback`. vehicle_fallback is admitted because an
# unidentified vehicle is still a vehicle; object_fallback is not, because an actor of unknown kind cannot
# be given a sensible category or footprint and a simulator would have to guess.
MOVABLE_L1 = frozenset({"animal", "four_wheeler", "heavy", "three_wheeler", "two_wheeler", "vru"})
MOVABLE_EXTRA_NAMES = frozenset({"vehicle_fallback"})

# How much recording to take either side of a point event, when the event has no end.
DEFAULT_PAD_S = 4.0


async def build_from_event(db: AsyncSession, event_id: str, *, near_m: float = DEFAULT_NEAR_M,
                           pad_s: float = DEFAULT_PAD_S, road_network_file: str = "map.xodr") -> dict:
    """One scenario for one timeline event: the window around it, and the actors that were near.

    Returns the document plus what was included and excluded, because a scenario that silently dropped the
    other vehicle is worse than one that refuses: it looks complete.
    """
    ev = await db.get(TimelineEvent, _uuid.UUID(event_id))
    if ev is None:
        return {"error": "event not found", "event_id": event_id}

    t0 = int(ev.t_start_ns) - int(pad_s * 1e9)
    t1 = int(ev.t_end_ns or ev.t_start_ns) + int(pad_s * 1e9)
    result = await build_from_window(db, session_id=str(ev.session_id), t_start_ns=t0, t_end_ns=t1,
                                     near_m=near_m, road_network_file=road_network_file,
                                     name=f"{ev.kind}_{str(ev.event_id)[:8]}",
                                     description=f"Mined event '{ev.kind}' (confidence {ev.conf}).")
    result["event"] = {"event_id": str(ev.event_id), "kind": ev.kind,
                       "t_start_ns": ev.t_start_ns, "t_end_ns": ev.t_end_ns}
    return result


async def build_from_window(db: AsyncSession, *, session_id: str, t_start_ns: int, t_end_ns: int,
                            near_m: float = DEFAULT_NEAR_M, road_network_file: str = "map.xodr",
                            name: str = "scenario", description: str = "") -> dict:
    """A scenario from an arbitrary time window of one session."""
    sess = await db.get(DbSession, _uuid.UUID(session_id))
    if sess is None:
        return {"error": "session not found", "session_id": session_id}

    rows = (await db.execute(
        select(ObjectDynamics.track_id, ObjectDynamics.ts_ns, ObjectDynamics.distance_m,
               ObjectDynamics.lateral_m, ObjectDynamics.heading_deg, ObjectDynamics.speed_kmh,
               OntologyClass.name)
        .join(Object, Object.object_id == ObjectDynamics.object_id)
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        # Scoped through the frame, which is what carries the session. Joining Session directly on a
        # constant would be a cross join with a true predicate: every row in the time window from every
        # session in the corpus, silently mixed into one scenario.
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id == _uuid.UUID(session_id),
               ObjectDynamics.ts_ns >= t_start_ns, ObjectDynamics.ts_ns <= t_end_ns,
               ObjectDynamics.track_id.isnot(None),
               ObjectDynamics.distance_m.isnot(None))
        .order_by(ObjectDynamics.track_id, ObjectDynamics.ts_ns))).all()

    by_track: dict[str, dict] = {}
    for track_id, ts_ns, dist, lat, heading, speed_kmh, class_name in rows:
        t = by_track.setdefault(str(track_id), {"class_name": class_name, "points": []})
        t["points"].append(TrajectoryPoint(
            t_s=round((int(ts_ns) - t_start_ns) / 1e9, 3),
            x=float(dist),
            # lateral_m is signed left-positive already, matching the OpenSCENARIO y axis, so no flip.
            y=float(lat or 0.0),
            heading_rad=_deg_to_rad(heading),
            speed_mps=(float(speed_kmh) / 3.6) if speed_kmh is not None else None,
        ))

    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    def _movable(class_name: str) -> bool:
        if class_name in MOVABLE_EXTRA_NAMES:
            return True
        try:
            return onto.by_name(class_name).l1 in MOVABLE_L1
        except Exception:  # noqa: BLE001
            return False

    actors: list[Actor] = []
    too_short = too_far = static = 0
    for track_id, t in by_track.items():
        pts = t["points"]
        if not _movable(t["class_name"]):
            static += 1
            continue
        if len(pts) < MIN_POSES:
            too_short += 1
            continue
        if min(abs(p.x) for p in pts) > near_m:
            too_far += 1
            continue
        actors.append(Actor(name=f"actor_{track_id[:8]}", class_name=t["class_name"],
                            track_id=track_id, points=pts))

    if not actors:
        return {"error": "no movable actor in this window has enough poses near the ego to reproduce",
                "tracks_seen": len(by_track), "dropped_static": static,
                "dropped_too_short": too_short, "dropped_too_far": too_far}

    # Ego sits at the origin of its own frame for the whole window. Not an approximation to apologise for:
    # every actor position here is already measured relative to ego, so the ego trajectory in this frame is
    # the origin by construction. A simulator that wants ego to move replays its own speed profile, which
    # is why EgoInitialSpeed is a declared parameter.
    duration_s = round((t_end_ns - t_start_ns) / 1e9, 3)
    ego = Actor(name="Ego", class_name="sedan", points=[
        TrajectoryPoint(t_s=0.0, x=0.0, y=0.0),
        TrajectoryPoint(t_s=duration_s, x=0.0, y=0.0),
    ])

    xml = build_scenario(
        name=name, actors=actors, ego=ego, road_network_file=road_network_file,
        description=(f"{description} Session {session_id}, vehicle {sess.vehicle_id}, "
                     f"{len(actors)} actor(s) over {duration_s}s.").strip(),
        # Stamped from the recording rather than the clock, so the same window always exports the same
        # bytes and the document can be content-addressed like every other export here.
        date=datetime.fromtimestamp(t_start_ns / 1e9, tz=UTC).isoformat(),
    )

    log.info("scenario.built", session=session_id, actors=len(actors), tracks=len(by_track),
             static=static, too_short=too_short, too_far=too_far)
    return {
        "name": name, "xml": xml, "duration_s": duration_s,
        "actors": [{"name": a.name, "class_name": a.class_name, "track_id": a.track_id,
                    "poses": len(a.points)} for a in actors],
        # Reported, not silently applied. A scenario that dropped the other vehicle is worse than one that
        # refused, because it looks complete.
        "excluded": {"static": static, "too_short": too_short, "too_far": too_far, "near_m": near_m,
                     "note": (f"{static} track(s) were roadside furniture rather than actors, "
                              f"{too_short} had fewer than {MIN_POSES} poses, and {too_far} stayed beyond "
                              f"{near_m}m. All are omitted rather than interpolated or written out as "
                              f"lamp posts following trajectories")},
        "road_network_file": road_network_file,
    }


def _deg_to_rad(heading_deg: float | None) -> float:
    import math

    return 0.0 if heading_deg is None else round(math.radians(float(heading_deg)), 6)
