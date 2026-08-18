"""OpenSCENARIO export: handing a mined event to a simulator without overstating what it is.

The map half of sim handover already existed. What did not was any way to get the event out, so a cut-in
with an autorickshaw stayed in the database. These tests pin the two ways this could be quietly wrong: a
document that claims to be a parameterised scenario when it is a replay, and a document full of lamp posts
following trajectories.
"""

from __future__ import annotations

import asyncio
import uuid
import xml.etree.ElementTree as ET

import pytest

from core.config import get_settings
from services.export.adapter_openscenario import (
    Actor,
    TrajectoryPoint,
    build_scenario,
    category_for,
)

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


def _scenario(**kw) -> ET.Element:
    a = Actor(name="actor_1", class_name="autorickshaw", points=[
        TrajectoryPoint(0.0, 22.0, -3.4, 0.10, 8.0),
        TrajectoryPoint(0.5, 19.5, -2.1, 0.20, 7.6),
        TrajectoryPoint(1.0, 17.4, -0.6, 0.25, 7.2),
    ])
    ego = Actor(name="Ego", class_name="sedan",
                points=[TrajectoryPoint(0.0, 0, 0), TrajectoryPoint(1.0, 0, 0)])
    return ET.fromstring(build_scenario(name="cutin", actors=[a], ego=ego, **kw))


# --- the document ------------------------------------------------------------------------------


def test_the_document_is_well_formed_openscenario_1_2():
    root = _scenario()
    assert root.tag == "OpenSCENARIO"
    header = root.find("FileHeader")
    assert header.get("revMajor") == "1" and header.get("revMinor") == "2"
    assert root.find("RoadNetwork/LogicFile") is not None
    assert root.find("Storyboard/Init") is not None
    assert root.find("Storyboard/Story") is not None


def test_the_header_says_it_is_a_replay_rather_than_a_parameterised_scenario():
    """The claim that would be dishonest to leave out.

    Turning one observed cut-in into a parameterised scenario means inventing which parts of the manoeuvre
    were incidental, and the recording does not say. A customer sweeping parameters needs to know they are
    sweeping assumptions, not measurements.
    """
    desc = _scenario().find("FileHeader").get("description")
    assert "Recorded replay" in desc
    assert "monocular" in desc and "planar" in desc


def test_the_trajectory_is_written_out_pose_by_pose():
    root = _scenario()
    vertices = root.findall(".//ManeuverGroup[@name='actor_1_group']//Vertex")
    assert len(vertices) == 3
    first = vertices[0].find("Position/WorldPosition")
    assert float(first.get("x")) == 22.0 and float(first.get("y")) == -3.4


def test_following_mode_is_position_not_follow():
    """The recording says where the actor was, not what throttle and steering produced it. Asking a
    simulator to reproduce the poses dynamically substitutes its vehicle model's opinion for the
    observation."""
    mode = _scenario().find(".//TrajectoryFollowingMode")
    assert mode.get("followingMode") == "position"


def test_the_same_window_always_produces_the_same_bytes():
    """Every other export path here seals a content-addressed commit id. A document that differs on each
    run because it stamped the wall clock cannot be one of them."""
    a = build_scenario(name="s", actors=[Actor("a", "sedan", points=[
        TrajectoryPoint(0.0, 1, 0), TrajectoryPoint(1.0, 2, 0)])], date="2020-01-01T00:00:00")
    b = build_scenario(name="s", actors=[Actor("a", "sedan", points=[
        TrajectoryPoint(0.0, 1, 0), TrajectoryPoint(1.0, 2, 0)])], date="2020-01-01T00:00:00")
    assert a == b


# --- categories --------------------------------------------------------------------------------


def test_india_specific_classes_map_to_the_nearest_standard_category():
    """OpenSCENARIO has a short closed list of categories and no autorickshaw. Mapping is unavoidable; doing
    it silently is what would not be."""
    assert category_for("autorickshaw") == ("Vehicle", "car")
    assert category_for("cycle_rickshaw") == ("Vehicle", "bicycle")
    assert category_for("bus") == ("Vehicle", "bus")


def test_a_pedestrian_is_a_pedestrian_and_an_animal_is_not_a_vehicle():
    """A cow modelled as a car behaves like a car in any simulator that reads the category."""
    assert category_for("pedestrian") == ("Pedestrian", "pedestrian")
    kind, _ = category_for("cattle")
    assert kind == "MiscObject"


def test_an_unknown_class_falls_back_to_a_car_rather_than_failing():
    assert category_for("something_new") == ("Vehicle", "car")


# --- what belongs in a scenario -----------------------------------------------------------------


def test_roadside_furniture_is_not_an_actor():
    """The finding that made the first real export useless.

    A twelve-second window of one Bangalore session produced 316 actors, among them advertisement_board,
    cctv_pole and street_light, each carrying a FollowTrajectoryAction describing how a lamp post appeared
    to move as the ego drove past it. Enormous, meaningless, and authoritative-looking.
    """
    from services.autolabel.ontology import get_ontology
    from services.export.scenario_build import MOVABLE_EXTRA_NAMES, MOVABLE_L1

    onto = get_ontology()

    def movable(name: str) -> bool:
        return name in MOVABLE_EXTRA_NAMES or onto.by_name(name).l1 in MOVABLE_L1

    for static in ("street_light", "hoarding", "traffic_sign", "pole", "buildings", "barrier"):
        assert not movable(static), f"{static} is furniture, not an actor"
    for moving in ("pedestrian", "autorickshaw", "bus", "cattle", "motorcycle", "rider"):
        assert movable(moving), f"{moving} moves and belongs in the scenario"


def test_an_object_of_unknown_kind_is_excluded_but_an_unknown_vehicle_is_not():
    """An actor whose kind nobody knows cannot be given a category or a footprint, so a simulator would
    have to guess. An unidentified vehicle is still a vehicle."""
    from services.export.scenario_build import MOVABLE_EXTRA_NAMES

    assert "vehicle_fallback" in MOVABLE_EXTRA_NAMES
    assert "object_fallback" not in MOVABLE_EXTRA_NAMES


@requires_infra
def test_a_window_scoped_to_one_session_does_not_pull_in_another():
    """The join in the first draft was against Session on a constant, which is a cross join with a true
    predicate: every dynamics row in the time window from every session in the corpus, silently merged."""
    import inspect

    from services.export import scenario_build

    src = inspect.getsource(scenario_build.build_from_window)
    assert "Frame.session_id ==" in src, "the window has to be scoped through the frame"


@requires_infra
def test_a_window_with_nothing_reproducible_refuses_rather_than_emitting_an_empty_scenario():
    from db.session import get_sessionmaker
    from services.export.scenario_build import build_from_window

    async def _flow():
        from core.timebase import now_ns, seconds_to_ns
        from db.models import Session as DbSession
        from services.autolabel.ontology import get_ontology

        sid, ts = uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                             ontology_version=get_ontology().version))
            await db.commit()
        async with get_sessionmaker()() as db:
            r = await build_from_window(db, session_id=str(sid), t_start_ns=ts, t_end_ns=ts + 10**10)
        assert "error" in r and "no movable actor" in r["error"]

    run_async(_flow())
