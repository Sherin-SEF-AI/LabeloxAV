"""A relabel run moved 1,047 buses into a bus shelter, at confidence 0.989.

The provenance of one of them, verbatim: `agent_relabel: ["bus -> bmtc_bus_shelter (0.989)"]`, written over
the top of a detector saying `bus` at 0.917, an open-vocabulary path saying `bus` at 0.803, and a reasoner
recording "aspect 0.81 is typical for bus", "the track agrees on bus across 6 neighbouring frames" and "2
independent paths agree on bus". Across the run, 2,023 objects crossed the `l0` boundary and 2,986 moved from
a countable thing to uncountable stuff.

The ontology already forbids the result: `bmtc_bus_shelter` is in `STUFF_NAMES`, so the fusion step of the
same runner drops stuff detections before they are ever persisted as instances. A confidence threshold
cannot catch a confident wrong answer, and this one arrived at 0.989. The check is structural instead.
"""

from __future__ import annotations

import pytest

from services.agent.class_move import is_refinement, refuse_reason
from services.autolabel.ontology import get_ontology


@pytest.fixture(scope="module")
def onto():
    return get_ontology()


def _id(onto, name: str) -> int:
    return onto.by_name(name).id


class TestTheReportedMoves:
    @pytest.mark.parametrize("src", ["bus", "traffic_sign", "hoarding"])
    def test_nothing_becomes_a_bus_shelter(self, onto, src):
        """The three lineages that contaminated one class: 1,047 from bus, 708 from traffic_sign, 522 from
        hoarding."""
        reason = refuse_reason(onto, _id(onto, src), _id(onto, "bmtc_bus_shelter"))
        assert reason is not None
        assert "stuff" in reason

    def test_a_vehicle_does_not_become_a_toll_booth_either(self, onto):
        # 62 of these in the same run.
        assert refuse_reason(onto, _id(onto, "bus"), _id(onto, "toll_booth")) is not None


class TestRefinementsStillPass:
    @pytest.mark.parametrize("src,dst", [
        ("sedan", "mpv"),            # 8,435 in the run: a body-style refinement, exactly what relabel is for
        ("sedan", "hatchback"),      # 8,394
        ("truck", "container_truck"),  # 2,485
        ("motorcycle", "moped"),     # 1,259
        ("bus", "tempo"),            # 831: wrong perhaps, but a vehicle either way, so not this check's job
    ])
    def test_a_class_may_be_sharpened(self, onto, src, dst):
        assert is_refinement(onto, _id(onto, src), _id(onto, dst)), \
            "a refinement inside one kind of thing is what a relabel exists to do"


class TestTheBoundaries:
    def test_crossing_l0_is_refused_even_between_two_things(self, onto):
        # `hoarding` is infra and `advertisement_board` is an object; 149 of these happened.
        reason = refuse_reason(onto, _id(onto, "hoarding"), _id(onto, "advertisement_board"))
        assert reason is not None and "l0" in reason

    def test_stuff_to_thing_is_refused_in_that_direction_too(self, onto):
        # Symmetric on purpose: a relabeller that can invent instances out of uncountable stuff produces
        # boxes around road surface, which is the same error read backwards.
        reason = refuse_reason(onto, _id(onto, "vegetation"), _id(onto, "bus"))
        assert reason is not None

    def test_the_reason_names_the_boundary_it_crossed(self, onto):
        """A skipped proposal that cannot say why is an unexplained shortfall in a count."""
        reason = refuse_reason(onto, _id(onto, "bus"), _id(onto, "bmtc_bus_shelter"))
        assert reason and reason != "refused"


class TestEdges:
    def test_a_class_moving_to_itself_is_not_a_move(self, onto):
        assert refuse_reason(onto, _id(onto, "bus"), _id(onto, "bus")) is None

    def test_an_unknown_class_is_not_this_check_s_business(self, onto):
        """The caller already drops proposals it cannot store. Reporting a missing class as a forbidden move
        would confuse two different failures and hide the real one."""
        assert refuse_reason(onto, _id(onto, "bus"), 999_999) is None
        assert refuse_reason(onto, 999_999, _id(onto, "bus")) is None

    def test_every_class_can_move_to_something(self, onto):
        """A guard that refuses everything is a guard that gets turned off. Sampled across the ontology,
        each class keeps at least one legal target within its own kind."""
        classes = [c for c in onto.classes][:40]
        for c in classes:
            peers = [d for d in onto.classes if d.id != c.id and d.l0 == c.l0
                     and onto.is_stuff(d.id) == onto.is_stuff(c.id)]
            if peers:
                assert is_refinement(onto, c.id, peers[0].id)
