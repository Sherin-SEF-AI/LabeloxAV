"""Four guards that existed and did not fire, and one that another subsystem could lift.

Each of these was a control the codebase had already written and reasoned about, applied at some call
sites and not the one that mattered. They are grouped here because that is the shared shape, and because
none of them had a test that failed before the fix.
"""

from __future__ import annotations

import uuid

import pytest

from services.api.main import _is_sse_stream, _required_role
from services.review_policy import OBJECT_STATES, ReviewStateError, state_for

# The gate tests at the bottom need a real session_health table. The pure-policy classes above do not, but
# a module-level mark is the idiom here and the alternative is splitting one coherent story across files.
pytestmark = pytest.mark.db


class TestTheCredentialInAUrlReachesOnlyStreams:
    """EventSource cannot set a header, so a token in the query string is accepted - for four GETs.

    The exception used to be the whole /api/events/ prefix, which also holds PATCH and DELETE
    /api/events/{id} plus three ordinary JSON reads. A full-privilege token in a URL, where it reaches
    proxy and access logs, was authorising row deletion.
    """

    def test_the_four_streams_qualify(self):
        for path in ("/api/events/jobs", "/api/events/notifications", "/api/events/system"):
            assert _is_sse_stream(path), path

    def test_the_templated_training_stream_qualifies(self):
        assert _is_sse_stream(f"/api/events/training/{uuid.uuid4()}")

    def test_the_timeline_event_mutations_do_not(self):
        # PATCH and DELETE /api/events/{event_id} - services/api/routers/inertial.py
        assert not _is_sse_stream(f"/api/events/{uuid.uuid4()}")

    def test_the_plain_json_reads_do_not(self):
        for path in ("/api/events/search", "/api/events/taxonomy", "/api/events/corpus-summary"):
            assert not _is_sse_stream(path), path

    def test_a_route_added_under_the_prefix_later_does_not_inherit_it(self):
        # The reason this is an exact-path check and not a prefix one.
        assert not _is_sse_stream("/api/events/some-future-route")


class TestAnAnnotatorCannotWriteGroundTruth:
    """create_object wrote payload.state verbatim and it defaulted to "accepted".

    The clamp exists in review_policy and was only ever called from /api/review, which sits behind a
    reviewer floor - so it could never fire on the one path reachable from below. These pin the policy the
    route now applies; the route-level test lives in test_editor_api.py.
    """

    def test_an_annotator_asking_for_accepted_gets_submitted(self):
        assert state_for(None, "accepted", "annotator", None) == "submitted"

    def test_an_annotator_cannot_reach_auto_accept_either(self):
        assert state_for(None, "auto_accept", "annotator", None) == "submitted"

    def test_a_reviewer_is_not_clamped(self):
        assert state_for(None, "accepted", "reviewer", None) == "accepted"

    def test_an_unlisted_state_is_refused_rather_than_reaching_the_check_constraint(self):
        # This used to be written straight to the DB and surface as a 500 from ck_object_state.
        with pytest.raises(ReviewStateError):
            state_for(None, "gold", "admin", None)

    def test_the_default_the_route_falls_back_to_is_a_real_state(self):
        assert "submitted" in OBJECT_STATES


class TestTheFrameRouteSitsAtTheAnnotatorFloor:
    """Why the clamp is load-bearing: the middleware floor does not protect this path.

    /api/objects is a reviewer prefix, but object creation is POST /api/frames/{id}/objects, and
    /api/frames is not. Reading the floor from the live route table rather than asserting a constant, so
    this keeps telling the truth if the prefix lists move.
    """

    def test_object_create_is_only_annotator_floored(self):
        assert _required_role("/api/frames/abc/objects") == "annotator"

    def test_while_the_review_path_is_reviewer_floored(self):
        assert _required_role("/api/objects/abc/review") == "reviewer"


class TestOneSubsystemCannotLiftAnothersGate:
    """session_health is shared, and is_gated took the newest row from any writer.

    The inspector writes `inspector-idx-v1`; SANYX writes `sanyx-1` (and `sanyx-stream`) to the same table
    for its own purposes. A SANYX pass recorded after an inspector fail silently un-gated auto-labeling for
    that session - a fail-open across a module boundary, on the control that keeps a bad recording out of
    the labeling pipeline. There was no test for is_gated at all before this.
    """

    async def _session_with(self, db, rows):
        from db.models import Session as DbSession
        from db.models import SessionHealth
        from services.autolabel.ontology import get_ontology

        s = DbSession(session_id=uuid.uuid4(), vehicle_id="HEALTH-01", start_ts_ns=0, end_ts_ns=1,
                      city="BLR", sensors={}, ontology_version=get_ontology().version)
        db.add(s)
        await db.commit()
        for verdict, indexer in rows:
            db.add(SessionHealth(session_id=s.session_id, checks=[], verdict=verdict,
                                 indexer_version=indexer))
            # One commit per row, not one at the end. created_at defaults to func.now(), which in Postgres
            # is transaction-start time, so rows written in a single transaction share a timestamp exactly
            # and "the latest verdict" would be whichever the planner happened to return. Health checks
            # arrive as separate writes in reality; the fixture has to as well or it is testing a tie.
            await db.commit()
        return s.session_id

    async def test_an_inspector_fail_gates_the_session(self):
        from db.session import get_sessionmaker
        from services.inspector.health import is_gated

        async with get_sessionmaker()() as db:
            sid = await self._session_with(db, [("fail", "inspector-idx-v1")])
            assert await is_gated(db, sid) is True

    async def test_a_later_sanyx_pass_does_not_lift_it(self):
        from db.session import get_sessionmaker
        from services.inspector.health import is_gated

        async with get_sessionmaker()() as db:
            sid = await self._session_with(db, [("fail", "inspector-idx-v1"), ("pass", "sanyx-1")])
            assert await is_gated(db, sid) is True, (
                "a SANYX row lifted the inspector's gate; the gate is scoped to its own indexer_version")

    async def test_a_later_inspector_pass_does_lift_it(self):
        # The gate is not sticky - the inspector's own newer verdict is what clears it.
        from db.session import get_sessionmaker
        from services.inspector.health import is_gated

        async with get_sessionmaker()() as db:
            sid = await self._session_with(db, [("fail", "inspector-idx-v1"), ("pass", "inspector-idx-v1")])
            assert await is_gated(db, sid) is False

    async def test_a_session_only_sanyx_has_seen_is_not_gated_by_it(self):
        from db.session import get_sessionmaker
        from services.inspector.health import is_gated

        async with get_sessionmaker()() as db:
            sid = await self._session_with(db, [("fail", "sanyx-1")])
            assert await is_gated(db, sid) is False, (
                "SANYX quarantine is its own control with its own surface; it must not silently become the "
                "inspector's auto-label gate either")


class TestALanguageCommandCannotChangeWhatKindOfThingSomethingIs:
    """apply_edit reclassified without the l0 guard the other two bulk write paths already had.

    refuse_reason exists because one agent run moved 1,047 buses into a bus shelter at confidence 0.989.
    A typed sentence is if anything a looser filter than a detector's confidence, so it needed the guard
    more, not less - and it also did not exclude source == "human", so one command could overwrite a
    person's work. Both are fixed in services/agent/nl_edit.py; this pins the boundary itself.
    """

    def test_a_fallback_object_cannot_become_street_furniture(self):
        from services.agent.class_move import refuse_reason
        from services.autolabel.ontology import get_ontology

        onto = get_ontology()
        src = onto.fallback_ids()[0]
        # push_cart is l0 infra in the governed ontology, alongside cone and temp_barricade; the fallback
        # classes are l0 object. That is exactly the move this guard exists to refuse.
        reason = refuse_reason(onto, src, onto.by_name("push_cart").id)
        assert reason is not None and "l0" in reason.lower()

    def test_a_refinement_within_the_same_kind_is_allowed(self):
        from services.agent.class_move import refuse_reason
        from services.autolabel.ontology import get_ontology

        onto = get_ontology()
        src = onto.fallback_ids()[0]
        same_l0 = next(c for c in onto.classes
                       if c.l0 == onto.by_id(src).l0 and c.id != src and c.l1 != "fallback")
        assert refuse_reason(onto, src, same_l0.id) is None
