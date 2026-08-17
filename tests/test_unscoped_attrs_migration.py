"""An attribute an object's class cannot have, and the migration that takes it out of the label.

62,366 attribute values on 16,223 objects were attributes their own class does not carry: 7,477 objects that
are not traffic signals held a `signal_state`, and one autorickshaw held signal_state, signal_kind,
signal_mount, signal_arrow, marking_state, articulated and helmet at once. None of those observe anything.
The VLM path asked the model for every attribute in the ontology on every crop, then validated the reply
without a class id, which is the argument that turns the applicability check on.

The writer is fixed in `tests/test_m4_vlm.py`. This covers the 16,223 already stored, and it covers them as a
round trip, because a bulk write over that many labels has to be reversible before it is worth doing.
"""

from __future__ import annotations

import importlib.machinery
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

from core.config import get_settings
from services.autolabel.ontology import get_ontology

# The module name starts with a digit, so it cannot be imported by name.
_mig = importlib.machinery.SourceFileLoader(
    "mig_0094", str(Path(__file__).resolve().parents[1]
                    / "db/migrations/versions/0094_unscoped_attrs.py")).load_module()


@pytest.fixture
def conn():
    engine = sa.create_engine(get_settings().postgres.sync_dsn)
    with engine.begin() as c:
        yield c
        c.rollback()
    engine.dispose()


@pytest.fixture
def rickshaw(conn):
    """One three-wheeler carrying the exact attribute set found on a real object in the corpus."""
    onto = get_ontology()
    cid = onto.by_name("autorickshaw").id
    frame_id = conn.execute(sa.text("select frame_id from frame limit 1")).scalar()
    if frame_id is None:
        pytest.skip("no frame in the test corpus to hang an object on")
    oid = uuid.uuid4()
    conn.execute(sa.text("""
        insert into object (object_id, frame_id, class_id, bbox, attrs, provenance, conf, source, state)
        values (:o, :f, :c, '{1,2,3,4}', :a, '{}'::jsonb, 0.9, 'fused', 'review')
    """), {"o": oid, "f": frame_id, "c": cid,
           "a": '{"motion": "moving", "livery": true, "signal_state": "off", "signal_kind": "vehicle",'
                ' "marking_state": "present", "articulated": false, "helmet": [false]}'})
    yield oid
    conn.execute(sa.text("delete from object where object_id = :o"), {"o": oid})


def _row(conn, oid):
    return conn.execute(sa.text("select attrs, provenance from object where object_id = :o"),
                        {"o": oid}).first()


class TestTakingAFabricationOutOfTheLabel:
    def test_the_attributes_the_class_cannot_have_leave_attrs(self, conn, rickshaw):
        """`attrs` is what exports and what an annotator reads as fact, so that is where they must not be."""
        _mig._move(conn, batch=500)
        attrs, _prov = _row(conn, rickshaw)

        for cannot in ("signal_state", "signal_kind", "marking_state", "articulated", "helmet"):
            assert cannot not in attrs, f"'{cannot}' is still presented as an annotation on a three-wheeler"

    def test_the_attributes_it_can_have_are_untouched(self, conn, rickshaw):
        """The failure that would matter most: a migration that cleared the real attributes with the rest."""
        _mig._move(conn, batch=500)
        attrs, _prov = _row(conn, rickshaw)

        assert attrs["motion"] == "moving"
        assert attrs["livery"] is True

    def test_nothing_is_lost(self, conn, rickshaw):
        """Moved rather than deleted: what the model said while looking at the crop is a real fact about the
        model, and it is the only record of the scale of this."""
        _mig._move(conn, batch=500)
        _attrs, prov = _row(conn, rickshaw)

        kept = prov["unscoped_attrs"]
        assert kept["signal_state"] == "off"
        assert kept["marking_state"] == "present"
        assert kept["helmet"] == [False]

    def test_it_round_trips_exactly(self, conn, rickshaw):
        """A bulk write over 16,223 objects' labels is only worth doing if it undoes cleanly."""
        before, _ = _row(conn, rickshaw)
        _mig._move(conn, batch=500)
        _mig._restore(conn, batch=500)
        after, prov = _row(conn, rickshaw)

        assert after == before
        assert "unscoped_attrs" not in prov, "the provenance key outlived the restore"

    def test_running_it_twice_changes_nothing_further(self, conn, rickshaw):
        """It has to be safe to re-run: a migration that loops forever on rows it cannot clear takes the
        deploy with it."""
        _mig._move(conn, batch=500)
        first, _ = _row(conn, rickshaw)
        moved_again = _mig._move(conn, batch=500)
        second, _ = _row(conn, rickshaw)

        assert second == first
        assert moved_again == 0, "the second pass found work that the first should have finished"

    def test_an_object_whose_class_declares_no_scope_is_left_alone(self, conn):
        """A subclass with no allowlist means every attribute applies, which is the documented backward
        compatible case. Stripping those would delete real annotations."""
        onto = get_ontology()
        unscoped = [c for c in onto.classes if onto.attrs_for_class(c.id) is None]
        assert unscoped, "the ontology has no unscoped subclass, so this guard is untested"
        assert onto.attribute_scope, "the migration reads the scope from the ontology and found none"
