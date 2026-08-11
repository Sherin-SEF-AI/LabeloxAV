"""The operations panel started returning 500 the moment the corpus relabel finished.

`measure_operation` collects every object a kind's committed runs have touched and asks for their reviews in
one `IN` clause. asyncpg refuses a statement carrying more than 32,767 bound parameters, and `relabel` now has
34,067 committed child runs, one per frame, from the corpus pass. So:

  sqlalchemy.exc.InterfaceError: the number of query arguments cannot exceed 32767

It is worth naming what kind of bug this is. Nothing here was wrong when it was written; it broke because the
data got bigger, and it broke at exactly the moment the system started working properly. The endpoint has no
test that runs at corpus scale, so the first thing to notice was a 500 in a browser console.

The second test is the one that matters more than the count. Batching is an artefact of a driver limit, and
the caller keeps the *first* human verdict per object. If each batch is treated in isolation, "first" quietly
becomes "first within whichever batch happened to arrive", so the same data could measure differently
depending on a chunk size nobody thinks of as a parameter.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns
from db.models import AgentRun, Frame, Object, Review
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.training.op_precision import _chunks, measure_operation

pytestmark = pytest.mark.db


def test_chunking_covers_everything_in_order():
    items = list(range(25))
    assert [x for c in _chunks(items, 10) for x in c] == items
    assert [len(c) for c in _chunks(items, 10)] == [10, 10, 5]


def test_an_empty_set_produces_no_queries():
    """A kind nobody has run must not send a statement with an empty IN clause."""
    assert _chunks([], 10) == []


def test_a_single_full_batch_is_one_chunk():
    assert len(_chunks(list(range(8_000)), 8_000)) == 1


async def _seed(db, kind: str, n_objects: int, *, endorse: int) -> None:
    """One committed run touching `n_objects`, with the first `endorse` of them confirmed and the rest fixed."""
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-OPP", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/t.jpg", width=1920, height=1080)
    db.add(frame)
    await db.flush()

    ids = []
    for _ in range(n_objects):
        o = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=1, bbox=[1, 1, 20, 20],
                   conf=0.5, source="relabel", state="review")
        db.add(o)
        ids.append(o.object_id)
    await db.flush()

    run = AgentRun(run_id=uuid.uuid4(), kind=kind, status="committed",
                   scope={}, changes={str(i): {"to": 1} for i in ids}, created_by="t")
    db.add(run)
    await db.flush()

    base = now_ns() + 10_000_000_000
    for i, oid in enumerate(ids):
        db.add(Review(review_id=uuid.uuid4(), object_id=oid, reviewer="t", ts_ns=base + i,
                      action="confirm" if i < endorse else "reclassify"))
    await db.commit()


async def test_a_set_larger_than_one_chunk_is_measured_rather_than_failing():
    """The bug, at a chunk size small enough to reproduce it without seeding 33,000 rows."""
    kind = "relabel"
    async with get_sessionmaker()() as db:
        await _seed(db, kind, 40, endorse=30)
        out = await measure_operation(db, kind, id_chunk=5)
    assert out["measured"] is True
    assert out["n"] >= 40


async def test_the_answer_does_not_depend_on_the_chunk_size():
    """Batching is a driver limit, not a measurement decision. If it changes the number, it is a bug that
    would only ever show up as two environments disagreeing about the same data."""
    kind = "relabel"
    async with get_sessionmaker()() as db:
        await _seed(db, kind, 40, endorse=30)
        whole = await measure_operation(db, kind, id_chunk=10_000)
        split = await measure_operation(db, kind, id_chunk=3)
    assert whole["n"] == split["n"]
    assert whole["precision"] == split["precision"]
