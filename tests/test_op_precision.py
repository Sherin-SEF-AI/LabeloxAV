"""A batch operation reports how often it was right, or says it does not know.

Every agent operation reports volume and none reports correctness, so the fast path cannot show it is
right. These pin the rules that make the resulting number trustworthy, because a precision figure that
flatters the operation is worse than no figure: the unmeasured state at least forces a dry run.

Driven with asyncio.run rather than a session-scoped event loop, matching tests/test_error_daemon.py.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete

from db.models import AgentRun, Frame, Object, Review
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.training.op_precision import MIN_SAMPLES, measure_operation

KIND = "test_op_precision_kind"
STARTED = datetime(2026, 7, 1, tzinfo=UTC)


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


async def _seed(db, *, n_confirm: int, n_reject: int, n_untouched: int = 0,
                action_reject: str = "reject") -> list[uuid.UUID]:
    """One committed run touching objects, plus human verdicts on some of them."""
    # Review.object_id is a real foreign key, so the objects have to exist. The minimum chain is one
    # session, one frame, and the objects the run touched.
    sid, fid = uuid.uuid4(), uuid.uuid4()
    t0 = _ns(STARTED)
    db.add(DbSession(session_id=sid, vehicle_id="OPPREC-01", start_ts_ns=t0,
                     end_ts_ns=t0 + 1_000_000_000, ontology_version="labelox-in-0.1.0"))
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t0, cam_id="cam_front",
                 img_uri="s3://x/opprec.jpg", width=64, height=64))
    oids = [uuid.uuid4() for _ in range(n_confirm + n_reject + n_untouched)]
    for o in oids:
        db.add(Object(object_id=o, frame_id=fid, class_id=1, bbox=[0.0, 0.0, 8.0, 8.0], conf=0.5))
    await db.flush()
    db.add(AgentRun(run_id=uuid.uuid4(), kind=KIND, scope="frame", status="committed",
                    changes={str(o): {"from_state": "review", "to_state": "auto_accept"} for o in oids},
                    created_at=STARTED))
    after = _ns(STARTED) + 1_000_000_000
    for o in oids[:n_confirm]:
        db.add(Review(review_id=uuid.uuid4(), object_id=o, reviewer="t", action="confirm", ts_ns=after))
    for o in oids[n_confirm:n_confirm + n_reject]:
        db.add(Review(review_id=uuid.uuid4(), object_id=o, reviewer="t", action=action_reject, ts_ns=after))
    await db.commit()
    return oids


async def _clean(db) -> None:
    await db.execute(delete(AgentRun).where(AgentRun.kind == KIND))
    # Frames and objects cascade from the session, and reviews cascade from the object.
    await db.execute(delete(DbSession).where(DbSession.vehicle_id == "OPPREC-01"))
    await db.commit()


async def _measure(**seed) -> dict:
    """Seed a fresh run, measure it, then remove the run so tests cannot see each other's data."""
    async with get_sessionmaker()() as db:
        await _clean(db)
        try:
            await _seed(db, **seed)
            return await measure_operation(db, KIND)
        finally:
            await _clean(db)


def test_unmeasured_below_the_sample_floor():
    """Two confirmations is an anecdote. It must not render as 100% precision."""
    r = asyncio.run(_measure(n_confirm=2, n_reject=0))
    assert r["measured"] is False
    assert r["precision"] is None
    assert str(MIN_SAMPLES) in r["reason"]


def test_silence_is_not_success():
    """Objects nobody reviewed count toward neither side.

    Counting untouched objects as correct is how an automation measures itself at 100% by being ignored.
    Here 30 objects are reviewed and 500 are not, so the denominator has to be 30 and the operation's reach
    is reported separately.
    """
    r = asyncio.run(_measure(n_confirm=20, n_reject=10, n_untouched=500))
    assert r["measured"] is True
    assert r["n"] == 30, "untouched objects leaked into the denominator"
    assert r["precision"] == pytest.approx(20 / 30, abs=1e-4)
    assert r["objects_touched"] == 530
    assert r["reviewed_fraction"] == pytest.approx(30 / 530, abs=1e-4)


def test_a_correction_counts_against_the_operation():
    """A human who had to reclassify what the operation did is evidence it was wrong."""
    r = asyncio.run(_measure(n_confirm=10, n_reject=20, action_reject="reclassify"))
    assert r["misses"] == 20
    assert r["precision"] == pytest.approx(10 / 30, abs=1e-4)


def test_recall_is_null_not_invented():
    """Recall needs the objects the operation should have touched and did not. Nothing records that."""
    r = asyncio.run(_measure(n_confirm=20, n_reject=10))
    assert r["recall"] is None


def test_the_sampling_bias_is_stated():
    """Review is not a random sample, and a number that hides that invites the wrong conclusion."""
    r = asyncio.run(_measure(n_confirm=20, n_reject=10))
    assert "not a random sample" in r["caveat"]


def test_reviews_before_the_run_are_not_verdicts_on_it():
    """A ruling that predates the operation cannot be a judgement of it."""

    async def body() -> dict:
        async with get_sessionmaker()() as db:
            await _clean(db)
            try:
                oids = await _seed(db, n_confirm=26, n_reject=0)
                before = _ns(STARTED) - 60_000_000_000
                for o in oids:
                    db.add(Review(review_id=uuid.uuid4(), object_id=o, reviewer="t",
                                  action="reject", ts_ns=before))
                await db.commit()
                return await measure_operation(db, KIND)
            finally:
                await _clean(db)

    r = asyncio.run(body())
    assert r["misses"] == 0, "a review from before the run was counted against it"
    assert r["hits"] == 26


def test_an_operation_nobody_ran_is_unmeasured():
    async def body() -> dict:
        async with get_sessionmaker()() as db:
            await _clean(db)
            return await measure_operation(db, KIND)

    r = asyncio.run(body())
    assert r["measured"] is False
    assert r["n"] == 0
