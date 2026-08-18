"""Giving the error detectors a verdict path, and the calibration that unlocks.

298,529 candidates carry one verdict between them. Confirm and dismiss both took a single candidate id, so
the queue was not reviewable in principle, and no detector has a measurable precision because precision
needs verdicts in volume.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings

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


async def _seed_candidates(kind: str, n: int, *, score: float = 0.5) -> list[str]:
    """n pending candidates of one detector kind, on real objects so confirm can mutate them."""
    from core.timebase import now_ns, seconds_to_ns
    from db.models import ErrorCandidate, Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cls = next(c.id for c in onto.classes if c.name == "pedestrian")
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
    ids = []
    async with get_sessionmaker()() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/f.jpg",
                     width=320, height=240, quality=0.9, scene={}))
        await db.flush()
        for _ in range(n):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=cls, bbox=[1.0, 1.0, 30.0, 30.0],
                          conf=0.6, source="fused", state="auto_accept", attrs={}, provenance={}, version=1))
            await db.flush()
            c = ErrorCandidate(object_id=oid, kind=kind, score=score, proposed_label=None,
                               detail={}, status="pending")
            db.add(c)
            await db.flush()
            ids.append(str(c.candidate_id))
        await db.commit()
    return ids


@requires_infra
def test_a_bulk_dismissal_is_one_action_with_one_recorded_reason():
    """Dismissing 400 near-duplicate candidates is a judgement about a detector, not about 400 objects."""
    from db.models import ErrorCandidate
    from db.session import get_sessionmaker
    from services.errordetect.queue import bulk_verdict

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"

    async def _flow():
        ids = await _seed_candidates(kind, 25)
        async with get_sessionmaker()() as db:
            r = await bulk_verdict(db, ids, "dismissed", note="scores the frame, not the object")
        assert r["applied"] == 25 and r["already_decided"] == 0 and r["missing"] == 0

        async with get_sessionmaker()() as db:
            c = await db.get(ErrorCandidate, uuid.UUID(ids[0]))
            assert c.status == "dismissed"
            assert c.decided_at is not None, "a dismissal used to leave no trace at all"
            assert c.decision_note == "scores the frame, not the object"

    run_async(_flow())


@requires_infra
def test_a_candidate_already_ruled_on_is_skipped_rather_than_overwritten():
    """Two reviewers on the same queue is normal. Replacing the first verdict would corrupt the calibration
    with no trace of the disagreement."""
    from db.models import ErrorCandidate
    from db.session import get_sessionmaker
    from services.errordetect.queue import bulk_verdict

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"

    async def _flow():
        ids = await _seed_candidates(kind, 5)
        async with get_sessionmaker()() as db:
            await bulk_verdict(db, ids[:2], "dismissed", note="first reviewer")
        async with get_sessionmaker()() as db:
            second = await bulk_verdict(db, ids, "dismissed", note="second reviewer")
        assert second["applied"] == 3 and second["already_decided"] == 2

        async with get_sessionmaker()() as db:
            c = await db.get(ErrorCandidate, uuid.UUID(ids[0]))
            assert c.decision_note == "first reviewer", "the earlier verdict must survive"

    run_async(_flow())


@requires_infra
def test_a_missing_candidate_is_counted_not_silently_ignored():
    from db.session import get_sessionmaker
    from services.errordetect.queue import bulk_verdict

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"

    async def _flow():
        ids = await _seed_candidates(kind, 3)
        async with get_sessionmaker()() as db:
            r = await bulk_verdict(db, [*ids, str(uuid.uuid4())], "dismissed")
        assert r["applied"] == 3 and r["missing"] == 1

    run_async(_flow())


@requires_infra
def test_bulk_confirmation_still_writes_the_review_trail_per_object():
    """Confirming mutates the object and writes a Review row. A bulk UPDATE would skip that, and the audit
    trail is the reason confirmations are worth anything to the retrain loop."""
    from sqlalchemy import func, select

    from db.models import ErrorCandidate, Object, Review
    from db.session import get_sessionmaker
    from services.errordetect.queue import bulk_verdict

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"

    async def _flow():
        ids = await _seed_candidates(kind, 4)
        async with get_sessionmaker()() as db:
            before = (await db.execute(select(func.count(Review.review_id)))).scalar()
        async with get_sessionmaker()() as db:
            r = await bulk_verdict(db, ids, "confirmed_error", note="real misses")
        assert r["applied"] == 4

        async with get_sessionmaker()() as db:
            after = (await db.execute(select(func.count(Review.review_id)))).scalar()
            c = await db.get(ErrorCandidate, uuid.UUID(ids[0]))
            obj = await db.get(Object, c.object_id)
        assert after - before == 4, "each confirmation must leave its own review row"
        assert c.status == "confirmed_error"
        assert obj.state == "review", "no proposed class, so the object goes back to a human"

    run_async(_flow())


@requires_infra
def test_detector_precision_says_unmeasured_rather_than_low_when_verdicts_are_scarce():
    """A page full of intervals from zero to one invites the reading that the detectors are bad, when the
    truth is that nobody has looked at them."""
    from db.session import get_sessionmaker
    from services.errordetect.queue import (
        MIN_VERDICTS_FOR_PRECISION,
        bulk_verdict,
        detector_precision,
    )

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"

    async def _flow():
        ids = await _seed_candidates(kind, MIN_VERDICTS_FOR_PRECISION - 1)
        async with get_sessionmaker()() as db:
            await bulk_verdict(db, ids, "dismissed")
        async with get_sessionmaker()() as db:
            p = await detector_precision(db)
        d = p["per_kind"][kind]
        assert d["usable"] is False
        assert "unmeasured, not low" in d["note"]

    run_async(_flow())


@requires_infra
def test_detector_precision_is_confirmed_over_decided_once_there_are_enough_verdicts():
    """The calibration the queue has never had: a score that ranks is not a score that predicts."""
    from db.session import get_sessionmaker
    from services.errordetect.queue import bulk_verdict, detector_precision

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"

    async def _flow():
        ids = await _seed_candidates(kind, 30)
        async with get_sessionmaker()() as db:
            await bulk_verdict(db, ids[:9], "confirmed_error")
        async with get_sessionmaker()() as db:
            await bulk_verdict(db, ids[9:], "dismissed")
        async with get_sessionmaker()() as db:
            p = await detector_precision(db)

        d = p["per_kind"][kind]
        assert d["confirmed_error"] == 9 and d["dismissed"] == 21 and d["decided"] == 30
        assert d["usable"] is True
        assert d["precision"]["p"] == 0.3
        # and the interval is reported, so 0.3 from 30 verdicts is distinguishable from 0.3 from 3
        assert d["precision"]["n"] == 30 and d["precision"]["hi"] > 0.3

    run_async(_flow())


@requires_infra
def test_the_queue_can_be_read_one_detector_at_a_time():
    """Ranking is across detectors whose scores are not commensurable, so a mixed page is mostly whichever
    detector emits the biggest numbers. Judging one detector requires seeing only that detector."""
    from db.session import get_sessionmaker
    from services.errordetect.queue import list_candidates

    loud = f"test_loud_{uuid.uuid4().hex[:6]}"
    quiet = f"test_quiet_{uuid.uuid4().hex[:6]}"

    async def _flow():
        await _seed_candidates(loud, 5, score=0.99)
        await _seed_candidates(quiet, 5, score=0.10)
        async with get_sessionmaker()() as db:
            only_quiet = await list_candidates(db, "pending", 100, kind=quiet)
        assert len(only_quiet) == 5
        assert {c["kind"] for c in only_quiet} == {quiet}

    run_async(_flow())


def test_an_unknown_verdict_is_refused():
    from db.session import get_sessionmaker
    from services.errordetect.queue import bulk_verdict

    async def _flow():
        async with get_sessionmaker()() as db:
            with pytest.raises(ValueError, match="unknown verdict"):
                await bulk_verdict(db, ["x"], "maybe")

    run_async(_flow())
