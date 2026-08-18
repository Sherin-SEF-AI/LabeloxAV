"""A fixed operation could never earn its score back.

`relabel` moved 1,047 buses into a bus shelter in August. The ontology guard that makes that category change
impossible landed afterwards, and the 63 corrections a human made while cleaning up sat in the denominator
with nothing to outvote them: the chip on the button read 0% over 63 reviewed outcomes and would have read
0% for as long as the corpus existed, however well the repaired operation behaved.

The window has to bound the runs rather than the reviews. A review window does not help, because the
corrections to a bad run arrive whenever somebody gets round to cleaning it up, which is exactly when a
recent-reviews window would count them. The run is the only record of which behaviour produced an object.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete

from core.timebase import now_ns
from db.models import AgentRun, Frame, Object, OntologyClass, OntologyVersion, Review
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.training import op_precision as op

pytestmark = pytest.mark.db

KIND = "test_window_op"
DAY_NS = 86_400 * 1_000_000_000


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


async def _run(db, *, age_days: int, objects: list[uuid.UUID]) -> AgentRun:
    when = datetime.now(UTC) - timedelta(days=age_days)
    row = AgentRun(run_id=uuid.uuid4(), kind=KIND, scope={}, status="committed", policy={},
                   counts={}, changes={str(o): {"from_class": 1} for o in objects}, critic={},
                   created_at=when)
    db.add(row)
    await db.flush()
    return row


async def _objects(db, n: int) -> list[uuid.UUID]:
    """n real objects on one frame. Review.object_id is a foreign key, so these cannot be bare uuids."""
    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts = now_ns()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="WINDOW-1", start_ts_ns=ts, end_ts_ns=ts + 1,
                     city="BLR", sensors={}, ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                 img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    cid = next(c.id for c in onto.classes if c.name == "rider")
    ids = []
    for i in range(n):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[10.0 + i, 10.0, 60.0 + i, 110.0],
                      conf=0.5, source="fused", state="review", attrs={}, provenance={}, version=1))
        ids.append(oid)
    await db.flush()
    return ids


async def _review(db, object_id: uuid.UUID, action: str, *, age_days: int) -> None:
    """A human ruling, timestamped."""
    db.add(Review(object_id=object_id, reviewer="probe", action=action, before=None, after=None,
                  time_spent_ms=0, ts_ns=_ns(datetime.now(UTC) - timedelta(days=age_days))))
    await db.flush()


@pytest.fixture(autouse=True)
async def _clean():
    yield
    async with get_sessionmaker()() as db:
        await db.execute(delete(AgentRun).where(AgentRun.kind == KIND))
        await db.commit()


async def _seed(db, *, age_days: int, hits: int, misses: int, review_age_days: int = 0) -> None:
    ids = await _objects(db, hits + misses)
    await _run(db, age_days=age_days, objects=ids)
    for i, oid in enumerate(ids):
        await _review(db, oid, "confirm" if i < hits else "reclassify", age_days=review_age_days)
    await db.commit()


class TestTheWindow:
    async def test_an_old_run_no_longer_holds_a_repaired_operation_at_zero(self):
        """The whole point. Thirty misses in August must not decide what the button says in October."""
        async with get_sessionmaker()() as db:
            await _seed(db, age_days=90, hits=0, misses=30)
            await _seed(db, age_days=1, hits=30, misses=0)

            windowed = await op.measure_operation(db, KIND, runs_since_ns=_ns(datetime.now(UTC)) - 30 * DAY_NS)
            all_time = await op.measure_operation(db, KIND)

        assert windowed["precision"] == 1.0, "the recent runs are the only ones that describe current behaviour"
        assert windowed["n"] == 30
        assert all_time["precision"] == 0.5, "the all-time view still shows everything that ever happened"

    async def test_the_default_view_is_windowed(self, monkeypatch):
        """A caller that asks for nothing in particular gets the answer about now, since that is the one the
        chip on a button is asking."""
        # Only this kind, so the assertion is about the window and not about whatever else the shared test
        # database happens to hold.
        monkeypatch.setattr(op, "OPERATION_KINDS", (KIND,))
        async with get_sessionmaker()() as db:
            await _seed(db, age_days=90, hits=0, misses=30)
            out = (await op.measure_all(db))[KIND]
        assert not out["measured"]
        assert out["excluded_runs"] == 1

    def test_the_window_is_a_month(self):
        # Long enough to clear MIN_SAMPLES on an operation anybody uses, short enough that a fix proves
        # itself within a working month.
        assert op.WINDOW_DAYS == 30

    async def test_all_time_stays_available_for_an_audit(self, monkeypatch):
        """"What has this operation ever done" is a real question and must not become unanswerable."""
        monkeypatch.setattr(op, "OPERATION_KINDS", (KIND,))
        async with get_sessionmaker()() as db:
            await _seed(db, age_days=200, hits=10, misses=20)
            out = (await op.measure_all(db, window_days=None))[KIND]
        assert out["measured"] and out["n"] == 30


class TestSayingWhatWasExcluded:
    async def test_no_recent_runs_is_reported_differently_from_no_runs(self):
        """An operation nobody has run this month and an operation that has never run look identical under
        a bare "unmeasured", and they call for opposite decisions."""
        async with get_sessionmaker()() as db:
            never = await op.measure_operation(db, KIND, runs_since_ns=_ns(datetime.now(UTC)) - 30 * DAY_NS)
            assert "never" not in never["reason"]
            assert never["reason"] == "no committed runs of this operation"

            await _seed(db, age_days=90, hits=0, misses=30)
            stale = await op.measure_operation(db, KIND, runs_since_ns=_ns(datetime.now(UTC)) - 30 * DAY_NS)

        assert "older runs excluded" in stale["reason"]
        assert stale["excluded_runs"] == 1

    async def test_a_measured_score_says_how_much_it_left_out(self):
        """A number over part of the evidence is a different claim from a number over all of it, and the
        reader cannot tell without being told."""
        async with get_sessionmaker()() as db:
            await _seed(db, age_days=90, hits=0, misses=30)
            await _seed(db, age_days=1, hits=30, misses=0)
            out = await op.measure_operation(db, KIND, runs_since_ns=_ns(datetime.now(UTC)) - 30 * DAY_NS)

        assert out["excluded_runs"] == 1
        assert out["runs_scored"] == 1
        assert "1 older runs excluded" in out["dataset_slice"]


class TestWhatTheWindowMustNotChange:
    async def test_a_review_before_its_run_is_still_ignored(self):
        """Only a ruling made after the operation ran can be a verdict on it."""
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 30)
            await _run(db, age_days=2, objects=ids)
            for oid in ids:
                await _review(db, oid, "reclassify", age_days=5)   # before the run
            await db.commit()
            out = await op.measure_operation(db, KIND, runs_since_ns=_ns(datetime.now(UTC)) - 30 * DAY_NS)
        assert not out["measured"] and out["n"] == 0

    async def test_the_object_of_two_runs_is_scored_against_the_one_in_the_window(self):
        """The newer run is what the object carries, so the human is ruling on the newer run."""
        async with get_sessionmaker()() as db:
            ids = await _objects(db, 30)
            await _run(db, age_days=90, objects=ids)     # excluded by the window
            await _run(db, age_days=2, objects=ids)      # the state the reviewer saw
            for oid in ids:
                await _review(db, oid, "confirm", age_days=1)
            await db.commit()
            out = await op.measure_operation(db, KIND, runs_since_ns=_ns(datetime.now(UTC)) - 30 * DAY_NS)

        assert out["measured"] and out["precision"] == 1.0, (
            "attributing to the excluded older run would drop this evidence entirely")
