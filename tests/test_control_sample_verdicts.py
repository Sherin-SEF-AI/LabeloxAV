"""601 control samples were seeded and not one was ever judged.

The control sample is the gate auditing itself: a small random share of its own auto-accepts is mirrored
into a stream that is always sent to a human, and the fraction judged incorrect is the MEASURED precision
the drift detector watches. Its whole purpose is to be a number a buyer can trust over a self-reported one.

Seeding worked. `services/autolabel/persist.py` has called `maybe_sample` on every auto-accept all along,
and this corpus holds 601 rows. What was missing was any way to judge them: nothing listed what needed a
verdict, and the verdict route had no caller. So `measured_precision` returned `{"reviewed": 0,
"precision": null}` from the day the feature shipped, and the drift axis that reads it never fired.

A worklist nobody can see is the same as no worklist.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import ControlSample, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.govern.control_sample import measured_precision, pending_samples, record_verdict

pytestmark = pytest.mark.db


async def _sample(db, *, auto: bool = True, class_id: int = 1) -> str:
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CTRL-01", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()
    oid = uuid.uuid4()
    db.add(Object(object_id=oid, frame_id=fid, class_id=class_id, bbox=[1, 1, 9, 9],
                  conf=0.91, source="auto_accept", state="auto_accept"))
    await db.flush()
    sid = uuid.uuid4()
    db.add(ControlSample(sample_id=sid, object_id=oid, was_auto_accepted=auto))
    await db.commit()
    return str(sid)


class TestTheWorklistExists:
    async def test_pending_samples_can_be_listed_at_all(self):
        """The missing half. Seeding always worked; nothing could see what needed judging."""
        async with get_sessionmaker()() as db:
            sid = await _sample(db)
            out = await pending_samples(db, limit=500)
        assert sid in {s["sample_id"] for s in out["samples"]}

    async def test_each_one_carries_enough_to_rule_on(self):
        """A sample id alone is not reviewable. A crop, a class and a confidence are the judgement."""
        async with get_sessionmaker()() as db:
            sid = await _sample(db)
            out = await pending_samples(db, limit=500)
        s = next(x for x in out["samples"] if x["sample_id"] == sid)
        assert s["crop_url"].endswith("/crop") and s["class_name"] and s["conf"] == pytest.approx(0.91)
        assert s["frame_id"] and s["object_id"]

    async def test_a_judged_sample_leaves_the_list(self):
        async with get_sessionmaker()() as db:
            sid = await _sample(db)
            await record_verdict(db, sid, "correct")
            out = await pending_samples(db, limit=500)
        assert sid not in {s["sample_id"] for s in out["samples"]}


class TestTheVerdictIsTheMeasurement:
    async def test_a_correct_verdict_counts_as_reviewed(self):
        async with get_sessionmaker()() as db:
            before = await measured_precision(db)
            sid = await _sample(db)
            await record_verdict(db, sid, "correct")
            after = await measured_precision(db)
        assert after["reviewed"] == before["reviewed"] + 1
        assert after["incorrect"] == before["incorrect"]

    async def test_an_incorrect_verdict_moves_the_number(self):
        async with get_sessionmaker()() as db:
            before = await measured_precision(db)
            sid = await _sample(db)
            await record_verdict(db, sid, "incorrect")
            after = await measured_precision(db)
        assert after["incorrect"] == before["incorrect"] + 1

    async def test_precision_stops_being_null_once_anything_is_judged(self):
        """It was null for the life of the feature, and the drift detector was watching it."""
        async with get_sessionmaker()() as db:
            sid = await _sample(db)
            await record_verdict(db, sid, "correct")
            out = await measured_precision(db)
        assert out["precision"] is not None

    async def test_a_verdict_that_is_not_a_verdict_is_refused(self):
        """`measured_precision` treats anything that is not the literal "incorrect" as correct, so a typo
        used to report the gate as more accurate than it is. This is the number a buyer is meant to trust
        over a self-reported one, which is exactly the number that must not be quietly wrong."""
        async with get_sessionmaker()() as db:
            sid = await _sample(db)
            out = await record_verdict(db, sid, "looks fine")
            assert "error" in out
            cs = await db.get(ControlSample, uuid.UUID(sid))
            await db.refresh(cs)
        assert cs.human_verdict is None, "an unrecognised verdict was stored and counted as correct"

    async def test_a_missing_sample_is_reported_not_crashed(self):
        async with get_sessionmaker()() as db:
            assert "error" in await record_verdict(db, str(uuid.uuid4()), "correct")

    async def test_only_auto_accepted_controls_count_toward_precision(self):
        """The number is the precision OF THE GATE. A sample that was not auto-accepted was never the
        gate's claim to make."""
        async with get_sessionmaker()() as db:
            before = await measured_precision(db)
            sid = await _sample(db, auto=False)
            await record_verdict(db, sid, "incorrect")
            after = await measured_precision(db)
        assert after["reviewed"] == before["reviewed"]


class TestAVerdictAnywhereCountsHere:
    """Phase 1 of the autonomy work: reviewing a control-sampled object IS the control verdict.

    The 601/0 state above had a second cause beyond the missing worklist: verdicts only counted when
    they arrived through the dedicated control endpoint, so a reviewer who judged the same object in
    the grid did the work and the measurement never received it. The absorption lives in
    `apply_review_batch`, the one path every review surface flows through.
    """

    async def _seeded(self, db):
        sid = await _sample(db)
        cs = await db.get(ControlSample, uuid.UUID(sid))
        return cs, await db.get(Object, cs.object_id)

    async def test_an_accept_fills_the_pending_verdict_as_correct(self):
        from services.autolabel.ontology import get_ontology
        from services.review_apply import apply_review_batch

        async with get_sessionmaker()() as db:
            cs, obj = await self._seeded(db)
            res = await apply_review_batch(db, [obj], action="accept", onto=get_ontology(),
                                           role="reviewer", reviewer="t")
            await db.commit()
            assert res.control_judged == 1
            await db.refresh(cs)
            assert cs.human_verdict == "correct"
            assert cs.verdict_at is not None, "a verdict without a time cannot be aged, and phase 1 " \
                                              "makes staleness a first-class refusal"

    async def test_a_reject_fills_it_as_incorrect(self):
        from services.autolabel.ontology import get_ontology
        from services.review_apply import apply_review_batch

        async with get_sessionmaker()() as db:
            cs, obj = await self._seeded(db)
            await apply_review_batch(db, [obj], action="reject", onto=get_ontology(),
                                     role="reviewer", reviewer="t")
            await db.commit()
            await db.refresh(cs)
            assert cs.human_verdict == "incorrect", \
                "the gate auto-accepted something a person threw out; that is exactly a gate error"

    async def test_a_class_correction_fills_it_as_incorrect(self):
        from services.autolabel.ontology import get_ontology
        from services.review_apply import apply_review_batch

        async with get_sessionmaker()() as db:
            cs, obj = await self._seeded(db)
            other = next(c.id for c in get_ontology().classes if c.id != obj.class_id)
            await apply_review_batch(db, [obj], action="reclassify", onto=get_ontology(),
                                     class_id=other, role="reviewer", reviewer="t")
            await db.commit()
            await db.refresh(cs)
            assert cs.human_verdict == "incorrect", \
                "the gate accepted the wrong class; a relabel is a defect verdict, not an accept"

    async def test_an_already_judged_sample_is_not_overwritten(self):
        from services.autolabel.ontology import get_ontology
        from services.review_apply import apply_review_batch

        async with get_sessionmaker()() as db:
            cs, obj = await self._seeded(db)
            await record_verdict(db, str(cs.sample_id), "incorrect")
            await apply_review_batch(db, [obj], action="accept", onto=get_ontology(),
                                     role="reviewer", reviewer="t")
            await db.commit()
            await db.refresh(cs)
            assert cs.human_verdict == "incorrect", \
                "the first ruling stands; a later review must not quietly flip the measurement"

    async def test_precision_carries_its_interval_and_its_age(self):
        async with get_sessionmaker()() as db:
            sid = await _sample(db)
            await record_verdict(db, sid, "correct")
            prec = await measured_precision(db)
            assert prec["interval"]["p"] is not None
            assert 0.0 <= prec["interval"]["lo"] <= prec["interval"]["hi"] <= 1.0
            assert prec["measured_at"] is not None, \
                "the newest verdict just landed with a timestamp, so the age is known"
