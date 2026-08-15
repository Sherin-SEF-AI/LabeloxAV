"""Recall mining wrote rows nobody could judge, so the loop it feeds could never close.

Each mined candidate lands at `status="pending"`. There was no route and no service function that wrote
`confirmed` or `rejected`, so nothing ever left that state.

The cost is not the unread queue. `fit_channel_reliability` exists to replace the hand-guessed per-channel
priors with measured ones and selects on exactly those two statuses, so its query was guaranteed empty. The
priors stayed guesses, and a feature whose entire purpose is to learn which channel is worth trusting had no
way of being told.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import Frame, Object, RecallCandidate
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.recall.adjudicate import (
    AdjudicationError,
    adjudicate,
    adjudication_progress,
)

pytestmark = pytest.mark.db


async def _candidate(db, *, channels=("trackgap",), source="auto_accept", state="review"):
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="RECALL-01", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()
    oid = uuid.uuid4()
    db.add(Object(object_id=oid, frame_id=fid, class_id=1, bbox=[1, 1, 9, 9],
                  conf=0.4, source=source, state=state))
    await db.flush()
    cid = uuid.uuid4()
    db.add(RecallCandidate(candidate_id=cid, object_id=oid, frame_id=fid,
                           channels=list(channels), fn_value=0.8, class_id=1, status="pending"))
    await db.commit()
    return str(cid), oid


class TestAVerdictCanBeRecorded:
    async def test_confirming_moves_it_off_pending(self):
        """The whole bug in one assertion: nothing could write this status."""
        async with get_sessionmaker()() as db:
            cid, _ = await _candidate(db)
            out = await adjudicate(db, cid, "confirmed")
            row = await db.get(RecallCandidate, uuid.UUID(cid))
            await db.refresh(row)
        assert out["verdict"] == "confirmed" and row.status == "confirmed"

    async def test_rejecting_records_the_other_verdict(self):
        async with get_sessionmaker()() as db:
            cid, _ = await _candidate(db)
            await adjudicate(db, cid, "rejected")
            row = await db.get(RecallCandidate, uuid.UUID(cid))
            await db.refresh(row)
        assert row.status == "rejected"

    async def test_a_verdict_that_is_not_a_verdict_is_refused(self):
        async with get_sessionmaker()() as db:
            cid, _ = await _candidate(db)
            with pytest.raises(AdjudicationError):
                await adjudicate(db, cid, "maybe")

    async def test_a_missing_candidate_is_refused_rather_than_crashing(self):
        async with get_sessionmaker()() as db:
            with pytest.raises(AdjudicationError, match="not found"):
                await adjudicate(db, str(uuid.uuid4()), "confirmed")

    async def test_changing_your_mind_is_allowed(self):
        """A reviewer who confirms and then reconsiders should not need an administrator, and the fit reads
        the current status, so the last word is the one that counts."""
        async with get_sessionmaker()() as db:
            cid, _ = await _candidate(db)
            await adjudicate(db, cid, "confirmed")
            out = await adjudicate(db, cid, "rejected")
        assert out["was"] == "confirmed" and out["verdict"] == "rejected"


class TestWhatItDoesToTheObject:
    async def test_a_confirmed_candidate_routes_its_object_to_review(self):
        """Confirming says the miner found something real, so it belongs in the queue rather than sitting as
        a machine guess nobody will look at."""
        async with get_sessionmaker()() as db:
            cid, oid = await _candidate(db, state="auto_accept")
            out = await adjudicate(db, cid, "confirmed")
            obj = await db.get(Object, oid)
            await db.refresh(obj)
        assert out["object_routed_to_review"] is True and obj.state == "review"

    async def test_it_records_which_channel_found_it(self):
        async with get_sessionmaker()() as db:
            cid, oid = await _candidate(db, channels=("openvocab",))
            await adjudicate(db, cid, "confirmed", reviewer="ada")
            obj = await db.get(Object, oid)
            await db.refresh(obj)
        assert (obj.provenance or {})["recall_confirmed"]["channels"] == ["openvocab"]

    async def test_a_human_labelled_object_is_left_alone(self):
        """A recall confirmation says the miner was right, not that it may overwrite somebody's decision."""
        async with get_sessionmaker()() as db:
            cid, oid = await _candidate(db, source="human", state="accepted")
            out = await adjudicate(db, cid, "confirmed")
            obj = await db.get(Object, oid)
            await db.refresh(obj)
        assert out["object_routed_to_review"] is False and obj.state == "accepted"

    async def test_rejecting_does_not_touch_the_object(self):
        """It is a claim about the suggestion, not about the label. Deleting an object on that basis would
        be the miner marking its own homework in the other direction."""
        async with get_sessionmaker()() as db:
            cid, oid = await _candidate(db, state="review")
            await adjudicate(db, cid, "rejected")
            obj = await db.get(Object, oid)
            await db.refresh(obj)
        assert obj is not None and obj.state == "review"


class TestTheLoopCanNowClose:
    async def test_the_reliability_fit_has_something_to_read(self):
        """`fit_channel_reliability` selects on confirmed and rejected. That query was guaranteed empty, so
        the per-channel priors could never stop being guesses."""
        from sqlalchemy import select

        async with get_sessionmaker()() as db:
            cid_a, _ = await _candidate(db, channels=("trackgap",))
            cid_b, _ = await _candidate(db, channels=("trackgap",))
            await adjudicate(db, cid_a, "confirmed")
            await adjudicate(db, cid_b, "rejected")
            judged = (await db.execute(
                select(RecallCandidate).where(
                    RecallCandidate.status.in_(("confirmed", "rejected"))))).scalars().all()
        assert len(judged) >= 2

    async def test_progress_reports_per_channel_not_only_a_total(self):
        """The fit needs a minimum per channel before it applies a measurement, so a healthy total can hide
        a channel nobody has ruled on."""
        async with get_sessionmaker()() as db:
            cid, _ = await _candidate(db, channels=("region",))
            await adjudicate(db, cid, "confirmed")
            prog = await adjudication_progress(db)
        assert prog["judged"] >= 1
        assert prog["per_channel"]["region"]["confirmed"] >= 1

    async def test_progress_counts_pending_separately(self):
        async with get_sessionmaker()() as db:
            await _candidate(db)
            prog = await adjudication_progress(db)
        assert prog["pending"] >= 1
        assert prog["total"] >= prog["judged"] + prog["pending"] - prog["pending"]
