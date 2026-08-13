"""Fifty thousand individual rewrites are not reviewable. The decisions behind them are.

A relabel run rewrote 50,812 objects across 611 class pairs. Most were refinements. Some were category
errors, and an operator who spots one bus labelled as a bus shelter has no way to learn that 1,046 more share
it, or that 708 traffic signs and 522 hoardings reached the same class by different routes.

Every rewrite stamped `provenance.agent_relabel` with the move it made, so the lineages can be counted from
the corpus rather than guessed at, and `services/agent/class_move.py` says which of them the system would
refuse today. Measured on the live corpus: 107 lineages over 48,664 objects, of which 17 lineages and 3,057
objects are now refused.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from db.models import Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.agent.contamination import agent_relabel_lineages, revert_lineage, summarize
from services.agent.runs import revert_run
from services.autolabel.ontology import get_ontology as _onto

pytestmark = pytest.mark.db


async def _stamped(db, moves: list[tuple[str, str, int]]):
    """Objects carrying an `agent_relabel` stamp, `n` of each move, and the class that move produced.

    The class matters: an object is only outstanding while it still carries the class the move gave it, so a
    fixture that stamps the provenance and leaves the class alone describes an already-corrected object and
    nothing would be reported as work to do.
    """
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CONTAM-01", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()
    onto = _onto()
    for src, dst, n in moves:
        try:
            cid = onto.by_name(dst).id
        except KeyError:
            cid = 1          # a destination the ontology no longer names; the move still happened
        for _ in range(n):
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid, bbox=[1, 1, 9, 9],
                          conf=0.5, source="auto_accept", state="review",
                          provenance={"agent_relabel": [f"{src} -> {dst} (0.989)"]}))
    await db.commit()


def _find(rows, src, dst):
    return next((r for r in rows if r["from_name"] == src and r["to_name"] == dst), None)


class TestLineagesAreCounted:
    async def test_a_systematic_move_is_reported_with_its_size(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 30)])
            rows = await agent_relabel_lineages(db, min_count=25)
        r = _find(rows, "bus", "bmtc_bus_shelter")
        assert r and r["count"] >= 30

    async def test_a_one_off_is_not_a_lineage(self):
        """A move that happened twice is a coincidence. One that happened a thousand times is a policy
        somebody should get to review."""
        async with get_sessionmaker()() as db:
            await _stamped(db, [("cycle", "moped", 2)])
            rows = await agent_relabel_lineages(db, min_count=25)
        assert _find(rows, "cycle", "moped") is None

    async def test_examples_come_with_it_so_the_lineage_can_be_opened(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 30)])
            rows = await agent_relabel_lineages(db, min_count=25)
        r = _find(rows, "bus", "bmtc_bus_shelter")
        assert r["examples"], "a lineage nobody can open is a number, not a finding"
        assert len(r["examples"]) <= 8
        # The frame, not only the object: the correction flow lives in the frame editor, so an example
        # without one links nowhere.
        assert all(e["frame_id"] and e["object_id"] for e in r["examples"])

    async def test_the_biggest_lineage_comes_first(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("truck", "tempo", 26), ("truck", "petrol_tanker", 60)])
            rows = await agent_relabel_lineages(db, min_count=25)
        names = [(r["from_name"], r["to_name"]) for r in rows]
        assert names.index(("truck", "petrol_tanker")) < names.index(("truck", "tempo"))


class TestWhichWouldBeRefusedNow:
    async def test_a_category_error_is_marked_and_explained(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 30)])
            rows = await agent_relabel_lineages(db, min_count=25)
        r = _find(rows, "bus", "bmtc_bus_shelter")
        assert r["refused_now"] is True
        assert "stuff" in r["reason"]

    async def test_a_refinement_is_not_marked(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("sedan", "mpv", 30)])
            rows = await agent_relabel_lineages(db, min_count=25)
        r = _find(rows, "sedan", "mpv")
        assert r["refused_now"] is False and r["reason"] is None

    async def test_the_refused_only_view_is_the_cleanup_list(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 30), ("sedan", "mpv", 30)])
            rows = await agent_relabel_lineages(db, min_count=25, refused_only=True)
        assert all(r["refused_now"] for r in rows)
        assert _find(rows, "sedan", "mpv") is None

    async def test_a_class_the_ontology_no_longer_names_is_reported_not_dropped(self):
        """The move happened and the objects still carry it. It cannot be judged, which is different from
        not existing."""
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "a_class_that_was_retired", 30)])
            rows = await agent_relabel_lineages(db, min_count=25)
        r = _find(rows, "bus", "a_class_that_was_retired")
        assert r is not None and r["refused_now"] is False


class TestTheSummary:
    async def test_it_separates_what_moved_from_what_is_now_wrong(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 30), ("sedan", "mpv", 40)])
            rows = await agent_relabel_lineages(db, min_count=25)
        s = summarize(rows)
        assert s["objects"] >= 70
        assert s["refused_objects"] >= 30
        assert s["refused_objects"] < s["objects"]

    async def test_an_empty_corpus_summarizes_to_zeroes(self):
        assert summarize([]) == {"lineages": 0, "objects": 0,
                                 "refused_lineages": 0, "refused_objects": 0,
                                 "outstanding": 0, "refused_outstanding": 0}


class TestMalformedStamps:
    async def test_a_stamp_that_does_not_parse_is_skipped_rather_than_crashing(self):
        async with get_sessionmaker()() as db:
            sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CONTAM-02", start_ts_ns=0,
                             end_ts_ns=1, ontology_version="test")
            db.add(sess)
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                         img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
            await db.flush()
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=1, bbox=[1, 1, 9, 9],
                          conf=0.5, source="auto_accept", state="review",
                          provenance={"agent_relabel": ["not in the expected shape"]}))
            await db.commit()
            rows = await agent_relabel_lineages(db, min_count=1)
        assert all(r["from_name"] != "not in the expected shape" for r in rows)


class TestRevertingALineage:
    async def test_a_refused_lineage_goes_back(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 5)])
            out = await revert_lineage(db, "bus", "bmtc_bus_shelter")
        assert out["reverted"] >= 5 and out["run_id"]

    async def test_a_refinement_is_refused_rather_than_bulk_reverted(self):
        """Reverting a judgement call in bulk replaces one bulk opinion with another. Only a move the
        ontology itself forbids is safe to undo mechanically."""
        async with get_sessionmaker()() as db:
            with pytest.raises(ValueError, match="refinement"):
                await revert_lineage(db, "sedan", "mpv")

    async def test_a_human_labelled_object_is_never_swept_up(self):
        async with get_sessionmaker()() as db:
            onto = _onto()
            sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CONTAM-03", start_ts_ns=0,
                             end_ts_ns=1, ontology_version="test")
            db.add(sess)
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                         img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
            await db.flush()
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=onto.by_name("bmtc_bus_shelter").id,
                          bbox=[1, 1, 9, 9], conf=0.5, source="human", state="accepted",
                          provenance={"agent_relabel": ["bus -> bmtc_bus_shelter (0.9)"]}))
            await db.commit()
            await revert_lineage(db, "bus", "bmtc_bus_shelter")
            obj = await db.get(Object, oid)
            await db.refresh(obj)
        assert obj.class_id == _onto().by_name("bmtc_bus_shelter").id, "a person's decision was overwritten"

    async def test_an_object_already_corrected_is_left_alone(self):
        """It no longer carries the class the move produced, so it is somebody else's business now."""
        async with get_sessionmaker()() as db:
            onto = _onto()
            sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CONTAM-04", start_ts_ns=0,
                             end_ts_ns=1, ontology_version="test")
            db.add(sess)
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                         img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
            await db.flush()
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=onto.by_name("hoarding").id,
                          bbox=[1, 1, 9, 9], conf=0.5, source="auto_accept", state="review",
                          provenance={"agent_relabel": ["bus -> bmtc_bus_shelter (0.9)"]}))
            await db.commit()
            await revert_lineage(db, "bus", "bmtc_bus_shelter")
            obj = await db.get(Object, oid)
            await db.refresh(obj)
        assert obj.class_id == _onto().by_name("hoarding").id

    async def test_the_revert_is_itself_revertible(self):
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 3)])
            out = await revert_lineage(db, "bus", "bmtc_bus_shelter")
            back = await revert_run(db, uuid.UUID(out["run_id"]))
        assert back["reverted"] == out["reverted"]

    async def test_restored_objects_are_routed_to_review(self):
        """A class that was wrong and then mechanically restored has not been looked at by anyone."""
        async with get_sessionmaker()() as db:
            onto = _onto()
            await _stamped(db, [("bus", "bmtc_bus_shelter", 3)])
            out = await revert_lineage(db, "bus", "bmtc_bus_shelter")
            rows = (await db.execute(
                select(Object.state).where(Object.class_id == onto.by_name("bus").id,
                                           Object.state == "review"))).scalars().all()
        assert out["reverted"] >= 3 and rows


class TestTheBoardCanReachZero:
    async def test_outstanding_counts_what_is_still_wrong_not_what_once_happened(self):
        """The stamp is history and stays after the object is put right. Counting stamps meant the list read
        3,057 outstanding immediately after 3,010 of them had been restored: a board that can never go
        green, which is the failure this whole session has been about."""
        async with get_sessionmaker()() as db:
            await _stamped(db, [("bus", "bmtc_bus_shelter", 30)])
            before = _find(await agent_relabel_lineages(db, min_count=25), "bus", "bmtc_bus_shelter")
            assert before["outstanding"] >= 30

            await revert_lineage(db, "bus", "bmtc_bus_shelter")
            after = _find(await agent_relabel_lineages(db, min_count=25), "bus", "bmtc_bus_shelter")

        assert after["outstanding"] == 0, "restored objects still counted as work to do"
        assert after["count"] == before["count"], "the history was rewritten instead of the state"

    async def test_an_object_a_person_ruled_on_is_not_outstanding(self):
        """It carries the class, but somebody decided that deliberately. It is not the run's doing any more."""
        async with get_sessionmaker()() as db:
            onto = _onto()
            sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CONTAM-05", start_ts_ns=0,
                             end_ts_ns=1, ontology_version="test")
            db.add(sess)
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                         img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
            await db.flush()
            for _ in range(30):
                db.add(Object(object_id=uuid.uuid4(), frame_id=fid,
                              class_id=onto.by_name("toll_booth").id, bbox=[1, 1, 9, 9], conf=0.5,
                              source="human", state="accepted",
                              provenance={"agent_relabel": ["bus -> toll_booth (0.9)"]}))
            await db.commit()
            row = _find(await agent_relabel_lineages(db, min_count=25), "bus", "toll_booth")
        assert row["outstanding"] == 0

    async def test_the_summary_separates_history_from_work_left(self):
        async with get_sessionmaker()() as db:
            rows = await agent_relabel_lineages(db, min_count=25)
            s = summarize(rows)
        assert "outstanding" in s and "refused_outstanding" in s
        assert s["outstanding"] <= s["objects"]
