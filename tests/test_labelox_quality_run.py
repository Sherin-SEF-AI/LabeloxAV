"""`annotation_quality` has never had a row in it.

The table is read by the data lake export and its own model docstring says it is "surfaced in the workspace so
a reviewer sees which labels to trust". It holds zero rows and has since the migration that created it. The
scoring in quality.py is pure, tested and correct; nothing ever called it with a database attached.

An absent trust signal is not neutral. It renders as nothing wrong here, when what it means is nobody looked,
which is the same shape as the analytics page reporting an empty corpus while loading and the benchmark rows
naming artifacts that were never uploaded.

Two things needed pinning. The scorer's defaults assume 1920x1080, so an off-screen flag is meaningless
unless the frame's real size reaches it. And agreement must stay null where only one person reviewed an
object: on this corpus that is every object, since 557 reviews exist and no object has been touched by two
different reviewers, so a default of 1.0 would silently inflate every quality score in the table.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from core.timebase import now_ns
from db.models import AnnotationQuality, Frame, Object, Review
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.labelox.quality_run import score_corpus

pytestmark = pytest.mark.db


async def _seed(db, *, boxes: list[list[float]], width: int = 1920, height: int = 1080,
                source: str = "human", conf: float = 0.9) -> list[uuid.UUID]:
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-AQ", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/t.jpg", width=width, height=height)
    db.add(frame)
    await db.flush()
    ids = []
    for b in boxes:
        o = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=1, bbox=b,
                   conf=conf, source=source, state="accepted")
        db.add(o)
        ids.append(o.object_id)
    await db.commit()
    return ids


async def _quality(db, object_id):
    return (await db.execute(
        select(AnnotationQuality).where(AnnotationQuality.object_id == object_id))).scalar_one_or_none()


async def test_scoring_actually_writes_rows():
    """The whole gap. Everything below depends on this having happened at all."""
    async with get_sessionmaker()() as db:
        before = (await db.execute(select(func.count()).select_from(AnnotationQuality))).scalar() or 0
        ids = await _seed(db, boxes=[[10, 10, 200, 200]])
        out = await score_corpus(db, only_missing=True)
        after = (await db.execute(select(func.count()).select_from(AnnotationQuality))).scalar() or 0
    assert out["scored_this_run"] >= 1
    assert after > before
    assert ids


async def test_a_clean_human_box_scores_high_and_carries_no_flags():
    async with get_sessionmaker()() as db:
        ids = await _seed(db, boxes=[[10, 10, 200, 200]])
        await score_corpus(db, only_missing=True)
        row = await _quality(db, ids[0])
    assert row is not None
    assert row.quality > 0.5
    assert row.flags == []


async def test_a_degenerate_box_is_flagged_and_scored_down():
    async with get_sessionmaker()() as db:
        ids = await _seed(db, boxes=[[10, 10, 12, 12]])
        await score_corpus(db, only_missing=True)
        row = await _quality(db, ids[0])
    assert "tiny_box" in row.flags
    assert row.quality < 0.9


async def test_off_screen_is_judged_against_the_frame_not_a_default_resolution():
    """The scorer defaults to 1920x1080. On a 640x480 frame a box at x=1000 is off the image, and without
    the real size reaching the scorer it is silently called fine."""
    async with get_sessionmaker()() as db:
        ids = await _seed(db, boxes=[[1000, 100, 1200, 300]], width=640, height=480)
        await score_corpus(db, only_missing=True)
        row = await _quality(db, ids[0])
    assert "off_screen" in row.flags


async def test_agreement_stays_null_when_only_one_person_reviewed():
    """Every object on this corpus. A default of 1.0 would score one unchallenged opinion the same as four
    independent agreeing ones, inflating the entire table."""
    async with get_sessionmaker()() as db:
        ids = await _seed(db, boxes=[[10, 10, 200, 200]])
        db.add(Review(review_id=uuid.uuid4(), object_id=ids[0], reviewer="alice", action="confirm", ts_ns=now_ns()))
        await db.commit()
        await score_corpus(db, only_missing=True)
        row = await _quality(db, ids[0])
    assert row.agreement is None


async def test_two_reviewers_who_disagree_produce_an_agreement_below_one():
    async with get_sessionmaker()() as db:
        ids = await _seed(db, boxes=[[10, 10, 200, 200]])
        db.add(Review(review_id=uuid.uuid4(), object_id=ids[0], reviewer="alice", action="confirm", ts_ns=now_ns()))
        db.add(Review(review_id=uuid.uuid4(), object_id=ids[0], reviewer="bob", action="reclassify", ts_ns=now_ns()))
        await db.commit()
        await score_corpus(db, only_missing=True)
        row = await _quality(db, ids[0])
    assert row.agreement is not None and row.agreement < 1.0


async def test_rescoring_replaces_rather_than_duplicating():
    """Objects here are relabelled constantly, so a second run must not leave two verdicts side by side."""
    async with get_sessionmaker()() as db:
        ids = await _seed(db, boxes=[[10, 10, 200, 200]])
        await score_corpus(db, only_missing=True)
        await score_corpus(db, only_missing=False)
        n = (await db.execute(select(func.count()).select_from(AnnotationQuality)
                              .where(AnnotationQuality.object_id == ids[0]))).scalar()
    assert n == 1


async def test_the_run_reports_how_much_of_it_rested_on_a_single_opinion():
    """Zero agreement across the whole table is a fact a reader must be told, not one they must notice."""
    async with get_sessionmaker()() as db:
        await _seed(db, boxes=[[10, 10, 200, 200]])
        out = await score_corpus(db, only_missing=True)
    assert "rows_with_agreement" in out
    assert "more than one" in out["detail"]


async def test_a_run_larger_than_one_batch_scores_every_object():
    """The first corpus run stopped at 286,000 of 570,505 and called it done.

    `only_missing` filters to objects with no quality row, and the loop writes exactly those rows, so with
    OFFSET pagination the result set shrank underneath a growing offset and everything that slid past the
    cursor was skipped. A batch smaller than the seeded set reproduces it.
    """
    async with get_sessionmaker()() as db:
        ids = await _seed(db, boxes=[[10, 10, 100 + i, 100 + i] for i in range(25)])
        await score_corpus(db, only_missing=True, batch=5)
        scored = (await db.execute(
            select(func.count()).select_from(AnnotationQuality)
            .where(AnnotationQuality.object_id.in_(ids)))).scalar()
    assert scored == len(ids)
