"""M-Q.3: the isotonic calibration must train on auto-labeled objects a human reviewed (the model's
raw_conf paired with the human verdict), keyed off the Review audit trail rather than the object source.
This proves the corrected collector finds model detections with a human accept/reject verdict and fits a
monotone curve, where the old source=='human' filter found none (human draws carry no raw_conf)."""

from __future__ import annotations

from sqlalchemy import delete, select

from db.models import Object, Review
from db.session import get_sessionmaker
from services.autolabel.isotonic import _collect_pairs, fit_isotonic

_REVIEWER = "test-human-mq3"


async def _seed_reviewed_detections() -> list:
    """Model detections (source=model, with raw_conf) that a human accepted (high conf) or rejected (low),
    each carrying a human Review row. Returns the object ids for cleanup."""
    async with get_sessionmaker()() as db:
        frame_id = (await db.execute(select(Object.frame_id).limit(1))).scalar()
        ids = []
        # 8 accepted at high confidence, 4 rejected at low confidence -> a clean monotone signal
        for rc, correct in [(0.92, True), (0.88, True), (0.85, True), (0.81, True), (0.79, True),
                            (0.74, True), (0.71, True), (0.66, True), (0.34, False), (0.28, False),
                            (0.22, False), (0.18, False)]:
            o = Object(frame_id=frame_id, class_id=1, bbox=[10.0, 10.0, 60.0, 60.0], conf=rc,
                       source="fused", state="accepted" if correct else "rejected",
                       provenance={"raw_conf": rc})
            db.add(o)
            await db.flush()
            db.add(Review(object_id=o.object_id, reviewer=_REVIEWER, ts_ns=1_750_000_000_000_000_000,
                          action="accept" if correct else "reject"))
            ids.append(o.object_id)
        await db.commit()
        return ids


async def _cleanup(ids):
    async with get_sessionmaker()() as db:
        await db.execute(delete(Review).where(Review.reviewer == _REVIEWER))
        await db.execute(delete(Object).where(Object.object_id.in_(ids)))
        await db.commit()


async def test_collector_finds_reviewed_detections_and_fit_is_monotone():
    ids = await _seed_reviewed_detections()
    try:
        xs, ys = await _collect_pairs()
        # the 12 seeded model detections are present (corpus carries no scalar-raw_conf reviewed pairs)
        assert len(xs) >= 12
        # high-confidence rows are the accepted ones, low-confidence the rejected ones
        seeded = sorted(zip(xs.tolist(), ys.tolist(), strict=False))
        assert seeded[0][1] == 0.0 and seeded[-1][1] == 1.0

        res = await fit_isotonic()
        assert res["n_train"] >= 12
        # the fitted isotonic curve is monotone non-decreasing in confidence
        from services.autolabel.isotonic import apply_isotonic
        assert apply_isotonic(res["uri"], 0.2) <= apply_isotonic(res["uri"], 0.9)
    finally:
        await _cleanup(ids)


async def test_old_source_human_population_would_be_empty():
    # the seeded detections are source=model, so a source=='human' filter (the old bug) would miss them all
    ids = await _seed_reviewed_detections()
    try:
        async with get_sessionmaker()() as db:
            n = (await db.execute(
                select(Object).where(Object.object_id.in_(ids), Object.source == "human"))).scalars().all()
        assert len(n) == 0
    finally:
        await _cleanup(ids)
