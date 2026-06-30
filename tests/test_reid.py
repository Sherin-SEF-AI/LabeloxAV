"""Milestone G: split / merge re-ID. Two tracks of the same object must be temporally disjoint (a shared
frame blocks a merge); a split partitions objects at a boundary; the merged span covers both."""

from __future__ import annotations

from services.temporal.reid import partition_at, span_union, tracks_overlap


def test_disjoint_tracks_can_merge():
    assert tracks_overlap([0, 1, 2], [3, 4, 5]) is False


def test_shared_frame_blocks_merge():
    assert tracks_overlap([0, 1, 2], [2, 3]) is True            # both have a box at ts 2


def test_partition_splits_at_boundary():
    objs = [{"ts_ns": 0}, {"ts_ns": 5}, {"ts_ns": 10}]
    before, after = partition_at(objs, 5)
    assert [o["ts_ns"] for o in before] == [0]                  # boundary is exclusive on the before side
    assert [o["ts_ns"] for o in after] == [5, 10]


def test_partition_boundary_before_all_keeps_before_empty():
    before, after = partition_at([{"ts_ns": 3}, {"ts_ns": 4}], 0)
    assert before == [] and len(after) == 2                     # a no-op split is detectable as empty before


def test_span_union_covers_both():
    assert span_union((0, 5), (3, 12)) == (0, 12)


async def test_split_then_merge_roundtrip_on_real_data():
    import pytest
    from sqlalchemy import func, select

    from db.models import Frame, Object, Review
    from db.session import get_sessionmaker
    from services.temporal.reid import merge_tracks, split_track
    async with get_sessionmaker()() as db:
        row = (await db.execute(
            select(Object.track_id).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Object.track_id.isnot(None)).group_by(Object.track_id)
            .having(func.count(func.distinct(Frame.ts_ns)) >= 2).limit(1))).first()
    if row is None:
        pytest.skip("no multi-frame track in the corpus")
    tid = row[0]
    async with get_sessionmaker()() as db:
        tss = sorted({int(t) for t in (await db.execute(
            select(Frame.ts_ns).join(Object, Object.frame_id == Frame.frame_id)
            .where(Object.track_id == tid))).scalars().all()})
    boundary = tss[len(tss) // 2]

    split = await split_track(tid, boundary, "test")
    new_id = split.get("new_track")
    try:
        assert new_id and split["moved"] >= 1 and split["kept"] >= 1
        async with get_sessionmaker()() as db:                    # the moved objects audit as split_track
            n = (await db.execute(select(func.count()).select_from(Review)
                 .where(Review.action == "split_track"))).scalar()
        assert n >= split["moved"]
    finally:
        back = await merge_tracks(tid, new_id, "test")             # restore the original track
        assert back.get("into") == str(tid) and back["moved"] == split["moved"]
