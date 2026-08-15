"""Every export left here unsplit, and the obvious way to split one is the wrong way.

Consecutive dashcam frames are near-duplicates, so a per-frame train/val split puts the same vehicle on the
same road in both halves. The model is evaluated on what it memorised and the score is inflated by an amount
nobody can recover afterwards. `ExportRecord` has carried `session_id` all along; nothing used it.

Two properties are in tension. Groups are filled in a hash order that does not depend on which other groups
exist, so growth cannot reorder what came before it; but the frame budget depends on the corpus size, so a
train frame can be pulled into val by growth even though an evaluation frame is never released back. The
budget is what makes the realised fractions land near the requested ones, and on this corpus that matters:
377 sessions ranging from 1 to 1,928 frames put 42% of frames in a validation set that asked for 20% when
groups were bucketed independently.
"""

from __future__ import annotations

import uuid

import pytest

from services.export.records import ExportRecord
from services.export.splits import assign_splits, split_summary


def _rec(session_id: uuid.UUID, frame_id: uuid.UUID) -> ExportRecord:
    return ExportRecord(
        object_id=uuid.uuid4(), frame_id=frame_id, session_id=session_id, ts_ns=1, cam_id="cam_f",
        img_uri=f"s3://frames/{frame_id}.jpg", width=1920, height=1080, vehicle_id="V1", city="BLR",
        class_id=1, class_name="car", bbox=[0.0, 0.0, 10.0, 10.0], conf=0.9, state="accepted",
        source="human")


def _corpus(n_sessions: int, frames_each: int = 20, *, first: int = 0) -> list[ExportRecord]:
    """Deterministic ids, so a failure is reproducible rather than a coin toss.

    `first` shifts the session numbering, which is how a test adds sessions the corpus did not already
    have. Without it, a second call returns the same sessions and any growth test silently proves nothing.
    """
    out = []
    for s in range(first, first + n_sessions):
        sid = uuid.UUID(int=s + 1)
        for f in range(frames_each):
            out.append(_rec(sid, uuid.UUID(int=(s + 1) * 1000 + f)))
    return out


class TestWhereItSplits:
    def test_no_session_appears_in_two_splits(self):
        """The whole point: a session is one drive, and its frames are near-duplicates of each other."""
        recs = _corpus(12)
        a = assign_splits(recs, val_frac=0.2, test_frac=0.1, group_by="session", seed="s")
        by_session: dict[str, set[str]] = {}
        for r in recs:
            by_session.setdefault(str(r.session_id), set()).add(a[str(r.frame_id)])
        assert all(len(v) == 1 for v in by_session.values()), "a session was split across sets"

    def test_grouping_by_frame_is_available_and_does_split_frames(self):
        """Offered for corpora that are not drives, and named honestly so nobody picks it by accident."""
        recs = _corpus(2, frames_each=40)
        a = assign_splits(recs, val_frac=0.3, group_by="frame", seed="s")
        assert len(set(a.values())) > 1

    def test_every_frame_lands_somewhere_exactly_once(self):
        recs = _corpus(6)
        a = assign_splits(recs, val_frac=0.2, test_frac=0.2, seed="s")
        assert set(a) == {str(r.frame_id) for r in recs}
        assert set(a.values()) <= {"train", "val", "test"}


class TestStabilityUnderGrowth:
    def test_growth_never_takes_a_frame_out_of_val_or_test(self):
        """The property that actually holds, and the one that matters.

        Groups are filled in a hash order that does not depend on which other groups exist, so growth can
        only pull further groups into val and test; it never releases one back to train. A frame that was
        an evaluation frame stays one. The reverse direction is possible and is why a release pins its own
        assignment in splits.json rather than recomputing it.

        A shuffle-based splitter holds neither direction: it reshuffles everything on insertion.
        """
        base = _corpus(8)
        grown = base + _corpus(4, first=8)        # four sessions the corpus did not have
        a = assign_splits(base, val_frac=0.2, test_frac=0.1, seed="s")
        b = assign_splits(grown, val_frac=0.2, test_frac=0.1, seed="s")

        moved_out = [fid for fid, split in a.items()
                     if split in ("val", "test") and b[fid] == "train"]
        assert not moved_out, f"{len(moved_out)} evaluation frames were released back into training"

    def test_a_group_is_never_cut_in_half_by_a_budget(self):
        """The budget stops the walk, but a group that overshoots it still goes in whole: splitting a
        session is the leak the grouping exists to prevent."""
        recs = _corpus(2, frames_each=50) + _corpus(1, frames_each=1, first=2)
        a = assign_splits(recs, val_frac=0.05, seed="s")
        by_group: dict[str, set[str]] = {}
        for r in recs:
            by_group.setdefault(str(r.session_id), set()).add(a[str(r.frame_id)])
        assert all(len(v) == 1 for v in by_group.values())

    def test_the_same_input_and_seed_give_the_same_split(self):
        recs = _corpus(5)
        assert assign_splits(recs, val_frac=0.2, seed="x") == assign_splits(recs, val_frac=0.2, seed="x")

    def test_a_different_seed_gives_a_different_split(self):
        recs = _corpus(20)
        assert assign_splits(recs, val_frac=0.3, seed="x") != assign_splits(recs, val_frac=0.3, seed="y")

    def test_the_split_does_not_depend_on_record_order(self):
        recs = _corpus(6)
        assert assign_splits(recs, val_frac=0.25, seed="s") == \
               assign_splits(list(reversed(recs)), val_frac=0.25, seed="s")


class TestAskingForNothing:
    def test_no_split_requested_means_every_frame_is_train(self):
        """An export that asks for no split must be untouched, not quietly reorganised."""
        recs = _corpus(4)
        a = assign_splits(recs)
        assert set(a.values()) == {"train"}
        assert len(a) == len({str(r.frame_id) for r in recs})

    def test_an_empty_corpus_is_not_an_error(self):
        assert assign_splits([], val_frac=0.2) == {}


class TestRefusals:
    def test_it_refuses_to_leave_no_training_data(self):
        with pytest.raises(ValueError, match="training set"):
            assign_splits(_corpus(3), val_frac=0.6, test_frac=0.5)

    def test_it_refuses_a_negative_fraction(self):
        with pytest.raises(ValueError, match="negative"):
            assign_splits(_corpus(3), val_frac=-0.1)

    def test_it_refuses_an_unknown_grouping(self):
        # Track grouping in particular: a frame holds many tracks, so grouping by track would put one
        # frame's image in two splits at once, which is the leak this module exists to prevent.
        with pytest.raises(ValueError, match="group_by"):
            assign_splits(_corpus(3), val_frac=0.2, group_by="track")


class TestTheSummary:
    def test_it_reports_what_happened_not_what_was_asked(self):
        """Hash bucketing over uneven groups cannot hit a fraction exactly, and a reader comparing two
        datasets needs the number that happened."""
        recs = _corpus(10)
        a = assign_splits(recs, val_frac=0.2, test_frac=0.1, seed="s")
        s = split_summary(recs, a, group_by="session", seed="s", val_frac=0.2, test_frac=0.1)

        assert s["requested"] == {"train": pytest.approx(0.7), "val": 0.2, "test": 0.1}
        assert sum(s["frames"].values()) == len({str(r.frame_id) for r in recs})
        assert sum(s["objects"].values()) == len(recs)
        assert s["group_by"] == "session" and s["seed"] == "s"

    def test_the_realised_fractions_land_near_the_requested_ones(self):
        """Independent per-group bucketing failed this badly on the real corpus, which is what the frame
        budget exists to fix."""
        recs = _corpus(40, frames_each=25)
        a = assign_splits(recs, val_frac=0.2, test_frac=0.1, seed="s")
        s = split_summary(recs, a, group_by="session", seed="s", val_frac=0.2, test_frac=0.1)
        assert abs(s["realised"]["val"] - 0.2) < 0.05
        assert abs(s["realised"]["test"] - 0.1) < 0.05

    def test_uneven_groups_still_land_near_the_request(self):
        """Session sizes here span three orders of magnitude, so one big drive must not decide the split."""
        recs = _corpus(20, frames_each=5) + _corpus(3, frames_each=200, first=20)
        a = assign_splits(recs, val_frac=0.2, seed="s")
        s = split_summary(recs, a, group_by="session", seed="s", val_frac=0.2, test_frac=0.0)
        assert 0.05 < s["realised"]["val"] < 0.45, f"one group swamped the split: {s['realised']}"

    def test_it_names_any_group_that_leaked_across_splits(self):
        """Asserted in the artifact rather than left for a reader to check."""
        recs = _corpus(8)
        a = assign_splits(recs, val_frac=0.2, test_frac=0.2, group_by="session", seed="s")
        s = split_summary(recs, a, group_by="session", seed="s", val_frac=0.2, test_frac=0.2)
        assert s["groups_shared_between_splits"] == []

    def test_a_frame_grouping_shows_the_leak_it_causes(self):
        # Grouping by frame means the group IS the frame, so nothing is shared: the leak from frame
        # grouping is between near-duplicate frames, which no group check can see. Said plainly here so
        # the empty list is not read as a safety guarantee.
        recs = _corpus(3, frames_each=30)
        a = assign_splits(recs, val_frac=0.3, group_by="frame", seed="s")
        s = split_summary(recs, a, group_by="frame", seed="s", val_frac=0.3, test_frac=0.0)
        assert s["groups_shared_between_splits"] == []
        assert s["groups"]["val"] == s["frames"]["val"]
