"""The validation split must measure generalisation, and the trainset must stay disjoint from the yardstick.

Two defects made every reported number optimistic by an unknown margin:

1. The split was per frame. Consecutive dashcam frames are a fraction of a second apart and therefore
   near-duplicates, so a random per-frame split trains and validates on the same scene and the model is
   scored on images it has effectively already seen.
2. Nothing prevented the trainset and a sealed gold set from sharing objects. Both draw from the same Object
   table with overlapping predicates (human-reviewed and accepted), so a model could be trained on its own
   yardstick, which makes every gold metric it produces meaningless.

Pure unit tests over the split function: no database, no images."""
from __future__ import annotations

import random

from services.training.dataset_builder import _split_val_frames


def _corpus(n_sessions: int, frames_per_session: int) -> dict[str, list[dict]]:
    """frame_id -> [candidate objects], each carrying its session (what the split groups on)."""
    return {
        f"s{s}_f{f}": [{"session_id": f"sess-{s}"}]
        for s in range(n_sessions)
        for f in range(frames_per_session)
    }


def _sessions_of(by_frame: dict[str, list[dict]], frames) -> set[str]:
    return {by_frame[f][0]["session_id"] for f in frames}


def test_session_grouped_split_never_puts_a_session_on_both_sides():
    by_frame = _corpus(n_sessions=5, frames_per_session=10)
    val = _split_val_frames(by_frame, set(), 10, random.Random(7), group_by_session=True)
    train = set(by_frame) - val

    val_sessions = _sessions_of(by_frame, val)
    train_sessions = _sessions_of(by_frame, train)
    assert val_sessions, "the validation set must not be empty"
    assert not (val_sessions & train_sessions), "a session on both sides is the leak this fixes"


def test_per_frame_split_does_leak_which_is_why_the_default_changed():
    # Demonstrating the defect rather than asserting it in prose: the old behaviour splits mid-session.
    by_frame = _corpus(n_sessions=5, frames_per_session=10)
    val = _split_val_frames(by_frame, set(), 10, random.Random(7), group_by_session=False)
    train = set(by_frame) - val
    assert _sessions_of(by_frame, val) & _sessions_of(by_frame, train)


def test_split_is_deterministic_for_a_given_seed():
    # A validation set that moves between runs makes two training runs incomparable.
    by_frame = _corpus(4, 8)
    a = _split_val_frames(by_frame, set(), 8, random.Random(3), group_by_session=True)
    b = _split_val_frames(by_frame, set(), 8, random.Random(3), group_by_session=True)
    assert a == b


def test_gold_frames_are_preferred_for_validation():
    # Gold is the cleanest available label, so it belongs on the side that measures.
    by_frame = _corpus(4, 10)
    gold_frames = {f for f in by_frame if by_frame[f][0]["session_id"] == "sess-2"}
    val = _split_val_frames(by_frame, gold_frames, 10, random.Random(1), group_by_session=True)
    assert gold_frames <= val


def test_single_session_corpus_still_produces_a_validation_set():
    # There is nothing to hold out at session granularity, and an empty val set would silently disable
    # validation altogether, so the split degrades to per-frame rather than returning nothing.
    by_frame = _corpus(n_sessions=1, frames_per_session=20)
    val = _split_val_frames(by_frame, set(), 4, random.Random(5), group_by_session=True)
    assert 0 < len(val) < len(by_frame)


def test_zero_budget_yields_no_validation_frames():
    by_frame = _corpus(3, 5)
    assert _split_val_frames(by_frame, set(), 0, random.Random(0), group_by_session=True) == set()


def test_build_spec_defaults_to_the_safe_split():
    from services.training.dataset_builder import BuildSpec

    spec = BuildSpec()
    assert spec.group_split_by_session is True, "the leaky split must not be the default"
    assert spec.exclude_gold_id is None
