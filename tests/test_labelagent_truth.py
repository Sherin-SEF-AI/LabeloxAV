"""The label agent's answer key and its scoring.

Every number the agent is judged by comes from these two pieces, so the rules are pinned here: only a
person's last word is truth, machine rows never count, a box nobody predicted is not graded, and scoring keeps
"fixed a wrong label" apart from "broke a right one", because the second decides whether the agent may act.
"""

from __future__ import annotations

from services.labelagent.benchmark import Prediction, outcome, score, wilson_upper, with_floor
from services.labelagent.truth import NOT_OBJECT, TruthItem, derive, split_of

NOT_IDS = frozenset({229, 155})
SID, FID = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"


def rv(action, reviewer="admin", before=None, ts=1):
    return {"action": action, "reviewer": reviewer, "before": before, "after": None, "ts_ns": ts}


MACHINE = {"class_id": 16, "source": "fused", "bbox": [0, 0, 10, 10], "conf": 0.4, "provenance": {"x": 1}}


def obj(class_id=107, state="review", source="human"):
    return {"object_id": "o1", "class_id": class_id, "state": state, "source": source}


def d(reviews, current):
    return derive(reviews, current, not_object_ids=NOT_IDS, session_id=SID, frame_id=FID)


# ------------------------------------------------------------------------------------------------ truth

def test_a_reclassification_records_what_the_machine_said_and_what_the_person_ruled():
    it = d([rv("reclassify", before=MACHINE)], obj())
    assert (it.pred, it.truth, it.tier, it.pred_source) == (16, 107, "direct", "fused")
    assert it.bbox == [0.0, 0.0, 10.0, 10.0] and it.pred_conf == 0.4 and it.provenance == {"x": 1}


def test_a_reject_or_undefined_means_not_an_object():
    assert d([rv("reject", before=MACHINE)], obj(state="rejected")).truth == NOT_OBJECT
    assert d([rv("reclassify", before=MACHINE)], obj(class_id=229)).truth == NOT_OBJECT


def test_machine_rows_are_never_truth():
    """The track backfill copies one ruling onto a whole track; an agent's row is an opinion."""
    assert d([rv("reclassify_track_backfill", reviewer="track_relabel_backfill", before=MACHINE)], obj()) is None
    assert d([rv("reclassify", before=MACHINE), rv("flag", reviewer="vlm-qa", ts=2)], obj()) is None


def test_the_person_must_have_had_the_last_word():
    assert d([rv("reclassify", before=MACHINE), rv("revert", ts=2)], obj()) is None


def test_a_box_a_person_drew_has_no_prediction_to_grade():
    assert d([rv("create", before=None)], obj()) is None
    assert d([rv("reclassify", before={**MACHINE, "source": "human"})], obj()) is None


def test_a_history_that_starts_with_a_revert_is_not_the_machines_label():
    assert d([rv("revert", before=MACHINE), rv("reclassify", ts=2)], obj()) is None


def test_a_track_ruling_is_kept_apart_from_a_direct_one():
    assert d([rv("reclassify_track", before=MACHINE)], obj()).tier == "track"


def test_an_accept_is_a_correct_prediction():
    it = d([rv("accept", before={**MACHINE, "class_id": 107})], obj(state="accepted"))
    assert it.correct


def test_the_split_is_by_session_and_stable():
    assert split_of(SID) == split_of(SID)
    sides = {split_of(f"{n:08d}-0000-0000-0000-000000000000") for n in range(200)}
    assert sides == {"test", "calib"}


# ------------------------------------------------------------------------------------------------ scoring

def item(pred, truth, tier="direct", source="fused", split="test"):
    return TruthItem(object_id="o", frame_id=FID, session_id=SID, truth=truth, pred=pred, pred_source=source,
                     tier=tier, split=split, bbox=[0, 0, 10, 10])


def test_the_four_outcomes():
    assert outcome(item(1, 1), Prediction(None)) == "kept"
    assert outcome(item(1, 1), Prediction(2)) == "broke"
    assert outcome(item(1, 2), Prediction(2)) == "fixed"
    assert outcome(item(1, 2), Prediction(3)) == "changed_wrong"
    assert outcome(item(1, 2), Prediction(None)) == "missed"
    assert outcome(item(1, 2), Prediction(1)) == "missed"   # "changing" to the same label keeps it


class _C:
    def __init__(self, id, name, l1):
        self.id, self.name, self.l1 = id, name, l1


class _Onto:
    classes = [_C(1, "truck", "heavy"), _C(2, "autorickshaw", "three_wheeler"), _C(229, "undefined", "fallback")]

    def by_id(self, cid):
        for c in self.classes:
            if c.id == cid:
                return c
        raise KeyError(cid)


def test_score_separates_fixes_from_breaks_and_counts_not_object():
    items = [item(1, 2), item(1, 2), item(1, 1), item(1, 1), item(1, NOT_OBJECT)]
    preds = [Prediction(2), Prediction(None), Prediction(2), Prediction(None), Prediction(229)]
    h = score(items, preds, _Onto())["headline"]
    assert h["outcomes"] == {"fixed": 2, "missed": 1, "broke": 1, "kept": 1}
    assert h["fix_rate"] == round(2 / 3, 4) and h["wrong_fix_rate"] == 0.5
    # an answer of `undefined` is a "not an object" catch, not a confusion
    assert h["not_object_catch"] == 1.0
    assert h["change_precision"] == round(2 / 3, 4)


def test_score_keeps_tiers_apart_and_respects_the_split():
    items = [item(1, 2), item(1, 1, tier="keep", source="human"), item(1, 2, split="calib")]
    rep = score(items, [Prediction(2)] * 3, _Onto(), split="test")
    assert rep["headline"]["n"] == 1 and rep["keep"]["n"] == 1
    assert rep["keep"]["wrong_fix_rate"] == 1.0
    assert rep["pooled"]["n"] == 2


def test_the_wrong_fix_bound_is_honest_about_small_samples():
    assert wilson_upper(0, 137) > 0.02          # zero breaks in 137 still cannot promise 2%
    assert wilson_upper(0, 685) < 0.01
    assert wilson_upper(0, 0) is None


def test_a_floor_keeps_labels_the_stage_was_unsure_about():
    got = with_floor([Prediction(2, 0.9), Prediction(2, 0.3), Prediction(None)], 0.5)
    assert [p.label for p in got] == [2, None, None]
