"""Counterfactual perturbations, the drop that counts as one, and the simulator that is not installed.

Two kinds of thing are tested here. The perturbations, whose whole value depends on being deterministic
and on actually changing what they claim to; and the statistics, where the failure mode is a gate that
refuses a promotion because four objects moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.verdyx.counterfactual import (
    MAX_COUNTERFACTUAL_DROP,
    MIN_SUPPORT,
    drop_with_interval,
    match_recall,
    summarise,
)
from services.verdyx.perturb import DEFAULT_STRENGTHS, apply, dusk, fog, occlude, rain


def _frame(h: int = 120, w: int = 200, value: int = 128) -> np.ndarray:
    """A mid-grey frame with a little texture, so a blur has something to blur."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    img[::7, :, :] = 200
    img[:, ::11, :] = 60
    return img


class TestPerturbationsAreDeterministic:
    """A counterfactual whose result moves between runs cannot be compared to the previous run, which is
    exactly the situation where somebody wants to check a finding."""

    @pytest.mark.parametrize("name", sorted(DEFAULT_STRENGTHS))
    def test_the_same_seed_gives_the_same_image(self, name):
        img = _frame()
        box = [20.0, 20.0, 120.0, 100.0]
        a = apply(name, img, box=box, seed=11)
        b = apply(name, img, box=box, seed=11)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("name", ["occlude", "rain", "motion_blur"])
    def test_a_different_seed_gives_a_different_image(self, name):
        """A seeded perturbation that ignores its seed is a perturbation with one sample, and a
        measurement over it is a measurement of that one arrangement."""
        img = _frame()
        box = [20.0, 20.0, 120.0, 100.0]
        assert not np.array_equal(apply(name, img, box=box, seed=1),
                                  apply(name, img, box=box, seed=99))

    @pytest.mark.parametrize("name", sorted(DEFAULT_STRENGTHS))
    def test_the_shape_and_type_survive(self, name):
        img = _frame()
        out = apply(name, img, box=[10.0, 10.0, 90.0, 90.0])
        assert out.shape == img.shape and out.dtype == np.uint8

    @pytest.mark.parametrize("name", sorted(DEFAULT_STRENGTHS))
    def test_each_one_actually_changes_the_frame(self, name):
        """A perturbation that changes nothing measures nothing, and a zero drop from it would read as
        robustness."""
        img = _frame()
        out = apply(name, img, box=[10.0, 10.0, 90.0, 90.0])
        assert not np.array_equal(out, img), f"{name} left the frame untouched"

    def test_an_unknown_perturbation_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown perturbation"):
            apply("snow", _frame())


class TestWhatEachPerturbationDoes:
    def test_occlusion_covers_the_box_and_leaves_the_rest_alone(self):
        img = _frame()
        out = occlude(img, [20.0, 20.0, 120.0, 100.0], frac=0.5, seed=3)
        assert np.array_equal(out[:15, :15], img[:15, :15]), "outside the box is untouched"
        assert (out[20:100, 20:120] == 0).any(), "something inside the box was covered"

    def test_occlusion_comes_from_an_edge_rather_than_the_centre(self):
        """A centred hole leaves the object's outline intact on all four sides, which is a much easier
        problem than the one being asked about."""
        img = _frame()
        out = occlude(img, [0.0, 0.0, 200.0, 120.0], frac=0.3, seed=5)
        black = (out == 0).all(axis=2)
        rows, cols = np.where(black)
        touches_edge = (rows.min() == 0 or rows.max() == 119 or cols.min() == 0 or cols.max() == 199)
        assert touches_edge

    def test_occluding_nothing_is_a_no_op_rather_than_an_error(self):
        img = _frame()
        assert np.array_equal(occlude(img, [50.0, 50.0, 50.0, 50.0], frac=0.5), img)

    def test_dusk_darkens_rather_than_clipping_the_shadows(self):
        """Subtracting a constant clips the shadows to black and leaves the highlights linear, which is
        not what a camera does at dusk."""
        img = _frame()
        out = dusk(img, gamma=0.45)
        assert out.mean() < img.mean()
        assert out.max() > 0, "the highlights survive"
        assert (out == 0).mean() < 0.5, "the shadows are compressed, not clipped away"

    def test_fog_thickens_with_distance_not_with_nearness(self):
        """Distance grows toward the horizon in a forward frame. Written the other way round first, which
        put the haze on the bonnet and left the horizon clear: fog that thins with distance."""
        img = _frame(h=120, w=200)
        out = fog(img, strength=0.6)
        h = img.shape[0]
        far = np.abs(out[:h // 3].astype(int) - img[:h // 3].astype(int)).mean()
        near = np.abs(out[2 * h // 3:].astype(int) - img[2 * h // 3:].astype(int)).mean()
        assert far > near

    def test_fog_uses_a_depth_map_when_given_one(self):
        img = _frame()
        flat = np.full(img.shape[:2], 5.0, dtype=np.float32)
        distant = np.full(img.shape[:2], 90.0, dtype=np.float32)
        near_out = fog(img, strength=0.6, depth_m=flat)
        far_out = fog(img, strength=0.6, depth_m=distant)
        assert np.abs(far_out.astype(int) - img.astype(int)).mean() > \
               np.abs(near_out.astype(int) - img.astype(int)).mean()

    def test_rain_adds_both_streaks_and_a_veil(self):
        """Streaks alone are salt-and-pepper noise a convolution shrugs off; the veil is what costs a
        detector its contrast."""
        img = _frame(value=40)
        out = rain(img, strength=0.8, seed=4)
        assert out.mean() > img.mean(), "the veil lifts the blacks"
        assert out.std() != pytest.approx(img.std(), abs=0.5), "the streaks add structure"


class TestTheDropThatCountsAsOne:
    def test_a_large_drop_on_a_large_sample_is_significant(self):
        got = drop_with_interval(90, 100, 60, 100)
        assert got["drop"] == pytest.approx(0.30)
        assert got["significant_drop"] is True

    def test_four_objects_moving_is_not_a_regression(self):
        """3 of 4 falling to 2 of 4 is not a 25% regression, it is four objects, and a gate refusing on
        that is refusing on noise."""
        got = drop_with_interval(3, 4, 2, 4)
        assert got["significant_drop"] is False

    def test_an_improvement_is_never_a_significant_drop(self):
        got = drop_with_interval(60, 100, 90, 100)
        assert got["drop"] < 0
        assert got["significant_drop"] is False

    def test_no_gold_objects_is_unmeasured_rather_than_a_perfect_score(self):
        got = drop_with_interval(0, 0, 0, 0)
        assert got["measured"] is False and got["reason"]

    def test_a_thin_class_is_excluded_with_its_support(self):
        per_class = {"cattle": drop_with_interval(8, 10, 2, 10)}
        out = summarise(per_class)
        assert out["blocking"] == []
        assert out["unmeasured"] and str(MIN_SUPPORT) in out["unmeasured"][0]["reason"]

    def test_a_big_enough_class_falling_far_enough_blocks(self):
        per_class = {"rider": drop_with_interval(95, 100, 50, 100)}
        out = summarise(per_class)
        assert [b["class_name"] for b in out["blocking"]] == ["rider"]

    def test_a_small_drop_does_not_block_however_certain(self):
        per_class = {"rider": drop_with_interval(1000, 1000, 950, 1000)}
        out = summarise(per_class, max_drop=MAX_COUNTERFACTUAL_DROP)
        assert out["blocking"] == []

    def test_only_safety_classes_block_when_a_set_is_given(self):
        per_class = {"rider": drop_with_interval(95, 100, 50, 100),
                     "hoarding": drop_with_interval(95, 100, 50, 100)}
        out = summarise(per_class, safety_classes={"rider"})
        assert [b["class_name"] for b in out["blocking"]] == ["rider"]


class TestRecallMatching:
    def test_one_prediction_covers_one_gold_box(self):
        assert match_recall([[0, 0, 10, 10], [50, 50, 60, 60]], [[0, 0, 10, 10]]) == (1, 2)

    def test_two_predictions_on_one_box_do_not_cover_two(self):
        assert match_recall([[0, 0, 10, 10]], [[0, 0, 10, 10], [1, 1, 11, 11]]) == (1, 1)

    def test_no_predictions_covers_nothing(self):
        assert match_recall([[0, 0, 10, 10]], []) == (0, 1)


class TestTheGateClause:
    def _metrics(self, cf=None):
        m = {"map50": 0.5, "safe_miou": 0.8, "per_class": {}, "recapture": {"ok": True, "checked": True}}
        if cf is not None:
            m["counterfactual"] = cf
        return m

    def test_no_evaluation_blocks_nothing(self):
        from services.govern.champion import _counterfactual

        assert _counterfactual(self._metrics())["ok"] is True
        assert _counterfactual(self._metrics({"measured": False, "reason": "not run"}))["ok"] is True

    def test_a_measured_collapse_blocks_with_the_numbers_in_the_reason(self):
        from services.govern.champion import _counterfactual

        cf = {"measured": True, "perturbations": {"dusk": {"blocking": [
            {"class_name": "rider", "drop": 0.42, "recall_before": 0.9, "recall_after": 0.48,
             "support": 120}]}}}
        out = _counterfactual(self._metrics(cf))
        assert out["ok"] is False
        assert "rider" in out["reasons"][0] and "dusk" in out["reasons"][0]

    def test_surviving_every_perturbation_does_not_promote_on_its_own(self):
        """Surviving occlusion says nothing about mAP. The clause is one-directional, like the shadow
        clause beside it."""
        from core.config import get_settings
        from services.autolabel.ontology import get_ontology
        from services.govern.champion import _counterfactual, champion_gate

        cf = {"measured": True, "perturbations": {"dusk": {"blocking": []}}}
        assert _counterfactual(self._metrics(cf))["ok"] is True
        weak = {**self._metrics(cf), "map50": 0.10}
        champ = {"map50": 0.60, "safe_miou": 0.8, "per_class": {}}
        out = champion_gate(weak, champ, get_ontology(), get_settings().phase4.govern)
        assert out["promote"] is False


class TestTheSimulatorIsNotInstalled:
    def test_the_capability_refuses_with_what_is_missing_and_where_to_put_it(self):
        from services.forgyx.capabilities import CapabilityError, require

        try:
            require("esmini")
        except CapabilityError as exc:
            assert "esmini" in str(exc) and "sim.esmini_bin" in str(exc)

    def test_replay_returns_a_reason_rather_than_raising(self):
        """A missing simulator is a fact about the host, not an error in the caller. A scenario nobody
        could validate and one that failed validation are different, and an export that conflated them
        would claim a check it never performed."""
        from services.sim.esmini_runner import replay

        res = replay("/tmp/does-not-exist.xosc")
        assert res.ok is False and res.reason
        assert res.as_dict()["ok"] is False

    def test_the_csv_parser_survives_a_renamed_column(self):
        """A parser that returns nothing when the simulator renames a heading makes every scenario look
        like one where nobody moved."""
        from services.sim.esmini_runner import parse_csv_log

        got = parse_csv_log("time,entity,pos_x,pos_y\n0.0,Ego,1.0,2.0\n0.1,Ego,2.0,2.0\n")
        assert got == {"Ego": [(0.0, 1.0, 2.0), (0.1, 2.0, 2.0)]}

    def test_the_checksum_ignores_a_last_decimal_place(self):
        """A checksum that changed on a simulator patch release would report every scenario as regressed
        on the day of an upgrade."""
        from services.sim.esmini_runner import trajectory_checksum

        a = {"Ego": [(0.0, 1.0001, 2.0001)]}
        b = {"Ego": [(0.0, 1.0004, 2.0002)]}
        assert trajectory_checksum(a) == trajectory_checksum(b)

    def test_two_different_runs_do_not_share_a_checksum(self):
        from services.sim.esmini_runner import trajectory_checksum

        assert trajectory_checksum({"Ego": [(0.0, 1.0, 2.0)]}) != \
               trajectory_checksum({"Ego": [(0.0, 9.0, 2.0)]})
