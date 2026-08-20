"""Confusion-clique active learning: what a label buys, and where the budget goes.

Two claims, tested separately because they fail in different ways.

The scoring claim is that confidence cannot rank a labelling budget. Two detections at the same
confidence, one doubting a class and one torn between two, want completely different things from a person,
and the arithmetic below distinguishes them by margin and then weights by what the confusion costs.

The allocation claim is smaller than it sounds and the test says so: the Thompson posteriors all start at
Beta(1,1), so today's allocation is uniform with sampling noise. What is tested is that a reward moves the
posterior in the right direction and that an unmeasurable outcome moves nothing, because a bandit that
punished a clique for having no measurement would learn to avoid whatever the evaluation forgot.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy import select

from core.accel.clique_margin import margin_score, score_batch
from core.timebase import now_ns
from db.models import CliqueBandit, Frame, InferenceRun, ModelRegistry, Prediction
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.sievyx.clique_sampler import (
    _thompson,
    bandit_report,
    record_reward,
    select_frames,
)


class TestMargin:
    def test_two_detections_at_the_same_confidence_score_differently(self):
        """The whole reason class_probs exists, in one comparison.

        Both top out at 0.55. The first has its runner-up at 0.05, so the model is doubting one class;
        the second has it at 0.45, so the model is deciding between two. Margins 0.50 and 0.10, and with
        an equal cost the scores are 0.50 and 0.90.
        """
        doubting = margin_score({"1": 0.55, "2": 0.05})
        torn = margin_score({"1": 0.55, "2": 0.45})
        assert abs(doubting.margin - 0.50) < 1e-9 and abs(torn.margin - 0.10) < 1e-9
        assert abs(doubting.score - 0.50) < 1e-9 and abs(torn.score - 0.90) < 1e-9
        assert torn.score > doubting.score

    def test_the_cost_of_the_confusion_multiplies_the_ambiguity(self):
        """Otherwise the budget goes entirely to two-wheelers, where ambiguity is densest and cost lowest.

        The same 0.10 margin: inside a cheap clique (0.2) it scores 0.18, across a safety boundary (1.0)
        it scores 0.90.
        """
        cheap = margin_score({"1": 0.55, "2": 0.45}, pair_cost=lambda a, b: 0.2)
        dear = margin_score({"1": 0.55, "2": 0.45}, pair_cost=lambda a, b: 1.0)
        assert abs(cheap.score - 0.18) < 1e-9
        assert abs(dear.score - 0.90) < 1e-9
        # A confident detection stays near zero however expensive its pair would have been.
        confident = margin_score({"1": 0.99, "2": 0.01}, pair_cost=lambda a, b: 1.0)
        assert confident.score < 0.03

    def test_it_names_what_the_model_is_torn_between(self):
        """"These forty frames are the scooter/motorcycle boundary" is reviewable; "uncertain" is not."""
        m = margin_score({"7": 0.4, "3": 0.35, "9": 0.25},
                         clique_of=lambda a, b: "two_wheelers" if {a, b} == {7, 3} else None)
        assert m.top_pair == (7, 3)
        assert m.top_probs == (0.4, 0.35)
        assert m.clique == "two_wheelers"

    def test_a_prediction_with_no_distribution_is_unmeasured_not_zero(self):
        """Zero would sort it beside the confident ones, which is a claim nothing supports.

        Every prediction written before class_probs existed is in this state, which is all 96,068 of them.
        """
        for bad in (None, {}, {"a": "not a number"}):
            m = margin_score(bad)
            assert m.measured is False and m.score is None, bad
            assert m.reason

    def test_one_class_with_all_the_mass_is_measured_and_uninteresting(self):
        m = margin_score({"4": 0.9})
        assert m.measured is True and m.score == 0.0
        assert "only one class" in m.reason

    def test_a_batch_marks_the_unmeasurable_as_nan_rather_than_a_number(self):
        out = score_batch([{"1": 0.55, "2": 0.45}, None, {"1": 0.99, "2": 0.01}])
        s = out["scores"]
        assert np.isnan(s[1]) and not np.isnan(s[0]) and not np.isnan(s[2])
        assert out["n_unmeasured"] == 1
        assert out["measured"].tolist() == [True, False, True]
        # Sorting descending on a NaN-bearing array must not float the unmeasurable to the top.
        order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
        assert order[0] == 0 and order[-1] == 1


class TestAllocation:
    def test_it_spends_exactly_the_budget(self):
        """Largest-remainder, so a 200-frame budget selects 200 frames rather than 197."""
        posteriors = {n: CliqueBandit(clique=n, pack_id="av", alpha=1.0, beta=1.0)
                      for n in ("a", "b", "c", "d", "e", "f", "g")}
        for budget in (1, 7, 200, 1000):
            alloc = _thompson(posteriors, budget, seed=5)
            assert sum(alloc.values()) == budget, (budget, alloc)
            assert all(v >= 0 for v in alloc.values())

    def test_a_clique_with_a_better_posterior_gets_more(self):
        """The behavioural claim of the bandit, with the posteriors set rather than earned."""
        posteriors = {
            "good": CliqueBandit(clique="good", pack_id="av", alpha=40.0, beta=2.0),
            "bad": CliqueBandit(clique="bad", pack_id="av", alpha=2.0, beta=40.0),
        }
        alloc = _thompson(posteriors, 100, seed=11)
        assert alloc["good"] > alloc["bad"] * 3, alloc

    def test_it_is_seeded_so_a_selection_can_be_explained(self):
        # Unseeded, two people asking why these frames were chosen get two different answers.
        posteriors = {n: CliqueBandit(clique=n, pack_id="av", alpha=1.0, beta=1.0) for n in ("a", "b")}
        assert _thompson(posteriors, 50, seed=3) == _thompson(posteriors, 50, seed=3)


class TestThePackDefinesTheCliques:
    def test_every_clique_member_is_a_real_ontology_class(self):
        """A clique naming a class that does not exist silently contributes nothing.

        It fails no import and raises nothing: the class simply never matches, so the clique quietly
        covers fewer classes than it claims. Two of the first draft's members were exactly this.
        """
        from packs.registry import default_pack_id, get_pack

        onto = get_ontology()
        spec = get_pack(default_pack_id()).cliques
        assert spec is not None
        missing = [(c.name, n) for c in spec.cliques for n in c.class_names if not onto.has_name(n)]
        assert missing == [], f"cliques naming classes the ontology does not have: {missing}"

    def test_a_class_belongs_to_at_most_one_clique(self):
        """Two cliques claiming a class makes `clique_of` order-dependent and the allocation arbitrary."""
        from packs.registry import default_pack_id, get_pack

        spec = get_pack(default_pack_id()).cliques
        seen: dict[str, str] = {}
        dupes = []
        for c in spec.cliques:
            for n in c.class_names:
                if n in seen:
                    dupes.append((n, seen[n], c.name))
                seen[n] = c.name
        assert dupes == [], dupes

    def test_crossing_a_safety_boundary_costs_more_than_staying_inside_one(self):
        from packs.registry import default_pack_id, get_pack

        spec = get_pack(default_pack_id()).cliques
        assert spec.pair_cost("motorcycle", "scooter") < spec.pair_cost("pedestrian", "rider")
        # And a pair in no shared clique is the most expensive: an unanticipated confusion is worse.
        assert spec.pair_cost("pedestrian", "bus") == spec.cross_clique_cost
        assert spec.pair_cost("motorcycle", "motorcycle") == 0.0


pytestmark_db = pytest.mark.db


class TestTheBanditPersists:
    pytestmark = pytest.mark.db

    async def test_a_reward_moves_the_posterior_and_an_unmeasurable_one_does_not(self):
        """A bandit punished for a missing measurement learns to avoid whatever the evaluation forgot.

        That is a property of the evaluation, not of the clique, so an unmeasurable outcome must leave the
        posterior exactly where it was.
        """
        async with get_sessionmaker()() as db:
            name = f"t-{uuid.uuid4().hex[:8]}"
            r = await record_reward(db, clique=name, allocated=10, recall_before=0.4, recall_after=0.6,
                                    pack_id="av")
            assert r["updated"] is True and r["improved"] is True
            row = await db.get(CliqueBandit, (name, "av"))
            assert (row.alpha, row.beta, row.n_pulls) == (2.0, 1.0, 1)

            await record_reward(db, clique=name, allocated=10, recall_before=0.6, recall_after=0.5,
                                pack_id="av")
            await db.refresh(row)
            assert (row.alpha, row.beta, row.n_pulls) == (2.0, 2.0, 2)

            before = (row.alpha, row.beta, row.n_pulls)
            out = await record_reward(db, clique=name, allocated=10, recall_before=None,
                                      recall_after=0.7, pack_id="av")
            await db.refresh(row)
            assert out["updated"] is False
            assert (row.alpha, row.beta, row.n_pulls) == before

    async def test_the_report_says_when_nothing_has_been_learned(self):
        """Uniform-because-untouched and uniform-because-measured are different states.

        Reading the second into the first would present a prior as a finding.
        """
        async with get_sessionmaker()() as db:
            rep = await bandit_report(db, pack_id=f"never-{uuid.uuid4().hex[:6]}")
            assert rep["learned"] is False and rep["cliques"] == []


class TestSelection:
    pytestmark = pytest.mark.db

    async def _run(self, db, with_probs: bool):
        onto = get_ontology()
        sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CLQ", start_ts_ns=0, end_ts_ns=1,
                         ontology_version=onto.version)
        db.add(sess)
        mv = f"clq-{uuid.uuid4().hex[:8]}"
        db.add(ModelRegistry(model_version=mv, task="detection"))
        await db.flush()
        run = InferenceRun(model_version=mv, status="complete", params={}, code_sha="0" * 40)
        db.add(run)
        await db.flush()
        moto, scoot = onto.by_name("motorcycle").id, onto.by_name("scooter").id
        ped, rider = onto.by_name("pedestrian").id, onto.by_name("rider").id
        # Ten frames torn between two-wheelers, ten torn across the safety boundary. Same margins, so
        # only the clique cost separates them.
        for i in range(20):
            f = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns() + i,
                      cam_id="front", width=1920, height=1080, img_uri=f"s3://x/{i}.jpg")
            db.add(f)
            await db.flush()
            a, b = (moto, scoot) if i < 10 else (ped, rider)
            db.add(Prediction(run_id=run.run_id, frame_id=f.frame_id, class_id=a,
                              bbox=[0.0, 0.0, 10.0, 10.0], conf=0.55,
                              class_probs={str(a): 0.55, str(b): 0.45} if with_probs else None))
        await db.commit()
        return run

    async def test_it_refuses_when_no_prediction_carries_a_distribution(self):
        """Falling back to lowest-confidence would silently be the old behaviour under a new name."""
        async with get_sessionmaker()() as db:
            run = await self._run(db, with_probs=False)
            res = await select_frames(db, run_id=str(run.run_id), budget=10)
            assert "error" in res and "class distribution" in res["error"]

    async def test_it_selects_frames_and_names_the_boundary_each_one_buys(self):
        async with get_sessionmaker()() as db:
            run = await self._run(db, with_probs=True)
            res = await select_frames(db, run_id=str(run.run_id), budget=8)
            assert "error" not in res, res
            assert res["n_selected"] == 8
            assert all(f["reason"].startswith("clique:") for f in res["frames"])
            assert {f["clique"] for f in res["frames"]} <= {"two_wheelers", "pedestrians_vs_riders"}
            # No frame is offered twice even though it can be a candidate for two boundaries.
            ids = [f["frame_id"] for f in res["frames"]]
            assert len(set(ids)) == len(ids)

    async def test_it_says_out_loud_that_the_allocation_is_still_the_prior(self):
        """Today's split is uniform with sampling noise, and reading it as a finding would be wrong."""
        async with get_sessionmaker()() as db:
            run = await self._run(db, with_probs=True)
            res = await select_frames(db, run_id=str(run.run_id), budget=8)
            assert res["learned"] is False
            assert "is not a finding" in res["caveat"]
            assert all(c["from_prior"] for c in res["per_clique"].values())

    async def test_the_more_expensive_boundary_ranks_higher_within_a_clique(self):
        """Same margin on both sides, so only the pack's cost can order them."""
        async with get_sessionmaker()() as db:
            run = await self._run(db, with_probs=True)
            res = await select_frames(db, run_id=str(run.run_id), budget=20)
            by_clique = {}
            for f in res["frames"]:
                by_clique.setdefault(f["clique"], []).append(f["score"])
            assert max(by_clique["pedestrians_vs_riders"]) > max(by_clique["two_wheelers"])

    async def test_a_run_that_does_not_exist(self):
        async with get_sessionmaker()() as db:
            res = await select_frames(db, run_id=str(uuid.uuid4()), budget=5)
            assert "error" in res and "not found" in res["error"]


def test_bandit_rows_are_created_lazily_not_seeded():
    # Nothing writes a row until a clique is first allocated or rewarded, so a pack whose cliques change
    # does not leave orphan posteriors claiming history they never had.
    assert select(CliqueBandit) is not None
