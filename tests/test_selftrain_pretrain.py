"""The two WP5 tasks: what they build, what they refuse, and where the soft targets come from.

The parts that decide are pure and are tested exhaustively. Two of them can be silently wrong in a way
that produces a plausible model: a soft-target weight that collapses to zero teaches the detector to
ignore whole batches, and a consensus that averages two confidences lets one confident model launder its
own uncertainty as agreement.
"""

from __future__ import annotations

import pytest

from services.training.tasks.base import TASKS, get_task
from services.training.tasks.pretrain import (
    PRETRAIN_STEP_FLOOR,
    STEPS_PER_HOUR,
    default_steps,
    estimate_cost_usd,
)
from services.training.tasks.selftrain import (
    SOFT_TARGET_FLOOR,
    soft_target_weight,
)


class TestBothTasksAreRealPlugins:
    @pytest.mark.parametrize("task_type", ["pretrain", "selftrain"])
    def test_the_task_is_registered_and_conforms(self, task_type):
        task = get_task(task_type)
        assert task.task_type == task_type
        for method in ("default_base_weights", "build_dataset", "train", "evaluate", "gate"):
            assert callable(getattr(task, method)), f"{task_type} is missing {method}"
        assert isinstance(task.default_base_weights(), str)

    def test_registering_them_did_not_displace_the_existing_tasks(self):
        for existing in ("detection", "segmentation", "classification", "lane", "pose", "detect3d"):
            assert existing in TASKS


class TestTheCostGuard:
    def test_the_estimate_names_the_rate_it_used(self):
        """An estimate whose inputs are invisible is one nobody can check against the invoice."""
        est = estimate_cost_usd(3200, hourly_usd=2.0)
        assert est == {"steps": 3200, "hours": 1.0, "hourly_usd": 2.0, "usd": 2.0,
                       "steps_per_hour": STEPS_PER_HOUR}

    def test_cost_is_linear_in_the_step_budget(self):
        a = estimate_cost_usd(3200, hourly_usd=2.0)["usd"]
        b = estimate_cost_usd(6400, hourly_usd=2.0)["usd"]
        assert b == pytest.approx(2 * a)

    def test_the_default_budget_fits_inside_the_cap(self):
        """A fixed 20,000 steps costs $11.81 at the ship rate against a $10 cap, so every unmodified
        dispatch would have been refused by the guard. The default is derived instead."""
        from core.config import get_settings

        cfg = get_settings().cloud
        est = estimate_cost_usd(default_steps())
        assert est["usd"] <= float(cfg.per_job_cap_usd)

    def test_a_tiny_cap_yields_a_floor_rather_than_zero_steps(self, monkeypatch):
        from core.config import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg.cloud, "per_job_cap_usd", 0.01)
        assert default_steps() == PRETRAIN_STEP_FLOOR

    def test_raising_the_cap_raises_the_default(self, monkeypatch):
        from core.config import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg.cloud, "per_job_cap_usd", 10.0)
        small = default_steps()
        monkeypatch.setattr(cfg.cloud, "per_job_cap_usd", 40.0)
        assert default_steps() > small


class TestSoftTargetWeighting:
    def test_confident_pseudo_labels_count_for_nearly_everything(self):
        assert soft_target_weight([0.9, 0.95, 0.85]) == pytest.approx(0.9)

    def test_marginal_ones_count_for_less(self):
        assert soft_target_weight([0.35, 0.4]) < soft_target_weight([0.9, 0.9])

    def test_instances_below_the_floor_do_not_drag_the_weight_down(self):
        """They are dropped from the label file too. Letting them into the weight would punish a batch
        for containing a guess the batch was never trained on."""
        assert soft_target_weight([0.9, 0.9, 0.01]) == pytest.approx(0.9)

    def test_an_empty_batch_scores_one_rather_than_zero(self):
        """A background image carries real information: restricting training to labelled objects is what
        taught 92.4% of this corpus's traffic as background and put recall at 0.411. Zeroing the loss on
        an unlabelled frame would teach the model to ignore exactly where it hallucinates."""
        assert soft_target_weight([]) == 1.0
        assert soft_target_weight([0.01, 0.02]) == 1.0

    def test_the_weight_is_never_outside_the_unit_interval(self):
        for case in ([0.3], [1.0, 1.0], [0.5, 0.9], [], [0.31, 0.99]):
            w = soft_target_weight(case)
            assert 0.0 <= w <= 1.0

    def test_the_floor_is_a_product_of_two_confidences_not_one(self):
        """0.30 is roughly 0.55 from each of two models, which is the point: a single model at 0.55 is
        not evidence, and two models at 0.55 on the same box is."""
        assert SOFT_TARGET_FLOOR == pytest.approx(0.30)
        assert 0.54 ** 2 < SOFT_TARGET_FLOOR <= 0.56 ** 2


class TestTheConsensusIsAProduct:
    """The soft target multiplies the two models' confidences rather than averaging them."""

    def test_the_product_is_always_the_harder_bar(self):
        """For any two confidences that differ, the product is below the mean. So a product cannot be
        carried across the floor by one confident model the way a mean can."""
        for a, b in ((0.95, 0.40), (0.9, 0.5), (0.99, 0.36), (0.6, 0.45)):
            assert a * b < (a + b) / 2

    def test_two_mediocre_models_agreeing_is_not_evidence(self):
        """Both at 0.5 gives a product of 0.25 and is rejected, where the mean would be 0.5 and pass.
        Two models that are each unsure do not become sure by agreeing."""
        assert 0.5 * 0.5 < SOFT_TARGET_FLOOR
        assert (0.5 + 0.5) / 2 > SOFT_TARGET_FLOOR

    def test_two_agreeing_models_clear_the_floor(self):
        assert 0.7 * 0.7 > SOFT_TARGET_FLOOR


class TestLineage:
    def test_a_self_trained_model_records_its_teacher_and_reads_as_distilled(self):
        from services.training.jobs import TrainJobSpec, _lineage_for

        spec = TrainJobSpec(purpose="selftrain", task_type="selftrain", base_weights="yolo11n.pt")
        out = _lineage_for(spec, {"teachers": ["champ-a", "champ-b"]})
        assert out["origin"] == "distilled"
        assert out["teacher_version"] == "champ-a"
        assert out["parent_version"] == "yolo11n.pt"
        assert out["arch"] == "yolo11n"

    def test_a_pretraining_run_reads_as_pretrained_with_no_teacher(self):
        from services.training.jobs import TrainJobSpec, _lineage_for

        out = _lineage_for(TrainJobSpec(purpose="p", task_type="pretrain",
                                        base_weights="vit_base_patch16_dinov3.lvd1689m"), {})
        assert out["origin"] == "pretrained" and out.get("teacher_version") is None

    def test_an_ordinary_run_reads_as_trained(self):
        from services.training.jobs import TrainJobSpec, _lineage_for

        out = _lineage_for(TrainJobSpec(purpose="p", task_type="detection",
                                        base_weights="yolo11n.pt"), {})
        assert out["origin"] == "trained" and out.get("teacher_version") is None

    def test_a_job_that_took_the_task_default_still_records_its_parent(self):
        """A null parent reads as "started from nothing" rather than "started from the shipped weights",
        and the first real self-training run recorded exactly that."""
        from services.training.jobs import TrainJobSpec, _lineage_for

        spec = TrainJobSpec(purpose="p", task_type="selftrain", base_weights=None)
        out = _lineage_for(spec, {"teachers": ["t"]}, "/models/yolo11n.pt")
        assert out["parent_version"] == "/models/yolo11n.pt"
        assert out["arch"] == "yolo11n"


class TestPretrainRefusesLocally:
    def test_it_refuses_with_the_reason_rather_than_silently_doing_nothing(self):
        """Days of an A100 on the one 16 GB card this host has would stop every other loop in the
        program. The refusal names the cloud path rather than leaving a stub."""
        task = get_task("pretrain")
        with pytest.raises(RuntimeError, match="does not run locally"):
            task.train("/tmp/manifest.json", "vit", {}, lambda _d: None)

    def test_it_never_auto_promotes(self):
        task = get_task("pretrain")
        out = task.gate({"map50": 0.99}, {"map50": 0.1}, {})
        assert out["promote"] is False and out["reasons"]

    def test_evaluate_reports_unmeasured_rather_than_a_made_up_metric(self):
        task = get_task("pretrain")
        out = task.evaluate("/tmp/w.pt", "/tmp/m.json", 640)
        assert out["measured"] is False and "reason" in out
        assert "map50" not in out


@pytest.mark.db
class TestTheManifestOnRows:
    async def test_it_excludes_synthetic_frames_and_deduplicates(self):
        """A backbone that learns the seams of the copy-paste generator would find those seams similar
        for the rest of its life, and one that sees the same dashcam frame a thousand times learns that
        frame rather than the domain."""
        import uuid as _uuid

        from sqlalchemy import delete

        from core.origin import REAL, SYNTHETIC
        from core.timebase import now_ns
        from db.models import Frame
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.training.tasks.pretrain import build_manifest

        onto = get_ontology()
        tag = _uuid.uuid4().hex[:6]
        real_sid, synth_sid = _uuid.uuid4(), _uuid.uuid4()
        group = _uuid.uuid4()
        async with get_sessionmaker()() as db:
            for sid, origin in ((real_sid, REAL), (synth_sid, SYNTHETIC)):
                db.add(DbSession(session_id=sid, vehicle_id=f"PRE-{tag}", start_ts_ns=0, end_ts_ns=1,
                                 sensors={}, ontology_version=onto.version, origin=origin))
            await db.flush()
            keep = _uuid.uuid4()
            base = now_ns()
            # One canonical frame of a duplicate group, one non-canonical sibling, one composite.
            db.add(Frame(frame_id=keep, session_id=real_sid, ts_ns=base, cam_id="front", width=64,
                         height=64, img_uri=f"frames/{real_sid}/a.jpg", origin=REAL, selected=True,
                         dup_group_id=group, is_dup_canonical=True))
            db.add(Frame(frame_id=_uuid.uuid4(), session_id=real_sid, ts_ns=base + 1, cam_id="front",
                         width=64, height=64, img_uri=f"frames/{real_sid}/b.jpg", origin=REAL,
                         selected=True, dup_group_id=group, is_dup_canonical=False))
            db.add(Frame(frame_id=_uuid.uuid4(), session_id=synth_sid, ts_ns=base + 2, cam_id="front",
                         width=64, height=64, img_uri=f"frames/{synth_sid}/c.jpg", origin=SYNTHETIC,
                         selected=True))
            await db.commit()

        try:
            man = await build_manifest(max_frames=100000)
            ids = {f["frame_id"] for f in man["frames"]}
            sessions = {f["session_id"] for f in man["frames"]}
            assert str(keep) in ids, "the canonical frame of the group is kept"
            assert str(synth_sid) not in sessions, "no composite reaches a pretraining manifest"
            group_members = [f for f in man["frames"] if f["session_id"] == str(real_sid)]
            assert len(group_members) == 1, "one frame per duplicate group"
        finally:
            async with get_sessionmaker()() as db:
                await db.execute(delete(DbSession).where(
                    DbSession.session_id.in_((real_sid, synth_sid))))
                await db.commit()

    async def test_the_cap_bounds_the_manifest(self):
        from services.training.tasks.pretrain import build_manifest

        man = await build_manifest(max_frames=25)
        assert man["n_frames"] <= 25
        assert man["capped_at"] == 25


@pytest.mark.db
class TestTheConsensusLadder:
    async def test_it_names_which_rung_answered(self):
        """A label from two agreeing models and a label from a fused multi-path consensus are not the
        same evidence, so the run records which one it got."""
        from services.training.tasks.selftrain import gather_pseudo_labels

        res = await gather_pseudo_labels(limit=500)
        assert set(res) >= {"source", "n", "manifest"}
        if res["n"]:
            assert res["source"] in ("oraclyx_consensus", "model_agreement")
            assert all(0.0 <= r["soft_target"] <= 1.0 for r in res["manifest"])
        else:
            assert res.get("reason"), "an empty result must say why rather than looking like zero labels"

    async def test_the_pair_is_chosen_by_shared_frames_not_by_size(self):
        """Two runs that scored different frames cannot agree on anything however large they are.
        Picking by prediction count alone chose a pair overlapping on 4 frames and yielded 9 labels
        where the best-overlapping pair yields 1,190 over the same corpus."""
        from services.training.tasks.selftrain import gather_pseudo_labels

        res = await gather_pseudo_labels(limit=5000)
        if res["n"]:
            assert res.get("shared_frames", 0) > 0
            assert len(set(res["teachers"])) == 2, "a model cannot be its own consensus partner"

    async def test_two_runs_of_one_model_are_not_a_consensus(self):
        """A model agrees with itself. Letting two runs of one model count would launder a single
        model's confidence as agreement, which is the whole thing the product is guarding against."""
        import inspect

        from services.training.tasks import selftrain

        src = inspect.getsource(selftrain.gather_pseudo_labels)
        assert "model_of[a] == model_of[b]" in src, "same-model pairs must be skipped"


class TestTheMetricBasisIsNamed:
    """The registry column is called `gold_metrics` and two different yardsticks write into it.

    A self-trained model registered 0.4975 from a two-image val split beside a champion's 0.4407 from a
    202-frame sealed gold set, under the same column name. The promotion gate is unaffected because it
    re-scores both sides on common gold, but a person reading the registry was comparing two things.
    """

    def test_a_gold_evaluation_says_which_gold_set(self):
        import inspect

        from services.govern import gold_eval

        src = inspect.getsource(gold_eval._score_yaml)
        assert 'metrics["basis"] = f"sealed_gold:{gold_id}"' in src

    def test_an_auto_registered_job_says_it_was_a_val_split(self):
        import inspect

        from services.training import jobs

        src = inspect.getsource(jobs.run_job)
        assert '"basis": f"job_val_split:' in src
