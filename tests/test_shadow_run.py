"""Shadow mode on rows: the sweep's idempotency key, adjudication from human labels, and the gate clause.

Three things here can only be checked against a database. The idempotency key, because the defect it
fixes is that two nights with the same models and code produce the same key and the second silently
reuses the first. Adjudication, because it reads a person's boxes and writes a verdict from them.
And the gate clause, because "unmeasured blocks nothing, a measured loss blocks" is the whole point.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.origin import REAL
from core.timebase import now_ns
from db.models import (
    Frame,
    InferenceRun,
    ModelRegistry,
    Object,
    Prediction,
    ShadowDisagreement,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db

BOX = [100.0, 100.0, 200.0, 200.0]
FAR = [500.0, 500.0, 600.0, 600.0]


async def _seed(db) -> dict:
    """One real frame, two complete inference runs over it, and a champion plus challenger registered."""
    onto = get_ontology()
    ped = onto.by_name("pedestrian").id
    car = onto.by_name("sedan").id
    tag = uuid.uuid4().hex[:6]
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id=f"SHDW-{tag}", start_ts_ns=0, end_ts_ns=1, city="BLR",
                     route=f"shadow-fixture-{tag}", sensors={}, ontology_version=onto.version, origin=REAL))
    await db.flush()
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=now_ns(), cam_id="front", width=1280, height=960,
                 img_uri=f"frames/{sid}/front/1.jpg", origin=REAL, selected=True))
    # Neither is marked champion: the table allows one champion per task and the corpus already has one.
    # The matcher takes run ids, not the registry's opinion of which model is serving.
    for mv in (f"shadow-champ-{tag}", f"shadow-chall-{tag}"):
        db.add(ModelRegistry(model_version=mv, task="detection", gold_metrics={}, is_champion=False,
                             weights_uri=f"models/{mv}.pt"))
    await db.flush()

    runs = {}
    for side, mv in (("champion", f"shadow-champ-{tag}"), ("challenger", f"shadow-chall-{tag}")):
        r = InferenceRun(model_version=mv, gold_id=None, params={}, code_sha="deadbeef",
                         status="complete", frame_count=1)
        db.add(r)
        await db.flush()
        runs[side] = r
    # Champion sees a pedestrian and a far sedan; challenger sees the pedestrian as a sedan and misses
    # the far one. One class flip and one challenger miss.
    db.add(Prediction(run_id=runs["champion"].run_id, frame_id=fid, class_id=ped, bbox=BOX, conf=0.9))
    db.add(Prediction(run_id=runs["champion"].run_id, frame_id=fid, class_id=car, bbox=FAR, conf=0.8))
    db.add(Prediction(run_id=runs["challenger"].run_id, frame_id=fid, class_id=car, bbox=BOX, conf=0.85))
    await db.commit()
    return {"session_id": sid, "frame_id": fid, "runs": runs, "ped": ped, "car": car, "tag": tag,
            "champion": f"shadow-champ-{tag}", "challenger": f"shadow-chall-{tag}"}


async def _cleanup(sid):
    from sqlalchemy import delete

    async with get_sessionmaker()() as db:
        await db.execute(delete(DbSession).where(DbSession.session_id == sid))
        await db.execute(delete(ModelRegistry).where(ModelRegistry.model_version.like("shadow-%")))
        await db.commit()


class TestTheMatcherOnRows:
    async def test_it_writes_one_row_per_disagreement_and_none_for_agreement(self):
        from services.verdyx.shadow_run import match_predictions

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            sweep = uuid.uuid4()
            from db.models import AgentRun

            db.add(AgentRun(run_id=sweep, kind="shadow_sweep", scope={}, status="running", policy={},
                            counts={}, changes={}, critic={}, created_by="test"))
            await db.commit()
            res = await match_predictions(db, sweep_run_id=sweep,
                                          champion_run=str(fx["runs"]["champion"].run_id),
                                          challenger_run=str(fx["runs"]["challenger"].run_id),
                                          threshold=0.5)
            assert res["written"] == 2, res
            assert res["by_kind"]["class_flip"] == 1
            assert res["by_kind"]["challenger_miss"] == 1
            rows = (await db.execute(select(ShadowDisagreement).where(
                ShadowDisagreement.sweep_run_id == sweep))).scalars().all()
            assert {r.state for r in rows} == {"pending"}
            assert all(r.verdict is None for r in rows), "nobody has ruled yet"
        await _cleanup(fx["session_id"])

    async def test_a_partial_run_is_not_a_comparison(self):
        from services.verdyx.shadow_run import match_predictions

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            fx["runs"]["challenger"].status = "running"
            await db.commit()
            res = await match_predictions(db, sweep_run_id=uuid.uuid4(),
                                          champion_run=str(fx["runs"]["champion"].run_id),
                                          challenger_run=str(fx["runs"]["challenger"].run_id))
            assert "error" in res and "partial" in res["error"]
        await _cleanup(fx["session_id"])


class TestTheIdempotencyKey:
    def test_scope_is_part_of_the_key(self):
        """Two sweeps over different frames must be two runs. Without the scope the key is identical
        (same model, gold_id None, same code sha, same params) and the second night would reuse the
        first night's run and score nothing at all."""
        from services.verdyx.inference_run import _run_params

        a = _run_params(960, 0.001, "cuda:0", {"shadow_sweep": "night-1"})
        b = _run_params(960, 0.001, "cuda:0", {"shadow_sweep": "night-2"})
        bare = _run_params(960, 0.001, "cuda:0")
        assert a != b
        assert "scope" not in bare, "a gold run's key must not change shape"
        assert a["scope"] == {"shadow_sweep": "night-1"}

    def test_the_same_scope_is_the_same_key_so_a_retry_reuses(self):
        from services.verdyx.inference_run import _run_params

        assert (_run_params(960, 0.001, "cuda:0", {"shadow_sweep": "n", "chunk": 3})
                == _run_params(960, 0.001, "cuda:0", {"shadow_sweep": "n", "chunk": 3}))

    def test_the_model_is_loaded_once_per_run_not_once_per_batch(self):
        """A 2,000-frame sweep runs 125 batches. Constructing the model inside the batch loop re-read the
        checkpoint every time, which is the whole cost of the sweep spent on nothing."""
        import inspect

        from services.verdyx import inference_run

        assert "YOLO(" not in inspect.getsource(inference_run._infer)
        assert "YOLO(" in inspect.getsource(inference_run._load_model)


class TestAdjudication:
    async def _disagreements(self, db, fx, sweep):
        from db.models import AgentRun
        from services.verdyx.shadow_run import match_predictions

        db.add(AgentRun(run_id=sweep, kind="shadow_sweep", scope={}, status="running", policy={},
                        counts={}, changes={}, critic={}, created_by="test"))
        await db.commit()
        await match_predictions(db, sweep_run_id=sweep,
                                champion_run=str(fx["runs"]["champion"].run_id),
                                challenger_run=str(fx["runs"]["challenger"].run_id), threshold=0.5)

    async def test_a_persons_boxes_decide_who_was_right(self):
        from db.models import LabelJob, LabelProject, LabelTask
        from services.verdyx.shadow_run import adjudicate_for_job

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            sweep = uuid.uuid4()
            await self._disagreements(db, fx, sweep)

            proj = LabelProject(name=f"shadow-{fx['tag']}")
            db.add(proj)
            await db.flush()
            task = LabelTask(project_id=proj.project_id, name="t",
                             predicate={"frame_ids": [str(fx["frame_id"])]})
            db.add(task)
            await db.flush()
            job = LabelJob(task_id=task.task_id, frame_ids=[str(fx["frame_id"])], stage="annotation",
                           state="in_progress")
            db.add(job)
            await db.flush()
            await db.execute(
                ShadowDisagreement.__table__.update()
                .where(ShadowDisagreement.sweep_run_id == sweep)
                .values(task_id=task.task_id, state="queued"))
            # The person drew a pedestrian on the flip box and nothing on the far one.
            db.add(Object(frame_id=fx["frame_id"], class_id=fx["ped"], bbox=BOX, conf=1.0,
                          state="accepted", source="human"))
            await db.commit()

            res = await adjudicate_for_job(db, job.job_id)
            assert res["adjudicated"] == 2, res
            rows = {r.kind: r for r in (await db.execute(select(ShadowDisagreement).where(
                ShadowDisagreement.sweep_run_id == sweep))).scalars().all()}
            # The champion called it a pedestrian and the challenger a sedan: the champion was right.
            assert rows["class_flip"].verdict == "champion_right"
            assert rows["class_flip"].verdict_object_id is not None
            # The champion alone claimed a far sedan and the person found nothing there.
            assert rows["challenger_miss"].verdict == "challenger_right"
            assert rows["challenger_miss"].verdict_object_id is None
            assert all(r.state == "adjudicated" and r.adjudicated_at for r in rows.values())
        await _cleanup(fx["session_id"])

    async def test_a_box_only_the_challenger_found_and_nobody_confirmed_goes_to_the_champion(self):
        """The mirror of the case above, and the one that caught the verdicts being written backwards.

        `champion_miss` is named for the model that has nothing there. If the person also found nothing,
        the challenger invented a box and the champion was right to have missed it. Writing that the
        other way round would have turned every challenger hallucination into evidence in its favour,
        and the gate would have read a challenger that invents objects as one that finds them.
        """
        from db.models import AgentRun, LabelJob, LabelProject, LabelTask
        from services.verdyx.shadow_run import adjudicate_for_job, match_predictions

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            # Give the challenger a box nowhere near anything the champion or a person has.
            lone = [800.0, 700.0, 900.0, 800.0]
            db.add(Prediction(run_id=fx["runs"]["challenger"].run_id, frame_id=fx["frame_id"],
                              class_id=fx["ped"], bbox=lone, conf=0.95))
            sweep = uuid.uuid4()
            db.add(AgentRun(run_id=sweep, kind="shadow_sweep", scope={}, status="running", policy={},
                            counts={}, changes={}, critic={}, created_by="test"))
            await db.commit()
            await match_predictions(db, sweep_run_id=sweep,
                                    champion_run=str(fx["runs"]["champion"].run_id),
                                    challenger_run=str(fx["runs"]["challenger"].run_id), threshold=0.5)

            proj = LabelProject(name=f"shadow-mirror-{fx['tag']}")
            db.add(proj)
            await db.flush()
            task = LabelTask(project_id=proj.project_id, name="t",
                             predicate={"frame_ids": [str(fx["frame_id"])]})
            db.add(task)
            await db.flush()
            job = LabelJob(task_id=task.task_id, frame_ids=[str(fx["frame_id"])], stage="annotation",
                           state="in_progress")
            db.add(job)
            await db.flush()
            await db.execute(
                ShadowDisagreement.__table__.update()
                .where(ShadowDisagreement.sweep_run_id == sweep)
                .values(task_id=task.task_id, state="queued"))
            await db.commit()

            await adjudicate_for_job(db, job.job_id)
            miss = (await db.execute(select(ShadowDisagreement).where(
                ShadowDisagreement.sweep_run_id == sweep,
                ShadowDisagreement.kind == "champion_miss"))).scalars().all()
            assert len(miss) == 1
            assert miss[0].verdict == "champion_right"
            assert miss[0].verdict_object_id is None
        await _cleanup(fx["session_id"])

    async def test_a_job_with_no_disagreements_says_so_rather_than_failing(self):
        from db.models import LabelJob, LabelProject, LabelTask
        from services.verdyx.shadow_run import adjudicate_for_job

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            proj = LabelProject(name=f"shadow-empty-{fx['tag']}")
            db.add(proj)
            await db.flush()
            task = LabelTask(project_id=proj.project_id, name="t",
                             predicate={"frame_ids": [str(fx["frame_id"])]})
            db.add(task)
            await db.flush()
            job = LabelJob(task_id=task.task_id, frame_ids=[str(fx["frame_id"])], stage="annotation",
                           state="in_progress")
            db.add(job)
            await db.commit()
            res = await adjudicate_for_job(db, job.job_id)
            assert res["adjudicated"] == 0 and "reason" in res
        await _cleanup(fx["session_id"])


class TestTheGateClause:
    """Shadow evidence must be able to refuse a challenger and must never be able to promote one."""

    def _metrics(self, shadow=None):
        m = {"map50": 0.5, "safe_miou": 0.8, "per_class": {}, "recapture": {"ok": True, "checked": True}}
        if shadow is not None:
            m["shadow"] = shadow
        return m

    def test_unmeasured_shadow_blocks_nothing(self):
        from services.govern.champion import _shadow

        assert _shadow(self._metrics())["ok"] is True
        assert _shadow(self._metrics({"measured": False, "reason": "nobody looked"}))["ok"] is True

    def test_too_few_pairs_blocks_nothing(self):
        from services.govern.champion import MIN_SHADOW_PAIRS, _shadow

        s = {"measured": True, "discordant": MIN_SHADOW_PAIRS - 1, "challenger_right": 0,
             "share": 0.0, "lo": 0.0, "hi": 0.2}
        out = _shadow(self._metrics(s))
        assert out["ok"] is True and out["measured"] is False

    def test_a_measured_loss_blocks(self):
        from services.govern.champion import MIN_SHADOW_PAIRS, _shadow

        s = {"measured": True, "discordant": MIN_SHADOW_PAIRS + 20, "challenger_right": 5,
             "share": 0.1, "lo": 0.04, "hi": 0.24}
        out = _shadow(self._metrics(s))
        assert out["ok"] is False and out["reasons"]

    def test_a_measured_win_does_not_promote_on_its_own(self):
        """The clause is one-directional. A challenger that wins on the disagreements has said nothing
        about the frames the two models agree on, which is nearly all of them."""
        from core.config import get_settings
        from services.autolabel.ontology import get_ontology
        from services.govern.champion import MIN_SHADOW_PAIRS, _shadow, champion_gate

        s = {"measured": True, "discordant": MIN_SHADOW_PAIRS + 20, "challenger_right": 48,
             "share": 0.96, "lo": 0.86, "hi": 0.99}
        assert _shadow(self._metrics(s))["ok"] is True
        cfg = get_settings().phase4.govern
        # A challenger that loses on mAP is still refused, however well it did on the disagreements.
        weak = {**self._metrics(s), "map50": 0.10}
        champ = {"map50": 0.60, "safe_miou": 0.8, "per_class": {}}
        out = champion_gate(weak, champ, get_ontology(), cfg)
        assert out["promote"] is False


class TestTheHighWaterMark:
    """Only a committed sweep may move the window forward.

    The mark is what stops a sweep re-scoring frames it has already compared, so it is also what can
    permanently skip frames. A failed, refused or reverted sweep compared nothing on its frames, and
    letting its mark stand would step the window past them with nothing ever saying so. This was found
    by reverting a real sweep and watching the next one start after the frames the reverted one had
    named rather than at them.
    """

    async def test_a_reverted_sweep_does_not_move_the_window(self):
        from datetime import UTC, datetime, timedelta

        from db.models import AgentRun
        from services.govern.shadow_agent import KIND, _high_water_mark

        old = datetime(2026, 1, 1, tzinfo=UTC)
        new = datetime(2026, 6, 1, tzinfo=UTC)
        ids = []
        async with get_sessionmaker()() as db:
            for status, mark, age in (("committed", old, 2), ("reverted", new, 1)):
                rid = uuid.uuid4()
                ids.append(rid)
                db.add(AgentRun(run_id=rid, kind=KIND, scope={}, status=status, policy={},
                                counts={"high_water_mark": mark.isoformat()}, changes={}, critic={},
                                created_by="test",
                                created_at=datetime.now(UTC) - timedelta(hours=age)))
            await db.commit()
            got = await _high_water_mark(db)
            assert got == old, "the reverted sweep's mark must not count"

            from sqlalchemy import delete

            await db.execute(delete(AgentRun).where(AgentRun.run_id.in_(ids)))
            await db.commit()

    async def test_no_committed_sweep_means_no_mark_rather_than_now(self):
        from services.govern.shadow_agent import _high_water_mark

        async with get_sessionmaker()() as db:
            from sqlalchemy import select

            from db.models import AgentRun
            from services.govern.shadow_agent import KIND

            existing = (await db.execute(select(AgentRun.run_id).where(
                AgentRun.kind == KIND, AgentRun.status == "committed").limit(1))).first()
            if existing is not None:
                pytest.skip("this database already holds a committed sweep")
            assert await _high_water_mark(db) is None
