"""Both eval harnesses must score the same population, or their disagreement means nothing.

The promotion gate refuses a model whose val-pass mAP50 and prediction-plane AP50 differ by more than
`harness_reconcile_epsilon`, on the sound principle that two harnesses disagreeing about one model on one
gold set is a measurement fault. Every model tripped it, which is the shape of a fault in the harness rather
than in any model.

The cause was that the two scored different gold populations. `services/training/gold.py:_materialize_aligned`
drops gold objects whose ontology class the model was never taught, because a detector cannot be held
responsible for a class it has no output for. `evaluate_gold_patches` scored the sealed set unfiltered, so
those same objects arrived as false negatives on one side and were absent on the other.

Measured on a real case before the fix, the DashLab detector against gold-5326ab441bab34e4: the val pass
scored 10 objects on 5 frames, the prediction plane 47 on 7, and the 37 extra were exactly its
out-of-vocabulary classes (sedan, rider, pedestrian, cattle, traffic_sign, cycle). Prediction-plane AP50 read
0.133 against a val-pass 0.537. With the populations aligned it reads 0.443, and the delta falls from 0.248
to 0.094.

These tests pin the population, not the score. The residual delta is an artifact of two different AP
implementations over ten instances and is a sample-size question, which is what rebuilding the gold set
addresses.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns
from db.models import Frame, GoldSet, InferenceRun, ModelRegistry, Object, Prediction
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.analytics.evaluation import evaluate_gold_patches
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db

BOX = [10.0, 10.0, 50.0, 50.0]


async def _seed(db, gold_classes: list[str], model: str,
                predicts: dict[int, str] | None = None) -> tuple[str, str]:
    """One gold object per named class, each on its own frame, plus whatever the model predicted.

    One class per frame matters: it is what lets a test assert that a frame left with no in-vocabulary gold
    drops out of the scored set entirely rather than lingering as a source of phantom false positives.

    `predicts` maps a frame index to the class the model emitted there, so a fixture can be realistic about
    what a limited model does: it never predicts a class it has no output for, and it does sometimes fire on
    a frame whose gold it could never have matched. Defaults to predicting each frame's own gold class, which
    is the trivially-correct model.
    """
    onto = get_ontology()
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-HARNESS", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=onto.version)
    db.add(sess)

    gold_id = f"gold-pop-{uuid.uuid4().hex[:12]}"
    if not await db.get(ModelRegistry, model):
        db.add(ModelRegistry(model_version=model, weights_uri="s3://w.pt"))
        await db.flush()
    run = InferenceRun(run_id=uuid.uuid4(), model_version=model, gold_id=gold_id, status="complete",
                       params={"imgsz": 640})
    db.add(run)

    object_ids: list[str] = []
    for i, name in enumerate(gold_classes):
        cid = onto.by_name(name).id
        frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                      img_uri="s3://labeloxav/test.jpg", width=100, height=100)
        db.add(frame)
        obj = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=cid, bbox=list(BOX),
                     conf=1.0, source="human", state="accepted")
        db.add(obj)
        await db.flush()
        object_ids.append(str(obj.object_id))

        pred_name = name if predicts is None else predicts.get(i)
        if pred_name is not None:
            db.add(Prediction(run_id=run.run_id, frame_id=frame.frame_id,
                              class_id=onto.by_name(pred_name).id, bbox=list(BOX), conf=0.9))

    db.add(GoldSet(gold_id=gold_id, name="pop", object_ids=object_ids, n_objects=len(object_ids),
                   n_frames=len(object_ids), ontology_version=onto.version))
    await db.commit()
    return gold_id, str(run.run_id)


async def test_the_vocabulary_filter_shrinks_the_scored_population_to_the_val_passs():
    """The defect itself. Three gold classes, a model that knows one: two objects were being scored against a
    model with no output for them, as false negatives the aligned val pass never saw."""
    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, ["pedestrian", "cattle", "truck"], "m-pop-a")

        unfiltered = await evaluate_gold_patches(db, gold_id, run_id=run_id)
        assert unfiltered["gold_scored"] == 3
        assert unfiltered["vocabulary_filtered"] is False
        assert unfiltered["frames"] == 3

        filtered = await evaluate_gold_patches(db, gold_id, run_id=run_id,
                                               model_vocabulary=frozenset({"truck"}))
        assert filtered["gold_scored"] == 1, "only the class the model can emit is scorable"
        assert filtered["gold_resolvable"] == 3, "what was available must still be reported"
        assert filtered["vocabulary_filtered"] is True


async def test_an_untaught_class_stops_counting_as_a_false_negative():
    """The mechanism by which the population gap moved the number. Unfiltered, the two classes the model
    cannot emit are misses; filtered, they are simply not this model's question."""
    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, ["pedestrian", "cattle", "truck"], "m-pop-b",
                                          predicts={2: "truck"})

        unfiltered = await evaluate_gold_patches(db, gold_id, run_id=run_id)
        filtered = await evaluate_gold_patches(db, gold_id, run_id=run_id,
                                               model_vocabulary=frozenset({"truck"}))
        assert unfiltered["fn"] > filtered["fn"]
        assert filtered["fn"] == 0, "the one class it knows was predicted exactly, so nothing is missed"


async def test_a_frame_with_no_in_vocabulary_gold_leaves_the_scored_set_entirely():
    """Filtering the gold side has to remove the frame too. A frame kept with its gold dropped would let the
    model's predictions there score as pure false positives that the val pass never counted."""
    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, ["pedestrian", "truck"], "m-pop-c",
                                          predicts={0: "truck", 1: "truck"})

        filtered = await evaluate_gold_patches(db, gold_id, run_id=run_id,
                                               model_vocabulary=frozenset({"truck"}))
        assert filtered["frames"] == 1
        assert filtered["fp"] == 0, "the pedestrian frame's prediction must not be fetched at all"


async def test_a_model_sharing_no_class_with_gold_is_refused_not_scored_zero():
    """A model that cannot be measured has not scored badly. Returning 0.0 would read as the former and would
    be handed to the promotion gate as a real number."""
    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, ["pedestrian", "cattle"], "m-pop-d")

        out = await evaluate_gold_patches(db, gold_id, run_id=run_id,
                                          model_vocabulary=frozenset({"truck"}))
        assert "error" in out
        assert out.get("ap50") is None
        assert out["gold_scored"] == 0
        assert out["gold_resolvable"] == 2


async def test_omitting_the_vocabulary_keeps_the_old_behaviour():
    """The parameter is optional and callers that do not know a model's class list must be unaffected."""
    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, ["pedestrian", "truck"], "m-pop-e")

        out = await evaluate_gold_patches(db, gold_id, run_id=run_id)
        assert out["gold_scored"] == out["gold_resolvable"] == 2
        assert out["vocabulary_filtered"] is False


async def test_the_scored_population_is_always_reported():
    """A number over 10 of 400 objects and a number over 400 of 400 are both returned as `ap50`. Without
    these fields nothing downstream can tell them apart, which is how a set that resolved 12% was scored and
    reported for months."""
    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, ["truck"], "m-pop-f")

        out = await evaluate_gold_patches(db, gold_id, run_id=run_id)
        for field in ("gold_declared", "gold_resolvable", "gold_scored", "vocabulary_filtered"):
            assert field in out, f"{field} must be reported on every evaluation"
