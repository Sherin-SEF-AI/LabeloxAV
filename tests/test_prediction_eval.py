"""Measurement-integrity acceptance tests (Section 1).

These prove the destructive-provenance defect is gone. A detection a human confirmed now scores as a TRUE
POSITIVE, because predictions live in the immutable prediction plane that review never mutates. On the pre-fix
tree the confirmed detection (source="human") was excluded from the scored population and read as a false
negative, which is what produced the 0.034 / 0.018 artifact.
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest
from sqlalchemy import select

from core.timebase import now_ns
from db.models import Frame, GoldSet, InferenceRun, ModelRegistry, Object, Prediction, Review
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.analytics.evaluation import evaluate_gold_patches, gold_provenance_report
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db

GOLD_BOX = [10.0, 10.0, 50.0, 50.0]


async def _seed(db, *, pred_boxes, pred_conf=0.9, model="m-eval-test",
                reconstructed=False, review_source: str | None = None) -> dict:
    """Seed a frame with one human-confirmed gold object (a pedestrian), a sealed gold set, a registered model,
    and an inference run whose predictions are `pred_boxes` on that frame. Returns the ids."""
    onto = get_ontology()
    cid = onto.by_name("pedestrian").id
    ver = onto.version

    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=ver)
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/test.jpg", width=100, height=100)
    db.add(frame)
    gobj = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=cid, bbox=list(GOLD_BOX),
                  conf=1.0, source="human", state="accepted")
    db.add(gobj)
    await db.flush()

    if review_source is not None:
        # A review row whose `before` records the pre-review machine provenance (as the fixed sites now do).
        db.add(Review(object_id=gobj.object_id, reviewer="tester", action="accept",
                      before={"class_id": cid, "bbox": list(GOLD_BOX), "source": review_source, "conf": 0.8},
                      after={"class_id": cid}, ts_ns=now_ns()))

    gold_id = f"gold-test-{uuid.uuid4().hex[:12]}"
    db.add(GoldSet(gold_id=gold_id, name="t", object_ids=[str(gobj.object_id)], n_objects=1, n_frames=1,
                   ontology_version=ver))
    if not await db.get(ModelRegistry, model):
        db.add(ModelRegistry(model_version=model, weights_uri="s3://w.pt"))
        await db.flush()   # the run's FK needs the model row to exist first
    run = InferenceRun(run_id=uuid.uuid4(), model_version=model, gold_id=gold_id, status="complete",
                       params={"reconstructed": True} if reconstructed else {"imgsz": 640})
    db.add(run)
    await db.flush()
    for box in pred_boxes:
        db.add(Prediction(run_id=run.run_id, frame_id=frame.frame_id, class_id=cid, bbox=list(box),
                          conf=pred_conf))
    await db.commit()
    return {"gold_id": gold_id, "run_id": str(run.run_id), "frame_id": frame.frame_id,
            "gold_object_id": str(gobj.object_id), "model": model, "class_id": cid}


async def test_confirmed_detection_scores_as_true_positive():
    async with get_sessionmaker()() as db:
        s = await _seed(db, pred_boxes=[GOLD_BOX])   # the model detected exactly the gold box
        res = await evaluate_gold_patches(db, s["gold_id"], run_id=s["run_id"])
    assert "error" not in res
    assert res["tp"] == 1 and res["fn"] == 0 and res["fp"] == 0
    assert res["per_class_recall"]["pedestrian"] == 1.0
    assert res["ap50"] == 1.0                       # a perfect same-class IoU=1 match
    assert res["model_version"] == s["model"]


async def test_eval_refuses_a_run_that_does_not_exist():
    async with get_sessionmaker()() as db:
        s = await _seed(db, pred_boxes=[GOLD_BOX])
        res = await evaluate_gold_patches(db, s["gold_id"], run_id=str(uuid.uuid4()))
    assert "error" in res and "tp" not in res       # refused, not silently scored


async def test_predictions_do_not_leak_across_runs():
    async with get_sessionmaker()() as db:
        s = await _seed(db, pred_boxes=[GOLD_BOX])   # run A predicts the gold box (would be a TP)
        # A second, empty run B on the same gold set: it must score as a miss, never inherit run A's prediction.
        run_b = InferenceRun(run_id=uuid.uuid4(), model_version=s["model"], gold_id=s["gold_id"],
                             status="complete", params={"imgsz": 640, "variant": "b"})
        db.add(run_b)
        await db.commit()
        res_b = await evaluate_gold_patches(db, s["gold_id"], run_id=str(run_b.run_id))
        res_a = await evaluate_gold_patches(db, s["gold_id"], run_id=s["run_id"])
    assert res_b["tp"] == 0 and res_b["fn"] == 1     # run B saw no predictions -> the gold object is a miss
    assert res_a["tp"] == 1                           # run A still scores its own prediction correctly


async def test_reconstructed_run_yields_no_ap():
    async with get_sessionmaker()() as db:
        s = await _seed(db, pred_boxes=[GOLD_BOX], reconstructed=True)
        res = await evaluate_gold_patches(db, s["gold_id"], run_id=s["run_id"])
    assert res["ap50"] is None and res["ap50_95"] is None
    assert "caveat" in res


async def test_gold_provenance_report_splits_confirmed_from_scratch():
    async with get_sessionmaker()() as db:
        s = await _seed(db, pred_boxes=[GOLD_BOX], review_source="fused")  # a confirmed machine detection
        rep = await gold_provenance_report(db, s["gold_id"])
    assert rep["n_objects"] == 1
    assert rep["confirmed_machine"] == 1 and rep["drawn_from_scratch"] == 0


async def test_backfill_reconstructs_predictions_with_null_conf():
    from scripts.backfill_prediction_from_review import backfill

    onto = get_ontology()
    cid = onto.by_name("pedestrian").id
    tag = f"utest{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        sess = DbSession(session_id=uuid.uuid4(), vehicle_id="T", start_ts_ns=0, end_ts_ns=1,
                         ontology_version=onto.version)
        db.add(sess)
        frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="c",
                      img_uri="s3://x", width=10, height=10)
        db.add(frame)
        obj = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=cid, bbox=list(GOLD_BOX),
                     conf=1.0, source="human", state="accepted")
        db.add(obj)
        await db.flush()
        db.add(Review(object_id=obj.object_id, reviewer="t", action="accept", ts_ns=now_ns(),
                      before={"class_id": cid, "bbox": list(GOLD_BOX), "source": "fused", "conf": 0.7},
                      after={"class_id": cid}))
        await db.commit()

    res = await backfill(date_tag=tag, force=True)
    assert res["reconstructed"] >= 1
    async with get_sessionmaker()() as db:
        run = (await db.execute(select(InferenceRun).where(
            InferenceRun.model_version == f"reconstructed-pre-{tag}"))).scalars().first()
        assert run is not None and run.params.get("reconstructed") is True
        preds = (await db.execute(select(Prediction).where(Prediction.run_id == run.run_id))).scalars().all()
        assert preds and any(p.conf is None for p in preds)   # reconstructed predictions have no score
