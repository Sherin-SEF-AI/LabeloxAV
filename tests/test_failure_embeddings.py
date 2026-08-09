"""The failures nobody could see.

`build_failure_clusters` described 32 of 32,576 failures on the champion, and its own note said why: a false
negative is a gold `Object` carrying a DINOv3 vector, a false positive is a `Prediction` carrying none. So
99.9% of what the model got wrong was invisible to the failure map and to everything mining from it.

The missing half is the more actionable one. A false negative says the model missed something that was there;
a false positive says what it invents. With the crops encoded, the champion's map gained clusters reading
"hallucinated motorcycle (46 of 73)", which is a sentence somebody can act on.

The tests worth having here are about honesty of coverage. Encoding is bounded, and a bounded sample
presented as the whole set is exactly the failure this codebase keeps finding.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.timebase import now_ns
from db.models import (
    EvalPatch,
    Frame,
    InferenceRun,
    ModelRegistry,
    Prediction,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.verdyx.failure_embeddings import MIN_CROP_PX, embed_false_positives

pytestmark = pytest.mark.db


async def _seed(db, *, n_fp: int, bbox=(1.0, 1.0, 60.0, 60.0), img_uri="s3://labeloxav/t.jpg") -> uuid.UUID:
    onto_v = "test"
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-FPEMB", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=onto_v)
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri=img_uri, width=200, height=200)
    db.add(frame)
    mv = f"m-fpemb-{uuid.uuid4().hex[:8]}"
    db.add(ModelRegistry(model_version=mv, weights_uri="s3://w.pt"))
    await db.flush()
    run = InferenceRun(run_id=uuid.uuid4(), model_version=mv, status="complete", params={})
    db.add(run)
    await db.flush()

    eid = uuid.uuid4()
    for _ in range(n_fp):
        p = Prediction(run_id=run.run_id, frame_id=frame.frame_id, class_id=1,
                       bbox=list(bbox), conf=0.7)
        db.add(p)
        await db.flush()
        db.add(EvalPatch(eval_id=eid, gold_id="g", model_version=mv, run_id=run.run_id,
                         prediction_id=p.prediction_id, frame_id=frame.frame_id,
                         outcome="fp", pred_class_id=1, conf=0.7))
    await db.commit()
    return eid


@pytest.fixture
def fake_encoder(monkeypatch):
    """A deterministic stand-in, so the test is about the selection and bookkeeping rather than the GPU."""
    import services.intelligence.embed.dinov3 as d

    monkeypatch.setattr(d, "encode_images", lambda crops: np.ones((len(crops), 8), dtype=np.float32))
    return d


@pytest.fixture
def fake_image(monkeypatch):
    import services.recall.backends as b

    monkeypatch.setattr(b, "load_image_bgr", lambda store, uri: np.full((200, 200, 3), 128, np.uint8))
    return b


async def test_false_positives_get_vectors_in_the_same_space(fake_encoder, fake_image):
    """Without this the failure map can describe only the misses, which was 32 of 32,576."""
    async with get_sessionmaker()() as db:
        eid = await _seed(db, n_fp=5)
        out = await embed_false_positives(db, eid)
    assert out["encoded"] == 5
    assert out["vectors"].shape[0] == 5


async def test_the_bound_is_reported_rather_than_applied_silently(fake_encoder, fake_image):
    """A sample described as the whole set is the failure this codebase keeps finding."""
    async with get_sessionmaker()() as db:
        eid = await _seed(db, n_fp=6)
        out = await embed_false_positives(db, eid, limit=2)
    assert out["encoded"] == 2
    assert out["available"] == 6
    assert out["truncated"] is True


async def test_nothing_is_marked_truncated_when_everything_was_encoded(fake_encoder, fake_image):
    async with get_sessionmaker()() as db:
        eid = await _seed(db, n_fp=3)
        out = await embed_false_positives(db, eid, limit=50)
    assert out["truncated"] is False and out["available"] == 3


async def test_a_crop_too_small_to_mean_anything_is_skipped(fake_encoder, fake_image):
    """A distant spurious box is where the vector would be noise dressed as signal."""
    tiny = (0.0, 0.0, float(MIN_CROP_PX - 2), float(MIN_CROP_PX - 2))
    async with get_sessionmaker()() as db:
        eid = await _seed(db, n_fp=4, bbox=tiny)
        out = await embed_false_positives(db, eid)
    assert out["encoded"] == 0 and out["skipped"] == 4


async def test_a_frame_whose_image_is_missing_does_not_end_the_run(monkeypatch, fake_encoder):
    """One unreadable frame took out a 1,000-frame relabel batch once already."""
    import services.recall.backends as b

    def _boom(store, uri):
        raise RuntimeError("NoSuchKey")

    monkeypatch.setattr(b, "load_image_bgr", _boom)
    async with get_sessionmaker()() as db:
        eid = await _seed(db, n_fp=3)
        out = await embed_false_positives(db, eid)
    assert out["encoded"] == 0 and out["skipped"] == 3


async def test_an_evaluation_with_no_false_positives_returns_empty_not_an_error(fake_encoder, fake_image):
    async with get_sessionmaker()() as db:
        out = await embed_false_positives(db, uuid.uuid4())
    assert out["encoded"] == 0 and out["available"] == 0
    assert list(out["ids"]) == []


async def test_the_class_the_model_claimed_travels_with_each_crop(fake_encoder, fake_image):
    """It is what turns a cluster into "hallucinated motorcycle" rather than "some spurious boxes"."""
    async with get_sessionmaker()() as db:
        eid = await _seed(db, n_fp=2)
        out = await embed_false_positives(db, eid)
    assert out["class_ids"] == [1, 1]
