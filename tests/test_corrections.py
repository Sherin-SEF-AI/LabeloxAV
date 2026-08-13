"""The correct-once-apply-everywhere tool searched a table nothing writes to.

Correcting an object opened a dialog offering to apply the same fix to visually similar objects. It reported
zero every time, against a corpus of 2,361 objects in the class being corrected, 2,354 of which carry a
DINOv3 vector whose same-class neighbours score 0.99 against a corpus mean of 0.58.

The cause was the table. `services/intelligence/corrections.py` read the legacy CLIP `embedding` table while
the pipeline moved to pgvector `object_embedding`; every other find-similar surface was migrated and this one
was missed, including the coverage endpoint of the same feature, which reads the new table and reported
healthy coverage while the search read 39 rows. Worse, the old source-vector path wrote a row into the dead
table on every correction, and those objects then carried the NEW class, so the old-class filter could never
match them. It could not have worked at any threshold.
"""

from __future__ import annotations

import secrets
import uuid

import numpy as np
import pytest

from db.models import Frame, Object, ObjectEmbedding
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.intelligence.corrections import NO_NEIGHBOURS, NOT_EMBEDDED, correction_candidates

pytestmark = pytest.mark.db

DINO_DIM = 768


def _vec(seed: int, dim: int = DINO_DIM) -> list[float]:
    """A unit vector. Cosine distance is only meaningful over normalized vectors, which is what the
    embedding service writes."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _near(base: list[float], noise: float, seed: int) -> list[float]:
    """A vector close to `base`, for a candidate that should be found."""
    rng = np.random.default_rng(seed)
    v = np.asarray(base, dtype=np.float32) + rng.normal(scale=noise, size=len(base)).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


async def _scene(db, *, classes: list[int], embed: bool = True, sources: list[str] | None = None):
    """One frame carrying a query object plus one object per entry in `classes`.

    The base vector is fresh per scene. A fixed seed made every run's query object share one identical
    vector, so a later test measuring "nothing is similar enough" found the previous run's copies at
    similarity 1.0. The suite shares a database; a fixture that is deterministic across runs is a fixture
    that collides with itself.
    """
    seed = secrets.randbelow(2**31)
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="CORR-01", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()

    base = _vec(seed)
    qid = uuid.uuid4()
    db.add(Object(object_id=qid, frame_id=fid, class_id=classes[0] if classes else 1,
                  bbox=[10, 10, 200, 200], conf=0.5, source="auto_accept", state="review"))
    await db.flush()
    if embed:
        db.add(ObjectEmbedding(object_id=qid, dino_vec=base, model_versions={}))

    made = []
    for i, cid in enumerate(classes):
        oid = uuid.uuid4()
        src = (sources[i] if sources and i < len(sources) else "auto_accept")
        db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[20, 20, 210, 210],
                      conf=0.6, source=src, state="review"))
        await db.flush()
        db.add(ObjectEmbedding(object_id=oid, dino_vec=_near(base, 0.02, seed + 1 + i), model_versions={}))
        made.append(oid)
    await db.commit()
    return str(qid), [str(o) for o in made]


class TestTheReportedCase:
    async def test_similar_objects_are_found_at_all(self):
        """The whole bug in one assertion: this returned zero for every corpus, every class, every
        threshold, because it queried a table with nothing in it."""
        async with get_sessionmaker()() as db:
            qid, made = await _scene(db, classes=[1, 1, 1])
            res = await correction_candidates(db, qid, kind="class", old_class_id=1,
                                              new_value="hoarding", threshold=0.5)
        assert res["count"] >= 3
        found = {c["object_id"] for c in res["candidates"]}
        assert set(made) <= found

    async def test_the_query_object_is_not_offered_back_to_itself(self):
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[1, 1])
            res = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.5)
        assert qid not in {c["object_id"] for c in res["candidates"]}


class TestCandidatesSpanClasses:
    async def test_one_mistake_spread_over_several_classes_is_found_in_one_pass(self):
        """The relabel agent put 1,047 objects into bmtc_bus_shelter from `bus`, 708 from `traffic_sign` and
        522 from `hoarding`. Scoping to the class the operator happened to start from surfaces one of three.
        """
        async with get_sessionmaker()() as db:
            qid, made = await _scene(db, classes=[16, 35, 122])   # three different classes
            res = await correction_candidates(db, qid, kind="class", old_class_id=16, threshold=0.5)
        assert {c["object_id"] for c in res["candidates"]} >= set(made)
        assert len({c["class_name"] for c in res["candidates"]}) >= 2

    async def test_same_class_still_narrows_when_asked(self):
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[16, 35, 122])
            res = await correction_candidates(db, qid, kind="class", old_class_id=16,
                                              threshold=0.5, same_class=True)
        assert all(c["class_name"] == "bus" for c in res["candidates"])

    async def test_every_candidate_names_its_current_class(self):
        # With a mixed candidate set the label is load-bearing: a person agreeing to a batch has to see what
        # each thing currently is.
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[16, 35])
            res = await correction_candidates(db, qid, kind="class", old_class_id=16, threshold=0.5)
        assert all(c.get("class_name") for c in res["candidates"])


class TestItSaysWhyWhenItFindsNothing:
    async def test_an_unembedded_object_is_named_as_such(self):
        """Reported as "no similar objects above the threshold" before, which is a claim about similarity
        when the truth is that nothing was compared."""
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[1], embed=False)
            res = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.5)
        assert res["count"] == 0
        assert res["reason"] == NOT_EMBEDDED

    async def test_a_genuine_miss_is_distinguished_from_a_missing_embedding(self):
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[1])
            res = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.999999)
        assert res["count"] == 0
        assert res["reason"] == NO_NEIGHBOURS

    async def test_a_successful_search_carries_no_reason(self):
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[1, 1])
            res = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.5)
        assert res["reason"] is None


class TestGuards:
    async def test_a_human_labelled_object_is_flagged_rather_than_hidden(self):
        """A batch must not quietly overwrite somebody's decision, and hiding those objects would leave the
        operator wondering why the count moved."""
        async with get_sessionmaker()() as db:
            qid, made = await _scene(db, classes=[1, 1], sources=["human", "auto_accept"])
            res = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.5)
        by_id = {c["object_id"]: c for c in res["candidates"]}
        assert by_id[made[0]]["human"] is True
        assert by_id[made[1]]["human"] is False

    async def test_the_threshold_still_excludes(self):
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[1, 1])
            loose = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.5)
            tight = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.999)
        assert loose["count"] > tight["count"]

    async def test_scores_come_back_ordered_best_first(self):
        async with get_sessionmaker()() as db:
            qid, _ = await _scene(db, classes=[1, 1, 1, 1])
            res = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.0)
        scores = [c["score"] for c in res["candidates"]]
        assert scores == sorted(scores, reverse=True)

    async def test_a_rejected_object_is_never_a_candidate(self):
        async with get_sessionmaker()() as db:
            qid, made = await _scene(db, classes=[1])
            obj = await db.get(Object, uuid.UUID(made[0]))
            obj.state = "rejected"
            await db.commit()
            res = await correction_candidates(db, qid, kind="class", old_class_id=1, threshold=0.0)
        assert made[0] not in {c["object_id"] for c in res["candidates"]}
