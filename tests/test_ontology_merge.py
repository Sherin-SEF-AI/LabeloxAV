"""Merging a class must move the corpus without rewriting history.

The custom sidecar grew three spellings of one traffic signal, two of one autorickshaw with one misspelled,
duplicates of governed classes, and a class left by a test. That is not inert: `classify_crop` scores every
class the ontology knows, so the relabel agent proposed the misspelling for real objects and died outright
on the test class, which was proposable and unstorable at once.

The two properties worth defending are about what a class id is referenced by. `object` and `track` are the
corpus's present statement and move. `prediction` and `eval_patch` are records of what a model said and what
a measurement found, and rewriting those would change numbers that have already been reported. That is also
why a merged class is retired from the sidecar rather than deleted from the database: those historical rows
hold foreign keys into it.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns
from db.models import AgentRun, Frame, Object, OntologyClass, Prediction, Track
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.agent.ontology_merge import (
    MergeError,
    _sidecar_path,
    merge_class,
    rename_in_sidecar,
    retire_from_sidecar,
)
from services.agent.runs import revert_run

pytestmark = pytest.mark.db

SRC_ID, DST_ID = 9101, 9102


@pytest.fixture
def sidecar_restored():
    """The sidecar is a tracked repo file. Keep its exact bytes and put them back, whatever happens.

    Learned the hard way: a sibling test used to re-serialise it instead of restoring it, which quietly
    reformatted a governed file on every suite run.
    """
    p = _sidecar_path()
    original = p.read_text() if p.exists() else None
    yield p
    if original is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(original)
    from services.autolabel.ontology import get_ontology

    get_ontology.cache_clear()


async def _seed(db, *, n_objects: int = 3, human: int = 0) -> tuple[uuid.UUID, list[uuid.UUID]]:
    for cid, name in ((SRC_ID, "test_merge_src"), (DST_ID, "test_merge_dst")):
        if await db.get(OntologyClass, cid) is None:
            db.add(OntologyClass(id=cid, name=name, l0="object", l1="custom", india=True, map_to={}))
    await db.flush()

    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-MERGE", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/t.jpg", width=100, height=100)
    db.add(frame)
    await db.flush()

    ids = []
    for i in range(n_objects):
        o = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=SRC_ID,
                   bbox=[1.0, 1.0, 20.0, 20.0], conf=0.9,
                   source="human" if i < human else "fused", state="accepted")
        db.add(o)
        ids.append(o.object_id)
    await db.commit()
    return frame.frame_id, ids


async def test_a_merge_moves_objects_to_the_target_class():
    async with get_sessionmaker()() as db:
        _, ids = await _seed(db, n_objects=4)
        out = await merge_class(db, from_id=SRC_ID, to_id=DST_ID)
        assert out["objects"] == 4
        left = (await db.execute(
            select(Object).where(Object.object_id.in_(ids)))).scalars().all()
        assert {o.class_id for o in left} == {DST_ID}


async def test_a_merge_moves_tracks_too():
    """A track carries its own class, and leaving it behind would make the track disagree with every object
    on it."""
    async with get_sessionmaker()() as db:
        frame_id, _ = await _seed(db, n_objects=1)
        sess_id = (await db.execute(select(Frame.session_id).where(Frame.frame_id == frame_id))).scalar()
        t = Track(track_id=uuid.uuid4(), session_id=sess_id, class_id=SRC_ID,
                  first_ts_ns=now_ns(), last_ts_ns=now_ns())
        db.add(t)
        await db.commit()

        out = await merge_class(db, from_id=SRC_ID, to_id=DST_ID)
        assert out["tracks"] >= 1
        assert (await db.get(Track, t.track_id)).class_id == DST_ID


async def test_predictions_are_never_rewritten():
    """The invariant that shapes the whole module. A prediction records what a model said at a moment; a
    merge that edited it would change a measurement already reported."""
    async with get_sessionmaker()() as db:
        frame_id, _ = await _seed(db, n_objects=1)
        from db.models import InferenceRun, ModelRegistry

        mv = "m-merge-test"
        if await db.get(ModelRegistry, mv) is None:
            db.add(ModelRegistry(model_version=mv, weights_uri="s3://w.pt"))
            await db.flush()
        run = InferenceRun(run_id=uuid.uuid4(), model_version=mv, status="complete", params={})
        db.add(run)
        await db.flush()
        p = Prediction(run_id=run.run_id, frame_id=frame_id, class_id=SRC_ID,
                       bbox=[1.0, 1.0, 20.0, 20.0], conf=0.8)
        db.add(p)
        await db.commit()

        await merge_class(db, from_id=SRC_ID, to_id=DST_ID)
        await db.refresh(p)
        assert p.class_id == SRC_ID, "history must keep pointing at the class it was recorded against"


async def test_the_merged_class_row_survives():
    """Retiring is not deleting: prediction and eval_patch hold foreign keys into it."""
    async with get_sessionmaker()() as db:
        await _seed(db, n_objects=1)
        await merge_class(db, from_id=SRC_ID, to_id=DST_ID)
        assert await db.get(OntologyClass, SRC_ID) is not None


async def test_a_merge_is_reversible():
    async with get_sessionmaker()() as db:
        _, ids = await _seed(db, n_objects=3)
        out = await merge_class(db, from_id=SRC_ID, to_id=DST_ID)
        await revert_run(db, uuid.UUID(out["run_id"]))
        back = (await db.execute(select(Object).where(Object.object_id.in_(ids)))).scalars().all()
        assert {o.class_id for o in back} == {SRC_ID}
        assert (await db.get(AgentRun, uuid.UUID(out["run_id"]))).status == "reverted"


async def test_reverting_restores_human_labelled_objects_too():
    """The generic agent revert refuses to touch anything a person owns, which is right when an agent
    overruled them and wrong here. A person labelled the object with a class that has since been retired;
    undoing the retirement has to give them their class back."""
    async with get_sessionmaker()() as db:
        _, ids = await _seed(db, n_objects=3, human=2)
        out = await merge_class(db, from_id=SRC_ID, to_id=DST_ID)
        await revert_run(db, uuid.UUID(out["run_id"]))
        back = (await db.execute(select(Object).where(Object.object_id.in_(ids)))).scalars().all()
        assert {o.class_id for o in back} == {SRC_ID}
        assert sum(1 for o in back if o.source == "human") == 2


async def test_merging_into_a_class_that_does_not_exist_is_refused():
    """Otherwise the objects land on a foreign key that fails, inside a background task."""
    async with get_sessionmaker()() as db:
        await _seed(db, n_objects=1)
        with pytest.raises(MergeError, match="nowhere to go"):
            await merge_class(db, from_id=SRC_ID, to_id=987654)


async def test_merging_a_class_into_itself_is_refused():
    async with get_sessionmaker()() as db:
        await _seed(db, n_objects=1)
        with pytest.raises(MergeError, match="into itself"):
            await merge_class(db, from_id=SRC_ID, to_id=SRC_ID)


# ------------------------------------------------------------------------- the sidecar

def test_retiring_removes_the_name_the_classifier_can_propose(sidecar_restored):
    """This is what actually takes a class out of circulation: get_ontology merges the governed YAML with
    the sidecar, and classify_crop scores against whatever that produces."""
    p = sidecar_restored
    entries = json.loads(p.read_text())
    victim = entries[0]["id"]

    out = retire_from_sidecar({int(victim)})
    remaining = {int(e["id"]) for e in json.loads(p.read_text())}
    assert int(victim) not in remaining
    assert out["remaining"] == len(entries) - 1

    from services.autolabel.ontology import get_ontology

    assert victim not in {c.id for c in get_ontology().classes}


def test_retiring_leaves_the_other_classes_alone(sidecar_restored):
    p = sidecar_restored
    before = {int(e["id"]) for e in json.loads(p.read_text())}
    victim = next(iter(before))
    retire_from_sidecar({victim})
    after = {int(e["id"]) for e in json.loads(p.read_text())}
    assert after == before - {victim}


def test_renaming_keeps_the_id(sidecar_restored):
    """The id is what every object, track and prediction references. A rename that changed it would be a
    different class wearing the same label."""
    p = sidecar_restored
    entries = json.loads(p.read_text())
    target = int(entries[0]["id"])

    out = rename_in_sidecar(target, "A Corrected  Name")
    assert out["to"] == "a_corrected_name", "names normalise the same way the create path normalises them"
    now = {int(e["id"]): e["name"] for e in json.loads(p.read_text())}
    assert now[target] == "a_corrected_name"


def test_renaming_a_class_that_is_not_in_the_sidecar_is_refused(sidecar_restored):
    with pytest.raises(MergeError, match="not in the sidecar"):
        rename_in_sidecar(987654, "whatever")


def test_an_empty_rename_is_refused(sidecar_restored):
    entries = json.loads(sidecar_restored.read_text())
    with pytest.raises(MergeError, match="letters or digits"):
        rename_in_sidecar(int(entries[0]["id"]), "!!!")
