"""The label agent's decisions, and what it may and may not do to a label.

The decision rule is pinned without models, because every threshold in it was measured and should only move
with a new measurement. The database tests pin the guarantees that do not depend on any threshold: a person's
label is never touched, nothing is ever accepted, a fixed label stays in review, and every change comes back
out exactly, by whole run or by single object.
"""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from core.config import LabelAgentSettings
from services.labelagent.agent import ASK, FIX, FLAG, KEEP, Votes, decide
from services.labelagent.memory import Vote
from services.labelagent.truth import NOT_OBJECT
from services.labelagent.zeroshot import ZsAnswer

POLICY = LabelAgentSettings(enabled=True, min_rulings=10)
TRUCK, AUTO = 1, 2


def name_of(c):
    return {TRUCK: "truck", AUTO: "autorickshaw", NOT_OBJECT: "not an object"}.get(c, str(c))


def v(mem=(AUTO, 0.8), probe=(AUTO, 0.5), zs=None) -> Votes:
    return Votes(memory=Vote(mem[0], share=0.9, margin=mem[1]) if mem else None,
                 probe=(probe[0], 0.9, probe[1]) if probe else None,
                 zs=ZsAnswer(zs[0], zs[1], {}) if zs else None)


def test_strong_agreement_on_another_class_fixes_it():
    d = decide(TRUCK, v(), POLICY, name_of)
    assert d.action == FIX and d.label == AUTO and d.question == "truck or autorickshaw?"


def test_agreement_on_not_an_object_flags_instead_of_deleting():
    d = decide(TRUCK, v(mem=(NOT_OBJECT, 0.8), probe=(NOT_OBJECT, 0.5)), POLICY, name_of)
    assert d.action == FLAG and "real object" in d.question


def test_weaker_agreement_only_asks():
    d = decide(TRUCK, v(mem=(AUTO, 0.35), probe=(AUTO, 0.25)), POLICY, name_of)
    assert d.action == ASK and d.label == AUTO


def test_a_confident_zero_shot_dissent_turns_a_fix_into_a_question():
    d = decide(TRUCK, v(zs=(TRUCK, 0.6)), POLICY, name_of)
    assert d.action == ASK


def test_a_hesitant_zero_shot_dissent_does_not_veto():
    assert decide(TRUCK, v(zs=(TRUCK, 0.1)), POLICY, name_of).action == FIX


@pytest.mark.parametrize("votes", [
    v(probe=(TRUCK, 0.9)),                      # memory and probe disagree
    v(mem=(TRUCK, 0.9), probe=(TRUCK, 0.9)),    # both confirm the label
    v(probe=None),                              # a specialist had nothing to say
    v(mem=(AUTO, 0.1), probe=(AUTO, 0.1)),      # agreement too weak
])
def test_everything_else_keeps_the_label(votes):
    assert decide(TRUCK, votes, POLICY, name_of).action == KEEP


# ------------------------------------------------------------------------------------------------ database

def _cluster(direction: int, dim: int, seed: int, noise: float = 0.02) -> list[float]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, noise, dim)
    a[direction] += 1.0
    return (a / np.linalg.norm(a)).tolist()


async def _seed(db, truck_id: int, auto_id: int) -> dict:
    """Ruled examples of two classes in two directions of vector space, then the objects under test: a
    machine 'truck' that looks like the autorickshaws, a machine truck that looks like trucks, a person's
    truck that looks like an autorickshaw, and an accepted one."""
    from db.models import Frame, Object, ObjectEmbedding, Review
    from db.models import Session as DbSession

    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id=f"LA-{uuid.uuid4().hex[:6]}", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test"))
    await db.flush()
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=1_000, cam_id="CAM_FRONT", img_uri="s3://x/y.jpg",
                 width=1280, height=720))
    await db.flush()

    def emb(oid, direction, seed):
        db.add(ObjectEmbedding(object_id=oid, siglip_vec=_cluster(direction, 1152, seed),
                               dino_vec=_cluster(direction, 768, seed + 7), model_versions={}))

    # 12 ruled examples per class: a person reclassified a machine guess into each
    for n in range(24):
        cls, direction = (truck_id, 0) if n < 12 else (auto_id, 1)
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=cls, bbox=[0, 0, 50, 50], conf=0.5,
                      source="human", state="review"))
        await db.flush()
        emb(oid, direction, 100 + n)
        db.add(Review(object_id=oid, reviewer="admin", action="reclassify", ts_ns=time.time_ns(),
                      before={"class_id": auto_id if cls == truck_id else truck_id, "source": "fused",
                              "bbox": [0, 0, 50, 50], "conf": 0.4}, after={"class_id": cls}))

    made = {}
    for key, direction, source, state in (("wrong", 1, "fused", "auto_accept"), ("right", 0, "fused", "review"),
                                          ("human", 1, "human", "review"), ("accepted", 1, "fused", "accepted")):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=truck_id, bbox=[0, 0, 50, 50], conf=0.6,
                      source=source, state=state))
        await db.flush()
        emb(oid, direction, 900 + len(made))
        made[key] = oid
    await db.commit()
    return {"session_id": sid, **made}


@pytest.fixture
def no_text_encoder(monkeypatch):
    """The zero-shot needs the SigLIP 2 text tower; these tests are about what the agent does with votes,
    so it abstains. The fix gate is opened here; test_a_closed_gate_* closes it."""
    import services.labelagent.agent as agent
    import services.labelagent.report as report

    monkeypatch.setattr(agent, "classify", lambda *a, **k: ZsAnswer(None, 0.0, {}))
    monkeypatch.setattr(report, "fixes_allowed", lambda policy: (True, "test: gate open"))
    agent._cache.clear()


# ------------------------------------------------------------------------------------------------ the gate

def _report(upper):
    return {"stages": {"agent": {"all": {"pooled": {"wrong_fix_upper95": upper}}}}}


def test_fixes_are_licensed_only_by_a_measured_rate_under_the_limit():
    from services.labelagent.report import gate_from

    assert gate_from(_report(0.015), 0.02)[0] is True
    ok, why = gate_from(_report(0.039), 0.02)
    assert ok is False and "3.9%" in why
    assert gate_from(None, 0.02)[0] is False                       # no report, no fixes
    assert gate_from({"stages": {}}, 0.02)[0] is False             # a report without the agent, no fixes


@pytest.mark.db
async def test_the_agent_fixes_a_machine_label_and_leaves_people_alone(no_text_encoder):
    from db.models import AgentRun, Object
    from db.session import get_sessionmaker
    from services.agent.runs import revert_run
    from services.autolabel.ontology import get_ontology
    from services.labelagent.agent import run_session

    onto = get_ontology()
    truck, auto = onto.by_name("truck").id, onto.by_name("autorickshaw").id
    async with get_sessionmaker()() as db:
        ids = await _seed(db, truck, auto)
        dry = await run_session(db, ids["session_id"], policy=POLICY)
    assert dry["applied"] is False and dry["decisions"].get(FIX) == 1
    async with get_sessionmaker()() as db:
        assert (await db.get(Object, ids["wrong"])).class_id == truck      # a dry run changes nothing

    async with get_sessionmaker()() as db:
        out = await run_session(db, ids["session_id"], apply=True, policy=POLICY)
    assert out["applied"] and out["decisions"] == {KEEP: 1, FIX: 1}       # human + accepted never looked at
    async with get_sessionmaker()() as db:
        wrong = await db.get(Object, ids["wrong"])
        assert wrong.class_id == auto
        assert wrong.state == "review"                                     # was auto_accept: withheld, not kept
        assert wrong.provenance["labelagent"]["action"] == FIX
        assert (await db.get(Object, ids["right"])).class_id == truck
        assert (await db.get(Object, ids["human"])).class_id == truck
        assert (await db.get(Object, ids["accepted"])).state == "accepted"
        run = await db.get(AgentRun, uuid.UUID(out["run_id"]))
        assert run.kind == "labelagent" and set(run.changes) == {str(ids["wrong"])}

    async with get_sessionmaker()() as db:
        await revert_run(db, uuid.UUID(out["run_id"]))
    async with get_sessionmaker()() as db:
        wrong = await db.get(Object, ids["wrong"])
        assert (wrong.class_id, wrong.state) == (truck, "auto_accept")
        assert "labelagent" not in wrong.provenance


@pytest.mark.db
async def test_one_fix_can_be_undone_unless_a_person_took_over(no_text_encoder):
    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelagent.agent import run_session, undo_object

    onto = get_ontology()
    truck, auto = onto.by_name("truck").id, onto.by_name("autorickshaw").id
    async with get_sessionmaker()() as db:
        ids = await _seed(db, truck, auto)
        await run_session(db, ids["session_id"], apply=True, policy=POLICY)
    async with get_sessionmaker()() as db:
        got = await undo_object(db, ids["wrong"])
    assert got["class_id"] == truck and got["state"] == "auto_accept"
    async with get_sessionmaker()() as db:
        with pytest.raises(ValueError):
            await undo_object(db, ids["wrong"])                            # nothing left to undo

    async with get_sessionmaker()() as db:
        ids2 = await _seed(db, truck, auto)
        await run_session(db, ids2["session_id"], apply=True, policy=POLICY)
        obj = await db.get(Object, ids2["wrong"])
        obj.source = "human"                                               # a person edited it since
        await db.commit()
    async with get_sessionmaker()() as db:
        with pytest.raises(ValueError, match="person"):
            await undo_object(db, ids2["wrong"])


@pytest.mark.db
async def test_a_closed_gate_turns_every_fix_into_a_question(no_text_encoder, monkeypatch):
    import services.labelagent.report as report
    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelagent.agent import run_session

    monkeypatch.setattr(report, "fixes_allowed", lambda policy: (False, "test: gate closed"))
    onto = get_ontology()
    truck, auto = onto.by_name("truck").id, onto.by_name("autorickshaw").id
    async with get_sessionmaker()() as db:
        ids = await _seed(db, truck, auto)
        out = await run_session(db, ids["session_id"], apply=True, policy=POLICY)
    assert out["decisions"].get(FIX) is None and out["decisions"].get(ASK) == 1
    assert out["fixes_allowed"] is False
    async with get_sessionmaker()() as db:
        wrong = await db.get(Object, ids["wrong"])
        assert wrong.class_id == truck                               # the class did not move
        assert wrong.state == "review"                               # but acceptance is still withheld
        assert wrong.provenance["labelagent"]["action"] == ASK
        assert "test: gate closed" in wrong.provenance["labelagent"]["reasons"]
