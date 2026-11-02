"""The assistant: reading tools answer, changing tools only propose, and a reviewer's approval applies exactly
what was previewed. A scripted model stands in for the LLM, so these run without one."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from core.timebase import now_ns, seconds_to_ns
from services.assistant.llm import Reply, ToolCall

pytestmark = pytest.mark.db


class Scripted:
    """Replies in order; records what it was sent, so a test can check tool results reached it."""

    name = "scripted"

    def __init__(self, *replies: Reply):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, system, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else Reply(content="done")


def call(tool: str, /, **args) -> Reply:
    return Reply(tool_calls=[ToolCall(id=f"c{uuid.uuid4().hex[:6]}", name=tool, arguments=args)])


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _seed(visibility: str = "team") -> dict:
    """A session in its own project: three machine hatchbacks, one human hatchback, one machine cyclist."""
    from db.models import Frame, LabelProject, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    hatch, cyclist = onto.by_name("hatchback").id, onto.by_name("cyclist").id

    async def _go():
        pid, sid, fid, ts = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), now_ns()
        ids: dict[str, list[str]] = {"machine": [], "human": [], "cyclist": []}
        async with get_sessionmaker()() as db:
            db.add(LabelProject(project_id=pid, name=f"assist-{pid.hex[:6]}", visibility=visibility))
            await db.flush()
            db.add(DbSession(session_id=sid, project_id=pid, vehicle_id=f"ASSIST-{sid.hex[:4]}", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(1), city="DEL", sensors={}, ontology_version=onto.version))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/a.jpg",
                         width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            for i, (cid, source, key) in enumerate([(hatch, "fused", "machine")] * 3 + [(hatch, "human", "human"),
                                                                                       (cyclist, "fused", "cyclist")]):
                oid = uuid.uuid4()
                ids[key].append(str(oid))
                x = 10.0 + 70 * i
                db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[x, 10.0, x + 50, 60.0], conf=0.6,
                              source=source, state="review" if source != "human" else "accepted", attrs={},
                              provenance={}, version=1))
            await db.commit()
        return {"project_id": str(pid), "session_id": str(sid), "frame_id": str(fid), **ids}

    return _run(_go())


async def _classes(ids: list[str]) -> dict[str, tuple[int, str]]:
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Object.object_id, Object.class_id, Object.state)
                                 .where(Object.object_id.in_([uuid.UUID(i) for i in ids])))).all()
    return {str(o): (c, s) for o, c, s in rows}


def _turn(model: Scripted, text: str, *, session_id: str | None = None, hidden: frozenset = frozenset()) -> dict:
    from db.session import get_sessionmaker
    from services.assistant.loop import run_turn
    from services.assistant.tools import Context

    async def _go():
        async with get_sessionmaker()() as db:
            ctx = Context(db=db, user=None, hidden=hidden, request=text, session_id=session_id)
            return await run_turn(ctx, [{"role": "user", "content": text}], model)

    return _run(_go())


def test_a_reading_tool_answers_and_its_result_reaches_the_model():
    s = _seed()
    model = Scripted(call("count_objects", class_names=["hatchback"], session_id=s["session_id"]),
                     Reply(content="There are 4 hatchbacks."))
    out = _turn(model, "how many hatchbacks?")
    assert out["reply"] == "There are 4 hatchbacks." and out["proposals"] == []
    assert out["trace"][0]["tool"] == "count_objects" and out["trace"][0]["error"] is None
    tool_msg = next(m for m in model.seen[1] if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["total"] == 4


def test_a_bad_argument_goes_back_to_the_model_instead_of_failing_the_turn():
    s = _seed()
    model = Scripted(call("count_objects", class_names=["hatchbak"]),
                     call("count_objects", class_names=["hatchback"], session_id=s["session_id"]),
                     Reply(content="4"))
    out = _turn(model, "how many hatchbacks?")
    assert out["trace"][0]["error"] and "hatchbak" in out["trace"][0]["error"]
    assert out["trace"][1]["error"] is None and out["reply"] == "4"


def test_a_change_is_only_proposed_and_leaves_human_labels_out():
    s = _seed()
    model = Scripted(call("propose_reclassify", from_classes=["hatchback"], to_class="sedan",
                          session_id=s["session_id"]), Reply(content="Proposed; waiting for a reviewer."))
    out = _turn(model, "make the hatchbacks sedans")
    (p,) = out["proposals"]
    assert p["preview"]["count"] == 3 and p["preview"]["human_labels_left_alone"] == 1
    # Nothing changed yet.
    from services.autolabel.ontology import get_ontology

    hatch = get_ontology().by_name("hatchback").id
    assert all(c == hatch for c, _ in _run(_classes(s["machine"] + s["human"])).values())


def test_a_change_needs_a_scope():
    model = Scripted(call("propose_reclassify", from_classes=["hatchback"], to_class="sedan"), Reply(content="?"))
    out = _turn(model, "make every hatchback a sedan")
    assert out["proposals"] == [] and "scope" in (out["trace"][0]["error"] or "")


def test_approval_applies_exactly_what_was_previewed_and_only_once(monkeypatch):
    from _authutil import auth_headers

    from services.api.main import app
    from services.autolabel.ontology import get_ontology

    s = _seed()
    model = Scripted(call("propose_reclassify", from_classes=["hatchback"], to_class="sedan",
                          session_id=s["session_id"]), Reply(content="Proposed."))
    monkeypatch.setattr("services.assistant.loop.make_model", lambda: model)
    with TestClient(app) as c:
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "hatchbacks to sedan"}],
                                                "session_id": s["session_id"]}, headers=auth_headers("annotator"))
        assert r.status_code == 200, r.text
        pid = r.json()["proposals"][0]["proposal_id"]
        assert c.post(f"/api/assistant/proposals/{pid}/approve", headers=auth_headers("annotator")).status_code == 403
        r = c.post(f"/api/assistant/proposals/{pid}/approve", headers=auth_headers("reviewer"))
        assert r.status_code == 200 and r.json()["status"] == "approved", r.text
        assert r.json()["result"]["edited"] == 3
        assert c.post(f"/api/assistant/proposals/{pid}/approve", headers=auth_headers("reviewer")).status_code == 409

    onto = get_ontology()
    after = _run(_classes(s["machine"] + s["human"] + s["cyclist"]))
    assert all(after[i] == (onto.by_name("sedan").id, "review") for i in s["machine"])
    assert after[s["human"][0]][0] == onto.by_name("hatchback").id, "a human label is never changed"
    assert after[s["cyclist"][0]][0] == onto.by_name("cyclist").id


def test_rejecting_changes_nothing_and_a_broken_proposal_fails_with_its_reason():
    from _authutil import auth_headers

    from db.models import AssistantProposal
    from db.session import get_sessionmaker
    from services.api.main import app

    async def _make(tool: str) -> str:
        async with get_sessionmaker()() as db:
            p = AssistantProposal(tool=tool, args={}, summary=f"test {tool}", preview={})
            db.add(p)
            await db.commit()
            return str(p.proposal_id)

    rejected, broken = _run(_make("propose_reclassify")), _run(_make("propose_teleport"))
    with TestClient(app) as c:
        r = c.post(f"/api/assistant/proposals/{rejected}/reject", headers=auth_headers("reviewer"))
        assert r.status_code == 200 and r.json()["status"] == "rejected"
        r = c.post(f"/api/assistant/proposals/{broken}/approve", headers=auth_headers("reviewer"))
        assert r.status_code == 200 and r.json()["status"] == "failed"
        assert "propose_teleport" in r.json()["result"]["error"]


def test_hidden_projects_stay_hidden_from_the_assistant():
    from _authutil import auth_headers

    from services.api.main import app

    s = _seed(visibility="members")
    with TestClient(app) as c:
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                                "session_id": s["session_id"]}, headers=auth_headers("annotator"))
        assert r.status_code == 404
    # And through the tools themselves: a hidden session reads as not found.
    model = Scripted(call("session_overview", session_id=s["session_id"]), Reply(content="?"))
    out = _turn(model, "what is in it", hidden=frozenset({uuid.UUID(s["project_id"])}))
    assert "not found" in (out["trace"][0]["error"] or "")


def test_the_turn_stops_at_its_step_budget():
    model = Scripted(*[call("list_classes", query="car") for _ in range(20)])
    out = _turn(model, "loop forever")
    assert "ran out of steps" in out["reply"] and len(out["trace"]) == 6


def test_the_eval_scores_tool_choice_and_arguments(monkeypatch):
    from db.session import get_sessionmaker
    from services.assistant import eval as ev

    async def _anchor(_db):
        return "S", "F"

    monkeypatch.setattr(ev, "_anchor", _anchor)
    by_text = {c.text: c for c in ev.cases("S", "F")}

    class FirstAcceptable:
        """Always calls an acceptable tool, with no arguments: right tool, mostly wrong arguments."""
        name = "first-acceptable"

        def chat(self, system, messages, tools):
            return call(sorted(by_text[messages[-1]["content"]].accept)[0])

    async def _go():
        async with get_sessionmaker()() as db:
            return await ev.evaluate(db, FirstAcceptable())

    out = _run(_go())
    assert out["cases"] == len(by_text)
    assert out["tool_accuracy"] == 1.0
    assert 0 < out["full_accuracy"] < 1.0


def test_the_newer_reading_tools_answer_within_what_the_caller_may_see():
    s = _seed(visibility="members")
    hidden = frozenset({uuid.UUID(s["project_id"])})
    model = Scripted(call("list_projects"), call("model_status"), call("job_overview"), call("list_datasets"),
                     Reply(content="ok"))
    out = _turn(model, "status of everything", hidden=hidden)
    assert [t["tool"] for t in out["trace"]] == ["list_projects", "model_status", "job_overview", "list_datasets"]
    assert all(t["error"] is None for t in out["trace"])
    results = [json.loads(m["content"]) for m in model.seen[-1] if m["role"] == "tool"]
    assert s["project_id"] not in {p["project_id"] for p in results[0]["projects"]}, "a hidden project was listed"
    assert set(results[1]) == {"champions", "recent_training"}
    assert {"autolabel", "import", "training", "export"} <= set(results[2])


def test_an_unknown_export_format_goes_back_to_the_model():
    s = _seed()
    model = Scripted(call("propose_export", session_id=s["session_id"], formats=["tiff"]), Reply(content="?"))
    out = _turn(model, "export as tiff")
    assert out["proposals"] == [] and "unknown format" in (out["trace"][0]["error"] or "")


def test_export_and_training_proposals_apply_through_the_normal_paths(monkeypatch):
    from _authutil import auth_headers
    from sqlalchemy import select

    from db.models import ExportJob, LabelProject, TrainingJob
    from db.session import get_sessionmaker
    from services.api.main import app

    s = _seed()

    async def _project_name() -> str:
        async with get_sessionmaker()() as db:
            return (await db.get(LabelProject, uuid.UUID(s["project_id"]))).name

    pname = _run(_project_name())
    model = Scripted(call("propose_export", session_id=s["session_id"], formats=["coco"], name="assist-export"),
                     call("propose_training", project_name=pname, epochs=5), Reply(content="Proposed both."))
    monkeypatch.setattr("services.assistant.loop.make_model", lambda: model)
    with TestClient(app) as c:
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "export and train"}]},
                   headers=auth_headers("annotator"))
        assert r.status_code == 200, r.text
        export, training = r.json()["proposals"]
        assert export["preview"]["count"] == 5 and "assist-export" in export["summary"]
        assert "not promoted" in training["summary"]
        for p in (export, training):
            d = c.post(f"/api/assistant/proposals/{p['proposal_id']}/approve", headers=auth_headers("reviewer"))
            assert d.status_code == 200 and d.json()["status"] == "approved", d.text

    async def _jobs():
        async with get_sessionmaker()() as db:
            e = (await db.execute(select(ExportJob).where(ExportJob.name == "assist-export"))).scalars().first()
            t = (await db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc()))).scalars().first()
            return e, t

    e, t = _run(_jobs())
    assert e is not None and e.spec["session_id"] == s["session_id"]
    assert t is not None and t.config["dataset_spec"]["project_id"] == s["project_id"]
    assert t.config["promote"] is False and t.config["hparams"]["epochs"] == 5


def test_training_on_everything_must_be_asked_for_not_fallen_into():
    _seed()
    _seed()  # two projects at least
    model = Scripted(call("propose_training", epochs=5), call("propose_training", epochs=5, all_projects=True),
                     Reply(content="done"))
    out = _turn(model, "train a model")
    assert out["trace"][0]["error"] and "all_projects" in out["trace"][0]["error"]
    (p,) = out["proposals"]
    assert "every project" in p["summary"]
