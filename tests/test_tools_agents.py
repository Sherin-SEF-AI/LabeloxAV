"""Agent F's tools: bulk edits in plain language across many sessions (#17) and the overnight autopilot (#18).

#17: a sentence parses into a scope (city, vehicle, dates, scene and state terms, a dataset query) plus an
operation; the preview counts per session and class and says what it leaves alone; a reviewer's approval
applies exactly the previewed ids as one reversible run; human-confirmed labels never change.

#18: the night window honours office hours; the scheduler starts a night once, only when switched on; a night
keeps auto-accepts only for ladder-allowed classes the VLM and the error sweep cleared, and sends the rest to
review as one revertible run; the kill switch stops it; a dry run changes nothing; the every-tick agents skip
GPU work in office hours.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from core.timebase import now_ns, seconds_to_ns
from services.agent.nl_edit import parse_bulk_edit
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db

IST_TZ = timezone(timedelta(hours=5, minutes=30))


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _ist(h: int, m: int = 0, d: date = date(2026, 11, 26)) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, tzinfo=IST_TZ).astimezone(UTC)


# ---- #17 parsing ----------------------------------------------------------------------------------------------

def test_a_sentence_splits_into_scope_and_edit():
    onto = get_ontology()
    p = parse_bulk_edit("relabel every pickup in Delhi sessions as LCV", onto)
    assert p["operation"] == "reclassify"
    assert [onto.by_id(i).name for i in p["select_class_ids"]] == ["pickup"]
    assert onto.by_id(p["to_class_id"]).name == "lcv"
    assert p["scope"]["cities"] == ["DEL"]

    p = parse_bulk_edit("send all cyclists at night in Bengaluru drives back for review", onto)
    assert p["operation"] == "flag" and [onto.by_id(i).name for i in p["select_class_ids"]] == ["cyclist"]
    assert p["scope"]["cities"] == ["BLR"] and p["scope"]["predicate"] == {"time_of_day": ["night"]}

    p = parse_bulk_edit("relabel pickups as lcv from vehicle INDIA-DEL-01 in the last 7 days", onto,
                        today=date(2026, 11, 26))
    assert p["scope"]["vehicles"] == ["INDIA-DEL-01"] and onto.by_id(p["to_class_id"]).name == "lcv"
    assert (p["scope"]["date_from"], p["scope"]["date_to"]) == ("2026-11-20", "2026-11-27")

    p = parse_bulk_edit("relabel every pickup to lcv where night AND urban", onto)
    assert p["scope"]["predicate"] == {"time_of_day": ["night"], "road_type": ["urban"]}

    # An unknown dataset-query term is refused, not silently ignored (the query_lang rule).
    p = parse_bulk_edit("relabel every pickup to lcv where night AND unocluded", onto)
    assert p["errors"]

    # A word that is itself a class does not widen to its whole group.
    p = parse_bulk_edit("send every autorickshaw in Mumbai back for review", onto)
    assert [onto.by_id(i).name for i in p["select_class_ids"]] == ["autorickshaw"]


def test_scene_and_state_words_come_from_the_dataset_vocabulary():
    from services.datasets.query_lang import terms_in_text

    r = terms_in_text("unreviewed motorcycles in rainy streets at night")
    assert r["predicate"] == {"states": ["review", "annotate"], "weather": ["rain"], "time_of_day": ["night"]}


# ---- #17 preview, approve, revert ----------------------------------------------------------------------------

def _seed_bulk() -> dict:
    """Two sessions in Kochi (one night frame, one day frame each) and one in Coimbatore, with pickups and
    cyclists in every review state a person or machine can leave them in."""
    from db.models import Frame, LabelProject, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    onto = get_ontology()
    pickup, cyclist = onto.by_name("pickup").id, onto.by_name("cyclist").id
    tag = uuid.uuid4().hex[:6].upper()

    async def _go():
        out: dict = {"sessions": [], "editable": [], "human": [], "accepted": [], "vlm": [], "cyclist_night": [],
                     "cyclist_day": [], "other_city": [], "tag": tag}
        pid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(LabelProject(project_id=pid, name=f"bulk-{tag}"))
            await db.flush()
            for k, city in enumerate(["COK", "COK", "CJB"]):
                sid, ts = uuid.uuid4(), now_ns() - seconds_to_ns(3600 * k)
                out["sessions"].append(str(sid))
                db.add(DbSession(session_id=sid, project_id=pid, vehicle_id=f"T{tag}-{city}-{k:02d}", start_ts_ns=ts,
                                 end_ts_ns=ts + seconds_to_ns(60), city=city, sensors={},
                                 ontology_version=onto.version))
                await db.flush()
                for tod in ("night", "day"):
                    fid = uuid.uuid4()
                    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/a.jpg",
                                 width=1920, height=1080, quality=0.9, scene={"time_of_day": tod}))
                    await db.flush()
                    rows = [(pickup, "fused", "review", "editable"), (pickup, "auto_accept", "auto_accept", "editable"),
                            (pickup, "human", "accepted", "human"), (pickup, "fused", "accepted", "accepted"),
                            (pickup, "vlm_review", "accepted", "vlm"), (pickup, "fused", "rejected", None),
                            (cyclist, "fused", "review", f"cyclist_{tod}")]
                    for i, (cid, source, state, key) in enumerate(rows):
                        oid = uuid.uuid4()
                        x = 10.0 + 60 * i
                        db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[x, 10.0, x + 50, 60.0],
                                      conf=0.7, source=source, state=state, attrs={}, provenance={}, version=1))
                        if key is None:
                            continue
                        if city == "CJB" and key != "cyclist_night":
                            out["other_city"].append(str(oid))
                        elif city == "CJB":
                            out["cyclist_night_cjb"] = out.get("cyclist_night_cjb", []) + [str(oid)]
                        else:
                            out[key].append(str(oid))
            await db.commit()
        return out

    return _run(_go())


async def _state(ids: list[str]) -> dict[str, tuple[int, str]]:
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Object.object_id, Object.class_id, Object.state)
                                 .where(Object.object_id.in_([uuid.UUID(i) for i in ids])))).all()
    return {str(o): (int(c), s) for o, c, s in rows}


def test_preview_counts_per_session_and_leaves_people_alone():
    from db.session import get_sessionmaker
    from services.agent.nl_edit import resolve_bulk

    s = _seed_bulk()
    onto = get_ontology()

    async def _go():
        async with get_sessionmaker()() as db:
            return await resolve_bulk(db, parse_bulk_edit("relabel every pickup in Kochi sessions as LCV", onto))

    res = _run(_go())
    mine = [r for r in res["by_session"] if r["session_id"] in s["sessions"]]
    # Two Kochi sessions x two frames x (review + auto_accept + vlm_review accepted) = 12, all pickups.
    assert len(mine) == 2 and sum(r["total"] for r in mine) == 12
    assert all(r["classes"] == {"pickup": 6} for r in mine)
    assert s["sessions"][2] not in {r["session_id"] for r in res["by_session"]}, "Coimbatore is not in Kochi"
    # Human-drawn and reviewer-accepted pickups are left alone; rejected ones are not even counted.
    assert res["left_alone"]["human_confirmed"] >= 8
    assert set(s["editable"] + s["vlm"]) <= set(res["object_ids"])
    assert not set(s["human"] + s["accepted"]) & set(res["object_ids"])
    assert res["can_propose"] and res["sample"] and res["sample"][0]["crop_url"].startswith("/api/objects/")

    async def _night():
        async with get_sessionmaker()() as db:
            return await resolve_bulk(db, parse_bulk_edit(
                "send all cyclists at night in Kochi drives back for review", onto))

    night = _run(_night())
    assert set(s["cyclist_night"]) <= set(night["object_ids"])
    assert not set(s["cyclist_day"]) & set(night["object_ids"])


def test_approval_applies_the_preview_once_and_revert_undoes_it():
    from _authutil import auth_headers

    from services.api.main import app

    s = _seed_bulk()
    onto = get_ontology()
    vehicle = f"T{s['tag']}-COK-00"
    cmd = f"relabel every pickup from vehicle {vehicle} as lcv"
    with TestClient(app) as c:
        r = c.post("/api/tools/agents/edit/preview", json={"command": cmd}, headers=auth_headers("annotator"))
        assert r.status_code == 200, r.text
        prev = r.json()
        assert prev["count"] == 6 and prev["sessions"] == 1 and "object_ids" not in prev
        assert prev["left_alone"]["human_confirmed"] == 4

        r = c.post("/api/tools/agents/edit/propose", json={"command": cmd}, headers=auth_headers("annotator"))
        assert r.status_code == 200, r.text
        pid = r.json()["proposal_id"]
        assert r.json()["status"] == "pending" and r.json()["preview"]["count"] == 6
        # Nothing changed yet.
        assert all(v[0] == onto.by_name("pickup").id for v in _run(_state(s["editable"])).values())

        assert c.post(f"/api/assistant/proposals/{pid}/approve", headers=auth_headers("annotator")).status_code == 403
        r = c.post(f"/api/assistant/proposals/{pid}/approve", headers=auth_headers("reviewer"))
        assert r.status_code == 200 and r.json()["status"] == "approved", r.text
        assert r.json()["result"]["edited"] == 6

        after = _run(_state(s["editable"] + s["vlm"] + s["human"] + s["accepted"]))
        lcv, pickup = onto.by_name("lcv").id, onto.by_name("pickup").id
        changed = [i for i in s["editable"] + s["vlm"] if after[i][0] == lcv]
        assert len(changed) == 6 and all(after[i][1] == "review" for i in changed)
        assert all(after[i] == (pickup, "accepted") for i in s["human"] + s["accepted"]), "people's labels stand"

        r = c.get(f"/api/tools/agents/edit/proposals/{pid}", headers=auth_headers("annotator"))
        assert r.json()["run"]["status"] == "committed" and r.json()["run"]["edited"] == 6

        assert c.post(f"/api/tools/agents/edit/proposals/{pid}/revert",
                      headers=auth_headers("annotator")).status_code == 403
        r = c.post(f"/api/tools/agents/edit/proposals/{pid}/revert", headers=auth_headers("reviewer"))
        assert r.status_code == 200 and r.json()["reverted"] == 6, r.text
    back = _run(_state(changed))
    assert all(v[0] == pickup for v in back.values())


def test_the_assistant_proposes_a_bulk_edit_from_a_sentence(monkeypatch):
    from _authutil import auth_headers

    from services.api.main import app
    from services.assistant.llm import Reply, ToolCall

    s = _seed_bulk()

    class Scripted:
        name = "scripted"

        def __init__(self, *replies):
            self.replies = list(replies)

        def chat(self, system, messages, tools):
            assert any(t["name"] == "propose_bulk_edit" for t in tools)
            return self.replies.pop(0) if self.replies else Reply(content="done")

    model = Scripted(Reply(tool_calls=[ToolCall(id="c1", name="propose_bulk_edit", arguments={
        "from_classes": ["cyclist"], "operation": "flag", "vehicles": [f"T{s['tag']}-CJB-02"], "query": "night"})]),
        Reply(content="Proposed."))
    monkeypatch.setattr("services.assistant.loop.make_model", lambda: model)
    with TestClient(app) as c:
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "night cyclists, CJB"}]},
                   headers=auth_headers("annotator"))
        assert r.status_code == 200, r.text
        props = r.json()["proposals"]
        assert len(props) == 1 and props[0]["tool"] == "propose_bulk_edit" and props[0]["preview"]["count"] == 1


# ---- #18 the night window and the scheduler ---------------------------------------------------------------------

def test_the_night_window_honours_office_hours():
    from db.models import AutopilotConfig
    from services.agent.autopilot import night_of, window_state

    cfg = AutopilotConfig(id=1, earliest_start_ist="20:00")
    assert not window_state(cfg, _ist(11))["open"] and window_state(cfg, _ist(11))["quiet"]
    w = window_state(cfg, _ist(19))
    assert not w["open"] and "20:00" in w["reason"] and w["next_open"].startswith("2026-11-26T20:00")
    assert window_state(cfg, _ist(21))["open"] and window_state(cfg, _ist(3))["open"]
    assert not window_state(cfg, _ist(10, 30))["open"]
    # An evening and the morning after are one night.
    assert night_of(_ist(23)) == night_of(_ist(2, d=date(2026, 11, 27))) == date(2026, 11, 26)


def _seed_unlabelled(n_frames: int = 3) -> str:
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    onto = get_ontology()

    async def _go():
        sid, ts = uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id=f"AP-{sid.hex[:6]}", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(10), city="BLR", sensors={}, ontology_version=onto.version))
            await db.flush()
            for k in range(n_frames):
                db.add(Frame(frame_id=uuid.uuid4(), session_id=sid, ts_ns=ts + k, cam_id="cam_f",
                             img_uri="s3://x/a.jpg", width=1920, height=1080, quality=0.9, scene={}))
            await db.commit()
        return str(sid)

    return _run(_go())


async def _reset_autopilot():
    from sqlalchemy import delete, update

    from db.models import AutolabelJob, AutopilotConfig, AutopilotNight
    from db.session import get_sessionmaker
    from services.agent.autopilot import get_config

    async with get_sessionmaker()() as db:
        await db.execute(delete(AutopilotNight))
        # Earlier tests' unlabelled sessions would otherwise be this test's candidates too.
        await get_config(db)
        await db.execute(update(AutopilotConfig).values(enabled=False, killed=False, killed_reason=None,
                                                        gpu_minutes=240, max_sessions=20, vlm_max_checks=300,
                                                        earliest_start_ist="20:00",
                                                        steps={"autolabel": True, "vlm_check": True,
                                                               "error_sweep": True, "auto_accept": True}))
        await db.execute(update(AutolabelJob).where(AutolabelJob.status.in_(("pending", "running")))
                         .values(status="canceled"))
        from db.models import GovernanceState
        from services.govern.killswitch import get_state

        await get_state(db)
        await db.execute(update(GovernanceState).values(loop_enabled=True, auto_accept_enabled=True))
        # Sessions an earlier test left unlabelled are marked done, so each test plans only its own.
        from services.agent.autopilot import candidate_sessions

        for c in await candidate_sessions(db):
            db.add(AutolabelJob(session_id=uuid.UUID(c["session_id"]), status="done", params={}))
        await db.commit()


def test_the_scheduler_starts_a_night_once_and_only_when_switched_on():
    from sqlalchemy import select, update

    from db.models import AutopilotConfig, AutopilotNight, ToolJob
    from db.session import get_sessionmaker
    from services.agent.autopilot import maybe_autopilot

    _run(_reset_autopilot())
    sid = _seed_unlabelled()

    async def _go():
        async with get_sessionmaker()() as db:
            off = await maybe_autopilot(db, now=_ist(21))
            await db.execute(update(AutopilotConfig).values(enabled=True))
            await db.commit()
            quiet = await maybe_autopilot(db, now=_ist(11))
            first = await maybe_autopilot(db, now=_ist(21))
            again = await maybe_autopilot(db, now=_ist(2, d=date(2026, 11, 27)))
            if not first.get("ran"):
                return off, quiet, first, again, None, None
            night = await db.get(AutopilotNight, uuid.UUID(first["night_id"]))
            job = (await db.execute(select(ToolJob).where(ToolJob.job_id == night.tool_job_id))).scalar_one()
            return off, quiet, first, again, night, job

    off, quiet, first, again, night, job = _run(_go())
    assert off == {"ran": False, "reason": "switched off"}
    assert not quiet["ran"] and "office hours" in quiet["reason"]
    assert first["ran"], first
    assert again == {"ran": False, "reason": "already ran tonight"}
    assert night.status == "planned" and night.night_date == date(2026, 11, 26)
    assert sid in {s["session_id"] for s in night.plan["sessions"]["would_label"]}
    assert job.kind == "autopilot_night" and job.gpu and job.status == "pending"


def test_a_dry_run_reports_and_changes_nothing():
    from _authutil import auth_headers
    from sqlalchemy import func, select

    from db.models import AutolabelJob, Object
    from db.session import get_sessionmaker
    from services.api.main import app

    _run(_reset_autopilot())
    sid = _seed_unlabelled(5)

    async def _counts():
        async with get_sessionmaker()() as db:
            return ((await db.execute(select(func.count()).select_from(Object))).scalar_one(),
                    (await db.execute(select(func.count()).select_from(AutolabelJob))).scalar_one())

    before = _run(_counts())
    with TestClient(app) as c:
        assert c.post("/api/tools/agents/autopilot/dry-run", headers=auth_headers("annotator")).status_code == 403
        r = c.post("/api/tools/agents/autopilot/dry-run", headers=auth_headers("reviewer"))
        assert r.status_code == 200, r.text
        plan = r.json()["plan"]
        assert r.json()["dry_run"] and sid in {s["session_id"] for s in plan["sessions"]["would_label"]}
        assert "the autopilot is switched off" in plan["blockers"]
        assert any("auto-label" in a for a in plan["actions"])
        hist = c.get("/api/tools/agents/autopilot/nights", headers=auth_headers("annotator")).json()["nights"]
        assert hist[0]["dry_run"] and hist[0]["summary"]["would_label"] >= 1
        # Settings are an admin's; a reviewer can pull the kill switch.
        assert c.put("/api/tools/agents/autopilot/settings", json={"enabled": True},
                     headers=auth_headers("reviewer")).status_code == 403
        r = c.put("/api/tools/agents/autopilot/settings", json={"gpu_minutes": 2}, headers=auth_headers("admin"))
        assert r.status_code == 422 and "between" in r.json()["detail"]
        r = c.put("/api/tools/agents/autopilot/settings", json={"gpu_minutes": 30, "steps": {"vlm_check": False}},
                  headers=auth_headers("admin"))
        assert r.status_code == 200 and r.json()["gpu_minutes"] == 30 and r.json()["steps"]["vlm_check"] is False
    assert _run(_counts()) == before


def test_a_night_keeps_only_what_the_ladder_and_the_checks_allow():
    from sqlalchemy import select

    from db.models import AgentRun, AutolabelJob, AutopilotNight, Frame, Object
    from db.session import get_sessionmaker
    from services.agent.autopilot import run_night, start_night
    from services.agent.runs import revert_run
    from services.autolabel.paths.path_c_vlm import VlmResult
    from services.govern.class_autonomy import set_level

    _run(_reset_autopilot())
    onto = get_ontology()
    sid = _seed_unlabelled(2)
    bus, car, truck = onto.by_name("bus").id, onto.by_name("sedan").id, onto.by_name("truck").id
    made: dict[str, list[str]] = {"bus": [], "sedan": [], "truck": [], "review": []}

    async def fake_autolabel(job_id):
        # Stands in for the detector: what the gate wrote is what matters to the night's later steps.
        async with get_sessionmaker()() as db:
            job = await db.get(AutolabelJob, job_id)
            frames = (await db.execute(select(Frame).where(Frame.session_id == job.session_id))).scalars().all()
            for f in frames:
                for k, (cid, key, state) in enumerate(((bus, "bus", "auto_accept"), (car, "sedan", "auto_accept"),
                                                       (truck, "truck", "auto_accept"), (bus, "review", "review"))):
                    x = 10.0 + 300 * k
                    o = Object(object_id=uuid.uuid4(), frame_id=f.frame_id, class_id=cid, bbox=[x, 400.0, x + 200, 600.0],
                               conf=0.9, source="auto_accept" if state == "auto_accept" else "fused", state=state,
                               attrs={}, provenance={}, version=1)
                    db.add(o)
                    made[key].append(str(o.object_id))
            job.status, job.counts = "done", {"frames": len(frames), "objects": 4 * len(frames)}
            await db.commit()

    class FakeVlm:
        """Agrees with buses, thinks the sedans are something else; never sees trucks (the ladder stops them)."""

        def verify_object(self, img, bbox, class_id):
            return VlmResult(class_name="bus" if class_id == bus else "hatchback", confident=True)

    import services.recall.backends as backends

    orig = backends.load_image_bgr
    backends.load_image_bgr = lambda store, uri: object()
    try:
        async def _go():
            async with get_sessionmaker()() as db:
                for cid in (bus, car):
                    await set_level(db, cid, 1, set_by="test")
                await set_level(db, truck, 0, set_by="test")
                night = await start_night(db, trigger="manual", now=_ist(21))
            stages = []

            async def progress(frac, stage=None):
                stages.append(stage)

            out = await run_night(night.night_id, progress, autolabel_fn=fake_autolabel, verifier=FakeVlm())
            async with get_sessionmaker()() as db:
                n = await db.get(AutopilotNight, night.night_id)
                run = await db.get(AgentRun, n.agent_run_id)
                return out, n, run, stages

        out, night, run, stages = _run(_go())
    finally:
        backends.load_image_bgr = orig

    assert out["status"] == "done" and out["labelled_sessions"] == 1
    st = _run(_state(made["bus"] + made["sedan"] + made["truck"] + made["review"]))
    assert all(st[i][1] == "auto_accept" for i in made["bus"]), "ladder allows it and the VLM agreed"
    assert all(st[i][1] == "review" for i in made["sedan"]), "the VLM disagreed"
    assert all(st[i][1] == "review" for i in made["truck"]), "the ladder keeps trucks propose-only"
    rep = night.report
    assert rep["auto_accepted"]["by_class"] == {"bus": 2}
    assert rep["sent_to_review"]["from_auto_accept"] == 4
    assert rep["sent_to_review"]["reasons"] == {"VLM disagreed": 2, "ladder: this class is propose-only": 2}
    assert rep["vlm"]["checked"] == 4 and rep["vlm"]["disagreed"] == 2
    assert sid in {s["session_id"] for s in rep["sessions"]} and rep["notes"]
    assert not rep["error_sweep"]["errors"], rep["error_sweep"]
    assert run.status == "committed" and len(run.changes) == 4

    async def _revert():
        async with get_sessionmaker()() as db:
            return await revert_run(db, run.run_id)

    assert _run(_revert())["reverted"] == 4
    st = _run(_state(made["sedan"] + made["truck"]))
    assert all(v[1] == "auto_accept" for v in st.values())


def test_without_a_vlm_nothing_is_kept_unchecked_and_the_kill_switch_stops_it(monkeypatch):
    from _authutil import auth_headers
    from sqlalchemy import select, update

    from db.models import AutolabelJob, AutopilotConfig, Frame, Object, ToolJob
    from db.session import get_sessionmaker
    from services.agent.autopilot import maybe_autopilot, run_night, start_night
    from services.api.main import app
    from services.govern.class_autonomy import set_level

    _run(_reset_autopilot())
    onto = get_ontology()
    _seed_unlabelled(1)
    bus = onto.by_name("bus").id
    made: list[str] = []

    async def fake_autolabel(job_id):
        async with get_sessionmaker()() as db:
            job = await db.get(AutolabelJob, job_id)
            f = (await db.execute(select(Frame).where(Frame.session_id == job.session_id))).scalars().first()
            o = Object(object_id=uuid.uuid4(), frame_id=f.frame_id, class_id=bus, bbox=[1.0, 1.0, 50.0, 50.0],
                       conf=0.9, source="auto_accept", state="auto_accept", attrs={}, provenance={}, version=1)
            db.add(o)
            made.append(str(o.object_id))
            job.status = "done"
            await db.commit()

    async def _go():
        async with get_sessionmaker()() as db:
            await set_level(db, bus, 1, set_by="test")
            night = await start_night(db, trigger="manual", now=_ist(21))

        async def progress(frac, stage=None):
            return None

        # VLM disabled in this suite's settings: the night cannot check, so it does not keep.
        return await run_night(night.night_id, progress, autolabel_fn=fake_autolabel)

    from core.config import get_settings

    monkeypatch.setattr(get_settings().models.vlm, "enabled", False)
    _run(_go())
    assert all(v[1] == "review" for v in _run(_state(made)).values())

    # The kill switch: a queued night is cancelled and the scheduler refuses until an admin re-arms it.
    async def _queue():
        from sqlalchemy import delete

        from db.models import AutopilotNight

        async with get_sessionmaker()() as db:
            await db.execute(delete(AutopilotNight))
            await db.execute(update(AutopilotConfig).values(enabled=True))
            await db.commit()
            return await maybe_autopilot(db, now=_ist(22))

    q = _run(_queue())
    assert q["ran"]
    with TestClient(app) as c:
        r = c.post("/api/tools/agents/autopilot/kill", json={"reason": "test"}, headers=auth_headers("reviewer"))
        assert r.status_code == 200 and q["night_id"] in r.json()["stopped_nights"]
        assert c.post("/api/tools/agents/autopilot/run-now", headers=auth_headers("admin")).status_code == 409
        st = c.get("/api/tools/agents/autopilot", headers=auth_headers("annotator")).json()
        assert st["settings"]["killed"] and st["settings"]["killed_reason"] == "test"

    async def _after():
        async with get_sessionmaker()() as db:
            job = await db.get(ToolJob, uuid.UUID(q["job_id"]))
            return job.status, await maybe_autopilot(db, now=_ist(23))

    status, again = _run(_after())
    assert status == "canceled" and again == {"ran": False, "reason": "kill switch on"}
    with TestClient(app) as c:
        assert c.post("/api/tools/agents/autopilot/rearm", headers=auth_headers("reviewer")).status_code == 403
        assert c.post("/api/tools/agents/autopilot/rearm", headers=auth_headers("admin")).json()["killed"] is False


# ---- quiet hours for the every-tick agents ------------------------------------------------------------------------

def test_every_tick_agents_skip_gpu_work_in_office_hours(monkeypatch):
    from db.session import get_sessionmaker
    from services.agent.runtime import schedule

    calls: list[str] = []

    async def embed(db):
        calls.append("embed")
        return {"ran": False}

    async def campaigns(db):
        calls.append("campaigns_full")
        return {"ran": False}

    async def quiet_campaigns(db):
        calls.append("campaigns_quiet")
        return {"ran": False}

    async def settle(db):
        calls.append("settle")
        return {"ran": False}

    async def autopilot(db, now=None):
        calls.append("autopilot")
        return {"ran": False}

    monkeypatch.setattr("services.intelligence.embed.daemon.maybe_embed_pending", embed)
    monkeypatch.setattr("services.flywheel.campaign.maybe_tick_campaigns", campaigns)
    monkeypatch.setattr(schedule, "_tick_campaigns_quiet", quiet_campaigns)
    monkeypatch.setattr("services.govern.settlement_agent.maybe_tally_and_settle", settle)
    monkeypatch.setattr("services.agent.autopilot.maybe_autopilot", autopilot)

    async def _go(now):
        async with get_sessionmaker()() as db:
            await schedule.run_due(db, offhours=False, now=now)

    _run(_go(_ist(11)))
    assert calls == ["campaigns_quiet", "settle"]
    calls.clear()
    _run(_go(_ist(21)))
    assert calls == ["embed", "campaigns_full", "settle", "autopilot"]


def test_office_hours_campaign_tick_defers_gpu_stages(monkeypatch):
    from db.models import Campaign
    from db.session import get_sessionmaker
    from services.agent.runtime.schedule import _tick_campaigns_quiet

    ran: list[str] = []
    names = {f"q-train-{uuid.uuid4().hex[:6]}": "train", f"q-label-{uuid.uuid4().hex[:6]}": "label"}

    async def tick(db, cid, dry_run=False):
        c = await db.get(Campaign, uuid.UUID(cid))
        return {"action": "would_run", "stage": names.get(c.name, "waiting")} if c.name in names else {"action": "waiting"}

    async def run_stage(db, cid, stage):
        ran.append(stage)
        return {"action": stage}

    monkeypatch.setattr("services.flywheel.campaign.tick", tick)
    monkeypatch.setattr("services.flywheel.campaign.run_stage", run_stage)

    async def _go():
        async with get_sessionmaker()() as db:
            for n in names:
                db.add(Campaign(name=n, class_name="bus", status="running"))
            await db.commit()
            return await _tick_campaigns_quiet(db)

    out = _run(_go())
    assert ran == ["label"] and [d["stage"] for d in out["deferred_gpu"]] == ["train"]
