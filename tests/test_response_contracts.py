"""Every route that declares a response model sends exactly what its handler returned.

The model on a route is a filter (tests/_contract.py says how that goes wrong), so each declared route has
a case here that calls it with data seeded to reach its branches, and the shared helper compares the body
sent with the handler's own return value on the same request, then checks every key against the model.

Three guards keep the set honest rather than merely green:
  * a route that declares an ApiOut model and has no case here fails `test_every_declared_route_has_a_case`;
  * a route that declares one and is not served with exclude_unset fails
    `test_declared_routes_send_only_the_keys_the_handler_set`;
  * a case that reaches a different route from the one it is filed under fails where it runs.

One world is seeded for the module and deleted afterwards. It has to be deleted: it holds pending and
running jobs, a blind audit, a judging settlement lot and unjudged control samples, each of which some
other test in the suite counts corpus-wide.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from _contract import Exchange, assert_nothing_lost, call, models_in
from fastapi.routing import APIRoute

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db

MASK = [[10.0, 10.0, 60.0, 10.0, 60.0, 50.0, 10.0, 50.0]]


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


@dataclass
class World:
    """Ids of everything seeded, and the rows to delete afterwards (in the order they were added)."""

    admin: str = ""
    other: str = ""
    ids: dict[str, str] = field(default_factory=dict)
    created: list[tuple[type, object]] = field(default_factory=list)

    def __getattr__(self, name: str) -> str:
        try:
            return self.ids[name]
        except KeyError:
            raise AttributeError(name) from None


async def _seed(w: World) -> None:
    from sqlalchemy import inspect as sa_inspect

    from core.storage import get_object_store
    from db.models import (
        AutolabelJob,
        BlindAudit,
        BlindAuditFrame,
        ControlSample,
        DatasetCommit,
        ExportJob,
        Frame,
        ImportJob,
        InferenceRun,
        LabelJob,
        LabelProject,
        LabelTask,
        MapFusionJob,
        ModelRegistry,
        ModelRun,
        Notification,
        Object,
        ProjectMember,
        RelabelJob,
        SettlementLot,
        Track,
        TrainingJob,
        UserCredential,
    )
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.api.routers.objects import _write_mask
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    ped = onto.by_name("pedestrian").id
    other_cls = next(c.id for c in sorted(onto.classes, key=lambda c: c.id) if c.id != ped)
    tag = uuid.uuid4().hex[:10]
    store = get_object_store()
    store.ensure_bucket()

    def new(key: str | None, row):
        """Queue a row, remember its primary key for teardown, and optionally name its id."""
        db.add(row)
        pending.append((key, row))
        return row

    async def flush():
        await db.flush()
        for key, row in pending:
            pk = sa_inspect(row).identity
            w.created.append((type(row), pk))
            if key:
                w.ids[key] = str(pk[0])
        pending.clear()

    pending: list = []
    async with get_sessionmaker()() as db:
        admin = uuid.UUID(w.admin)
        new(None, UserCredential(user_id=admin, email="contract@example.com", password_hash="not-a-real-hash",
                                 recovery_hashes=["a", "b", "c"]))
        project = new("project", LabelProject(name=f"contract-{tag}", description="seeded for the contract tests",
                                              visibility="members", class_ids=sorted([ped, other_cls]),
                                              created_by=admin))
        await flush()
        new(None, ProjectMember(project_id=project.project_id, user_id=admin, added_by=admin))

        # The newest session in the database by a wide margin, so it is in the home page's recent six.
        start = now_ns() + seconds_to_ns(50_000)
        sid = uuid.uuid4()
        new("session", DbSession(session_id=sid, vehicle_id=f"CONTRACT-{tag}", start_ts_ns=start,
                                 end_ts_ns=start + seconds_to_ns(3), city="BLR", route="ORR-east", sensors={},
                                 ontology_version=onto.version, project_id=project.project_id))
        await flush()
        f1, f2, f3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        new("f1", Frame(frame_id=f1, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri="s3://labeloxav/x.jpg",
                        width=640, height=480, quality=0.9))
        # A LiDAR frame with scene context somebody corrected: every nullable key of the frame meta is set.
        new("f2", Frame(frame_id=f2, session_id=sid, ts_ns=start + seconds_to_ns(1), cam_id="cam_f",
                        img_uri="s3://labeloxav/x.jpg", width=640, height=480, quality=0.9,
                        lidar={"n_points": 120000, "bev": {"res": 0.05}},
                        scene={"haze": "light", "weather_conf": 0.8},
                        scene_provenance={"haze": {"by": "somebody", "ts_ns": start}}))
        new("f3", Frame(frame_id=f3, session_id=sid, ts_ns=start + seconds_to_ns(2), cam_id="cam_f",
                        img_uri="s3://labeloxav/x.jpg", width=640, height=480, quality=0.9))
        track = new("track", Track(session_id=sid, class_id=ped, first_ts_ns=start, last_ts_ns=start))
        await flush()

        # f1: one object carrying every optional geometry there is, and one carrying none of it.
        full_id = uuid.uuid4()
        mask_uri = _write_mask(store, sid, f1, full_id, MASK, 640, 480)
        new("o_full", Object(object_id=full_id, frame_id=f1, track_id=track.track_id, class_id=ped,
                             bbox=[10.0, 10.0, 60.0, 50.0], bbox_amodal=[5.0, 5.0, 70.0, 60.0], mask_uri=mask_uri,
                             mask_encoding="polygon", attrs={}, conf=0.91, state="review", rot_deg=12.5,
                             quality_score=0.7, keypoints={"skeleton": "person", "points": [[20, 20, 2]]},
                             polyline=[[10.0, 10.0], [60.0, 50.0]],
                             cuboid_3d={"center": [1.0, 2.0, 3.0], "size": [1.0, 2.0, 3.0], "yaw": 0.1}))
        new("o_plain", Object(frame_id=f1, class_id=other_cls, bbox=[100, 100, 150, 160], conf=0.4, state="review"))
        new("o_masked", Object(frame_id=f1, class_id=ped, bbox=[200, 100, 250, 160], conf=0.6, state="review"))
        # f2: imported labels, so the frame names its annotation source and the format it came from.
        for _ in range(2):
            new(None, Object(frame_id=f2, class_id=ped, bbox=[1, 1, 10, 10], conf=1.0, state="accepted",
                             source="imported", provenance={"import_format": "mapillary"}))
        # f3: the objects the write routes act on.
        for key, state in (("o_b1", "review"), ("o_b2", "review"), ("o_stale", "review"), ("o_del", "review"),
                           ("o_ctrl", "auto_accept")):
            new(key, Object(frame_id=f3, class_id=ped, bbox=[1, 1, 30, 30], conf=0.5, state=state))
        await flush()

        # A blind audit over f3, served through a job nobody is assigned to: the audit applies only to a
        # request that names the job, so every other case sees f3 as an ordinary frame.
        model = new(None, ModelRegistry(model_version=f"contract-model-{tag}"))
        await flush()
        run = new(None, InferenceRun(model_version=model.model_version, status="complete"))
        task = new(None, LabelTask(project_id=project.project_id, name=f"contract-audit-{tag}", session_id=sid))
        await flush()
        job = new("audit_job", LabelJob(task_id=task.task_id, frame_ids=[str(f3), str(f1)]))
        await flush()
        audit = new("audit", BlindAudit(run_id=run.run_id, job_id=job.job_id, n_frames=1, status="seeded"))
        await flush()
        new(None, BlindAuditFrame(audit_id=audit.audit_id, frame_id=f3))
        new("o_audit", Object(frame_id=f3, class_id=ped, bbox=[40, 40, 90, 90], conf=1.0, state="accepted",
                              source="human", blind_audit_id=audit.audit_id))
        new(None, ControlSample(object_id=uuid.UUID(w.o_ctrl), was_auto_accepted=True))
        new("lot", SettlementLot(class_id=ped, model_epoch="contract", population=1000, tier="default",
                                 far_bound=0.05, sample_object_ids=[w.o_ctrl], batch_id=f"settle-{tag}",
                                 status="judging", created_by="test"))

        # One job of every kind the unified list knows, with the fields each kind's label and detail read.
        commit_id = f"contract-{tag}"
        new("import_done", ImportJob(status="done", format="coco", source_uri=f"s3://bucket/drops/drive_{tag}.zip",
                                     target_vehicle="IMPORT-01", city="BLR", progress=1.0,
                                     counts={"frames": 3, "objects": 9}, session_id=sid))
        new("import_error", ImportJob(status="error", format="video", target_vehicle="IMPORT-01", progress=0.3,
                                      counts={}, error="could not read that source"))
        new("import_pending", ImportJob(status="pending", format="coco", target_vehicle="IMPORT-01"))
        new("import_running", ImportJob(status="running", format="coco", target_vehicle="IMPORT-01", progress=0.5))
        run_id = f"contract-run-{tag}"
        new("training_done", TrainingJob(status="done", purpose=f"contract-{tag}", stage="train", progress=1.0,
                                         counts={"epoch": 3, "total_epochs": 10}, metrics={"map50": 0.41},
                                         result={"promoted": False}, config={"epochs": 10}, run_id=run_id))
        new("autolabel_done", AutolabelJob(session_id=sid, status="done", progress=1.0,
                                           counts={"n_frames": 3, "n_objects": 9}, params={"executor": "api"}))
        # Two in the worker's queue, the first one older, so the second has somebody ahead of it. The age is
        # written rather than left to the default: now() is the transaction's start, the same for both.
        new(None, AutolabelJob(session_id=sid, status="pending", params={"executor": "worker"},
                               created_at=datetime.now(UTC) - timedelta(minutes=5)))
        new("autolabel_queued", AutolabelJob(session_id=sid, status="pending", params={"executor": "worker"}))
        new(None, ExportJob(name=f"contract-{tag}", status="done", progress=1.0, commit_id=commit_id, object_count=9))
        new(None, MapFusionJob(status="done", region="BLR-ORR", session_ids=[str(sid)], stage="fuse", progress=1.0,
                               counts={"fused": 12}, commit_id="map-contract"))
        new(None, RelabelJob(status="done", model_version=f"contract-model-{tag}", session_ids=[str(sid)],
                             stage="apply", progress=1.0, result={"applied": 4}))

        present = store.put_bytes(f"exports/{commit_id}/coco/annotations.json", b"{}", "application/json")
        new("commit", DatasetCommit(commit_id=commit_id, slice_spec={"name": f"contract-{tag}",
                                                                      "formats": ["coco", "parquet"]},
                                    object_count=9, ontology_version=onto.version,
                                    export_uris={"coco/annotations.json": present,
                                                 "parquet/objects.parquet": f"s3://labeloxav/exports/{commit_id}/gone"}))
        # Two runs on one purpose: a promoted one with a metric the task does not report, and a later one.
        new("run", ModelRun(run_id=run_id, base_weights="yolo11n.pt", weights_uri="s3://labeloxav/w.pt",
                            dataset_name=f"contract-{tag}", n_train=40, n_val=10, epochs=10,
                            metrics={"map50": 0.41, "safe_miou": None}, baseline_metrics={"map50": 0.38},
                            gate={"promote": True, "reasons": []}, promoted=True, ontology_version=onto.version,
                            purpose=f"contract-{tag}", notes="seeded"))
        new(None, ModelRun(run_id=f"{run_id}-b", base_weights="yolo11n.pt", dataset_name=f"contract-{tag}-b",
                           ontology_version=onto.version, purpose=f"contract-{tag}"))
        await flush()

        new("note_mine", Notification(user_id=admin, kind="job_failed", severity="warn", title="An import failed",
                                      body="could not read that source", href="/jobs", subject_type="import_job",
                                      subject_id=w.import_error, meta={"format": "video"}))
        new("note_role", Notification(role="admin", kind="gate_blocked", title="A promotion was blocked"))
        new("note_theirs", Notification(role="annotator", kind="assigned", title="Not for an admin"))
        await flush()
        await db.commit()
    w.ids.update(tag=tag, commit=commit_id, ped=str(ped))


async def _unseed(w: World) -> None:
    """Delete what was seeded, newest first, plus what the cases themselves created on top of it, and
    refuse to finish quietly if any of it is still there."""
    from sqlalchemy import delete, func, select, tuple_

    from db.models import AgentRun, LabelProject, NotificationRead, ProjectMember, Review
    from db.session import get_sessionmaker

    def ids_of(table: str) -> list:
        return [pk[0] for model, pk in w.created if model.__tablename__ == table]

    async with get_sessionmaker()() as db:
        for run_id in w.ids.get("batch_runs", "").split():
            await db.execute(delete(AgentRun).where(AgentRun.run_id == uuid.UUID(run_id)))
        made = [uuid.UUID(p) for p in w.ids.get("made_projects", "").split()]
        await db.execute(delete(ProjectMember).where(ProjectMember.project_id.in_(made)))
        await db.execute(delete(LabelProject).where(LabelProject.project_id.in_(made)))
        await db.execute(delete(Review).where(Review.object_id.in_(ids_of("object"))))
        await db.execute(delete(NotificationRead).where(
            NotificationRead.notification_id.in_(ids_of("notification"))))
        await db.commit()
        for model, pk in reversed(w.created):
            cols = list(model.__table__.primary_key.columns)
            await db.execute(delete(model).where(tuple_(*cols) == pk))
        await db.commit()
        left = []
        for model, pk in w.created:
            cols = list(model.__table__.primary_key.columns)
            if (await db.execute(select(func.count()).select_from(model).where(tuple_(*cols) == pk))).scalar_one():
                left.append(f"{model.__tablename__} {pk}")
        assert not left, f"the contract tests left rows behind that other tests count: {left}"


@pytest.fixture(scope="module")
def world():
    from _authutil import seed_user

    w = World(admin=seed_user("admin"), other=seed_user("annotator"))
    try:
        run_async(_seed(w))
        yield w
    finally:
        run_async(_unseed(w))


@pytest.fixture
def client(world):
    """An admin, with the routes mounted and the lifespan not run: startup recovers interrupted jobs, and
    this module seeds pending ones on purpose."""
    from _authutil import headers_for
    from fastapi.testclient import TestClient

    from services.api.main import app

    _clear_db_cache()
    c = TestClient(app)
    c.headers.update(headers_for(world.admin))
    return c


def _remember(w: World, key: str, value: str) -> None:
    w.ids[key] = (w.ids.get(key, "") + " " + value).strip()


# ---------------------------------------------------------------------------------------------- the cases
#
# One function per operation, filed under the key the generated clients use for it. Each returns the
# exchanges it made; `test_the_response_is_what_the_handler_returned` checks every one of them. A case that
# needs to show a particular branch was reached asserts that itself, on the exchange's body.

CASES: dict = {}


def case(operation: str, *, empty_ok: tuple[str, ...] = ()):
    def register(fn):
        assert operation not in CASES, operation
        CASES[operation] = (fn, empty_ok)
        return fn
    return register


# ---- what the installation is

@case("GET /api/platforms")
def _platforms(c, w):
    return [call(c, "GET", "/api/platforms")]


@case("GET /api/system/features")
def _features(c, w):
    return [call(c, "GET", "/api/system/features")]


@case("GET /api/system/checks")
def _checks(c, w):
    return [call(c, "GET", "/api/system/checks")]


@case("GET /api/system/about")
def _about(c, w):
    return [call(c, "GET", "/api/system/about")]


# ---- ontology

@case("GET /api/ontology")
def _ontology(c, w):
    whole = call(c, "GET", "/api/ontology")
    scoped = call(c, "GET", f"/api/ontology?frame_id={w.f1}")
    assert whole.raw["active_class_ids"] is None
    # The project's two classes plus the ones every project carries (ignore regions, custom classes).
    assert int(w.ped) in scoped.raw["active_class_ids"] and scoped.raw["class_set"] == "custom"
    assert len(scoped.raw["active_class_ids"]) < len(scoped.raw["classes"])
    assert any(cls["aliases"] for cls in whole.raw["classes"]), "no class with aliases: the field went unchecked"
    assert any(a["range"] for a in whole.raw["attributes"].values()), "no ranged attribute was served"
    return [whole, scoped]


@case("GET /api/ontology/templates")
def _templates(c, w):
    ex = call(c, "GET", "/api/ontology/templates")
    assert {t["all"] for t in ex.raw["templates"]} == {True, False}
    return [ex]


@case("POST /api/ontology/classes")
def _create_class(c, w):
    # A name that exists: the route answers with the existing class and writes nothing. Both branches of
    # add_custom_class return the same six keys, and this one leaves the ontology sidecar alone.
    ex = call(c, "POST", "/api/ontology/classes", json={"name": "pedestrian"})
    assert ex.raw["existed"] is True
    return [ex]


# ---- sessions

@case("GET /api/sessions")
def _sessions(c, w):
    ex = call(c, "GET", "/api/sessions")
    mine = next(s for s in ex.raw if s["session_id"] == w.session)
    assert mine["project_id"] == w.project and mine["route"] == "ORR-east" and mine["origin"] == "other"
    return [ex]


@case("GET /api/sessions/page")
def _sessions_page(c, w):
    ex = call(c, "GET", f"/api/sessions/page?vehicle_id=CONTRACT-{w.tag}&limit=5")
    assert ex.raw["total"] == 1
    return [ex]


@case("GET /api/sessions/states")
def _session_states(c, w):
    ex = call(c, "GET", "/api/sessions/states")
    assert next(s for s in ex.raw if s["session_id"] == w.session)["frames"] == 3
    return [ex]


@case("GET /api/sessions/{session_id}/stats")
def _session_stats(c, w):
    ex = call(c, "GET", f"/api/sessions/{w.session}/stats")
    assert ex.raw["frames"] == 3 and ex.raw["by_state"]
    return [ex]


@case("GET /api/sessions/{session_id}/first-frame")
def _first_frame(c, w):
    ex = call(c, "GET", f"/api/sessions/{w.session}/first-frame")
    assert ex.raw == {"frame_id": w.f1}
    return [ex]


# ---- home, find, the inbox

@case("GET /api/home")
def _home(c, w):
    ex = call(c, "GET", "/api/home")
    body = ex.raw
    # Three kinds of row share `waiting_for_people` with no field in common but `kind`; all three have to be
    # in the response for the model's optional fields to have been compared with anything.
    assert {r["kind"] for r in body["waiting_for_people"]} == {"blind_audit", "control_samples", "settlement_lot"}
    assert any(f["job_id"] == w.import_error for f in body["jobs"]["failed"])
    assert any(s["session_id"] == w.session for s in body["recent_sessions"])
    return [ex]


@case("GET /api/find")
def _find(c, w):
    by_name = call(c, "GET", f"/api/find?q=CONTRACT-{w.tag}")
    assert [r["kind"] for r in by_name.raw["results"]] == ["session"]
    out = [by_name]
    for kind, ident in (("frame", w.f1), ("object", w.o_full), ("track", w.track)):
        ex = call(c, "GET", f"/api/find?q={ident[:13]}")
        assert kind in {r["kind"] for r in ex.raw["results"]}, kind
        out.append(ex)
    return out


@case("GET /api/notifications")
def _notifications(c, w):
    ex = call(c, "GET", "/api/notifications")
    ids = {n["notification_id"] for n in ex.raw["notifications"]}
    assert {w.note_mine, w.note_role} <= ids and w.note_theirs not in ids
    return [ex, call(c, "GET", "/api/notifications?unread_only=true&limit=1")]


@case("GET /api/notifications/count")
def _notification_count(c, w):
    ex = call(c, "GET", "/api/notifications/count")
    assert ex.raw["unread"] >= 1
    return [ex]


@case("POST /api/notifications/{notification_id}/read")
def _notification_read(c, w):
    # The route's three answers: marked, no such notification, and one addressed to somebody else.
    marked = call(c, "POST", f"/api/notifications/{w.note_mine}/read")
    missing = call(c, "POST", f"/api/notifications/{uuid.uuid4()}/read")
    theirs = call(c, "POST", f"/api/notifications/{w.note_theirs}/read")
    assert marked.raw == {"marked": True, "notification_id": w.note_mine}
    assert missing.raw == {"marked": False, "reason": "not found"}
    assert theirs.raw == {"marked": False, "reason": "not addressed to you"}
    return [marked, missing, theirs]


@case("POST /api/notifications/read-all")
def _notification_read_all(c, w):
    ex = call(c, "POST", "/api/notifications/read-all")
    assert isinstance(ex.raw["marked"], int) and not isinstance(ex.raw["marked"], bool)
    return [ex]


# ---- people

@case("GET /api/users")
def _users(c, w):
    return [call(c, "GET", "/api/users?limit=5")]


@case("GET /api/users/me")
def _whoami(c, w):
    ex = call(c, "GET", "/api/users/me")
    assert ex.raw["user_id"] == w.admin and ex.raw["role"] == "admin"
    return [ex]


@case("GET /api/auth/methods")
def _auth_methods(c, w):
    return [call(c, "GET", "/api/auth/methods")]


@case("GET /api/auth/profile")
def _profile(c, w):
    ex = call(c, "GET", "/api/auth/profile")
    assert ex.raw["email"] == "contract@example.com" and ex.raw["recovery_codes_left"] == 3
    return [ex]


# ---- projects

@case("GET /api/projects")
def _projects(c, w):
    ex = call(c, "GET", "/api/projects")
    mine = next(p for p in ex.raw["projects"] if p["project_id"] == w.project)
    assert mine["is_member"] and mine["members"] == 1 and mine["sessions"] == 1 and mine["class_set"] == "custom"
    return [ex]


def _made_project(c, w) -> str:
    """A project created through the API for the cases that change one, made once and deleted with the world."""
    if "scratch_project" not in w.ids:
        r = c.post("/api/projects", json={"name": f"contract-scratch-{w.tag}", "description": "to be changed"})
        assert r.status_code == 200, r.text
        w.ids["scratch_project"] = r.json()["project_id"]
        _remember(w, "made_projects", w.ids["scratch_project"])
    return w.ids["scratch_project"]


@case("POST /api/projects")
def _create_project(c, w):
    ex = call(c, "POST", "/api/projects", json={"name": f"contract-made-{w.tag}", "visibility": "members"})
    _remember(w, "made_projects", ex.raw["project_id"])
    return [ex]


@case("PATCH /api/projects/{project_id}")
def _change_project(c, w):
    pid = _made_project(c, w)
    return [call(c, "PATCH", f"/api/projects/{pid}", json={"visibility": "members", "description": "changed"}),
            call(c, "PATCH", f"/api/projects/{pid}", json={})]


@case("GET /api/projects/{project_id}/members")
def _members(c, w):
    ex = call(c, "GET", f"/api/projects/{w.project}/members")
    assert [m["user_id"] for m in ex.raw["members"]] == [w.admin]
    return [ex]


@case("PUT /api/projects/{project_id}/members/{user_id}")
def _add_member(c, w):
    pid = _made_project(c, w)
    return [call(c, "PUT", f"/api/projects/{pid}/members/{w.other}")]


@case("DELETE /api/projects/{project_id}/members/{user_id}")
def _remove_member(c, w):
    pid = _made_project(c, w)
    c.put(f"/api/projects/{pid}/members/{w.other}")
    ex = call(c, "DELETE", f"/api/projects/{pid}/members/{w.other}")
    assert ex.raw["member"] is False
    return [ex]


@case("PUT /api/projects/{project_id}/classes")
def _set_classes(c, w):
    pid = _made_project(c, w)
    generic = call(c, "PUT", f"/api/projects/{pid}/classes", json={"template": "generic"})
    custom = call(c, "PUT", f"/api/projects/{pid}/classes", json={"class_ids": [int(w.ped)]})
    everything = call(c, "PUT", f"/api/projects/{pid}/classes", json={"template": "india"})
    assert generic.raw["class_set"] == "generic" and custom.raw["class_set"] == "custom"
    assert everything.raw["class_ids"] is None, "the whole-ontology template is a null list, not an empty one"
    return [generic, custom, everything]


@case("POST /api/projects/{project_id}/sessions")
def _move_sessions(c, w):
    # Moved into the project it is already in: the answer has the same shape and nothing else in the
    # module finds the session somewhere new.
    ex = call(c, "POST", f"/api/projects/{w.project}/sessions", json={"session_ids": [w.session]})
    assert ex.raw == {"project_id": w.project, "moved": 1}
    return [ex]


# ---- the editor

@case("GET /api/frames/{frame_id}")
def _frame(c, w):
    plain = call(c, "GET", f"/api/frames/{w.f1}")
    lidar = call(c, "GET", f"/api/frames/{w.f2}")
    audited = call(c, "GET", f"/api/frames/{w.f3}?job_id={w.audit_job}")
    assert plain.raw["prev_frame_id"] is None and plain.raw["next_frame_id"] == w.f2
    assert lidar.raw["is_lidar"] and lidar.raw["lidar_points"] == 120000 and lidar.raw["lidar_res"] == 0.05
    assert lidar.raw["annotation_source"] == "imported" and lidar.raw["import_format"] == "mapillary"
    assert lidar.raw["context"]["haze"] == "light" and lidar.raw["context_provenance"]["haze"]["by"] == "somebody"
    # Under the audit the count is the auditor's own boxes, the source is withheld, and next follows the job.
    assert audited.raw["blind_audit_id"] == w.audit and audited.raw["annotation_source"] is None
    assert audited.raw["n_objects"] == 1 and audited.raw["next_frame_id"] == w.f1
    # The point count is stored as an integer and has to arrive as one, not as 120000.0.
    assert '"lidar_points":120000,' in lidar.response.text.replace(" ", "")
    return [plain, lidar, audited]


@case("GET /api/frames/{frame_id}/objects")
def _frame_objects(c, w):
    ex = call(c, "GET", f"/api/frames/{w.f1}/objects")
    by_id = {o["object_id"]: o for o in ex.raw}
    full, plain = by_id[w.o_full], by_id[w.o_plain]
    assert full["mask_polygons"] and full["keypoints"] and full["polyline"] and full["cuboid_3d"]
    assert full["bbox_amodal"] and full["track_id"] == w.track and full["quality_score"] == 0.7
    assert plain["mask_polygons"] == [] and plain["keypoints"] is None and plain["track_id"] is None
    audited = call(c, "GET", f"/api/frames/{w.f3}/objects?job_id={w.audit_job}")
    assert [o["object_id"] for o in audited.raw] == [w.o_audit]
    return [ex, audited]


@case("GET /api/frames/{frame_id}/filmstrip")
def _filmstrip(c, w):
    ex = call(c, "GET", f"/api/frames/{w.f2}/filmstrip")
    assert [t["frame_id"] for t in ex.raw["frames"]] == [w.f1, w.f2, w.f3]
    assert [t["current"] for t in ex.raw["frames"]] == [False, True, False]
    assert ex.raw["frames"][1]["n_confirmed"] == 2
    return [ex]


@case("PATCH /api/frames/{frame_id}/context")
def _frame_context(c, w):
    ex = call(c, "PATCH", f"/api/frames/{w.f2}/context", json={"attrs": {"dust": True}})
    assert ex.raw["context"]["dust"] is True and ex.raw["context"]["haze"] == "light"
    assert set(ex.raw["context_provenance"]) == {"haze", "dust"}
    return [ex]


@case("PUT /api/objects/{object_id}/mask")
def _mask(c, w):
    ex = call(c, "PUT", f"/api/objects/{w.o_masked}/mask", json={"polygons": MASK})
    assert ex.raw["mask_polygons"] and ex.raw["version"] == 2
    return [ex]


@case("POST /api/objects/{object_id}/review")
def _review(c, w):
    confirm = call(c, "POST", f"/api/objects/{w.o_plain}/review", json={"action": "confirm"})
    # Geometry, a keypoint skeleton and an amodal box in one write, on the object that already has a mask.
    adjust = call(c, "POST", f"/api/objects/{w.o_full}/review", json={
        "action": "adjust_geometry", "bbox": [12, 12, 62, 52], "rot_deg": 5.0, "bbox_amodal": [4, 4, 72, 62],
        "keypoints": {"skeleton": "person", "points": [[22, 22, 2]]}})
    assert confirm.raw["keypoints"] is None and confirm.raw["bbox_amodal"] is None
    assert adjust.raw["keypoints"] and adjust.raw["bbox_amodal"] == [4, 4, 72, 62] and adjust.raw["rot_deg"] == 5.0
    # Not an ObjectDetail, whatever the web client's type has said: no frame, no image, no mask.
    assert not {"frame_id", "image_url", "mask_polygons", "conf", "provenance"} & set(adjust.raw)
    return [confirm, adjust]


# The second request below changes nothing, so its list of stale objects is empty on purpose; the first
# request is the one that fills it, and asserts that it did.
@case("POST /api/objects/bulk-review", empty_ok=("$.skipped_stale",))
def _bulk_review(c, w):
    gone = str(uuid.uuid4())
    ex = call(c, "POST", "/api/objects/bulk-review", json={
        "object_ids": [w.o_b1, w.o_b2, w.o_stale, gone], "action": "confirm",
        "expected_versions": {w.o_stale: 99}})
    assert ex.raw["updated"] == 2 and ex.raw["skipped_missing"] == [gone]
    assert ex.raw["skipped_stale"] == [{"object_id": w.o_stale, "expected": 99, "current": 1}]
    assert ex.raw["run_id"], "a batch that changed something has a handle to revert it"
    _remember(w, "batch_runs", ex.raw["run_id"])
    # A batch that changes nothing has no run to revert, and says so with a null rather than a missing key.
    nothing = call(c, "POST", "/api/objects/bulk-review", json={"object_ids": [gone], "action": "confirm"})
    assert nothing.raw["run_id"] is None and nothing.raw["updated"] == 0
    return [ex, nothing]


@case("DELETE /api/objects/{object_id}")
def _delete_object(c, w):
    ex = call(c, "DELETE", f"/api/objects/{w.o_del}")
    assert ex.raw == {"deleted": w.o_del}
    return [ex]


# ---- jobs

@case("GET /api/jobs")
def _jobs(c, w):
    ex = call(c, "GET", "/api/jobs?limit=1000")
    kinds = {j["kind"] for j in ex.raw}
    assert kinds >= {"import", "training", "autolabel", "export", "map_fusion", "relabel"}, kinds
    done = next(j for j in ex.raw if j["job_id"] == w.import_done)
    assert done["label"] == f"drive_{w.tag}.zip" and done["detail"] == "3fr / 9obj"
    assert next(j for j in ex.raw if j["job_id"] == w.training_done)["detail"] == "ep 3/10"
    return [ex]


@case("POST /api/jobs/{kind}/{job_id}/cancel")
def _cancel_job(c, w):
    # Four answers in three shapes. `status` and `stopped` are the optional fields, and each has to be both
    # present and absent somewhere in here.
    queued = call(c, "POST", f"/api/jobs/import/{w.import_pending}/cancel")
    live = call(c, "POST", f"/api/jobs/import/{w.import_running}/cancel")
    finished = call(c, "POST", f"/api/jobs/import/{w.import_done}/cancel")
    missing = call(c, "POST", f"/api/jobs/import/{uuid.uuid4()}/cancel")
    assert queued.raw["stopped"] is True and live.raw["stopped"] is False
    assert set(finished.raw) == {"job_id", "canceled", "status", "detail"} and finished.raw["canceled"] is False
    assert set(missing.raw) == {"job_id", "canceled", "detail"}
    return [queued, live, finished, missing]


@case("GET /api/imports")
def _imports(c, w):
    ex = call(c, "GET", "/api/imports?limit=1000")
    assert next(j for j in ex.raw if j["job_id"] == w.import_done)["session_id"] == w.session
    assert next(j for j in ex.raw if j["job_id"] == w.import_error)["error"]
    return [ex]


@case("GET /api/imports/{job_id}")
def _import_status(c, w):
    return [call(c, "GET", f"/api/imports/{w.import_done}"), call(c, "GET", f"/api/imports/{w.import_error}")]


@case("GET /api/imports/formats")
def _import_formats(c, w):
    ex = call(c, "GET", "/api/imports/formats")
    assert ex.raw["annotated"] and ex.raw["raw"]
    return [ex]


@case("GET /api/autolabel/{job_id}")
def _autolabel_status(c, w):
    done = call(c, "GET", f"/api/autolabel/{w.autolabel_done}")
    queued = call(c, "GET", f"/api/autolabel/{w.autolabel_queued}")
    # Null when the job is not in the worker's queue, a count when it is; the key is there either way.
    assert done.raw["queue_position"] is None and done.raw["executor"] == "api"
    assert queued.raw["queue_position"] >= 1 and queued.raw["executor"] == "worker"
    return [done, queued]


@case("GET /api/training")
def _training(c, w):
    ex = call(c, "GET", "/api/training?limit=1000")
    assert next(j for j in ex.raw if j["job_id"] == w.training_done)["counts"]["total_epochs"] == 10
    return [ex]


@case("GET /api/training/{job_id}")
def _training_status(c, w):
    return [call(c, "GET", f"/api/training/{w.training_done}")]


@case("GET /api/training/tasks")
def _training_tasks(c, w):
    return [call(c, "GET", "/api/training/tasks")]


@case("GET /api/training/registry")
def _registry(c, w):
    ex = call(c, "GET", "/api/training/registry")
    line = next(m for m in ex.raw if m["purpose"] == f"contract-{w.tag}")
    assert len(line["runs"]) == 2 and line["promoted"]["run_id"] == w.run
    assert line["promoted"]["map50"] == 0.41 and line["promoted"]["safe_miou"] is None
    return [ex]


@case("GET /api/training/runs")
def _runs(c, w):
    ex = call(c, "GET", "/api/training/runs?limit=500")
    mine = next(r for r in ex.raw if r["run_id"] == w.run)
    assert mine["gate"] == {"promote": True, "reasons": []} and mine["notes"] == "seeded"
    return [ex]


# ---- datasets

@case("GET /api/datasets")
def _datasets(c, w):
    ex = call(c, "GET", "/api/datasets?limit=2000")
    mine = next(d for d in ex.raw if d["commit_id"] == w.commit)
    assert mine["n_files"] == 2 and mine["formats"] == ["coco", "parquet"]
    return [ex]


@case("GET /api/datasets/{commit_id}")
def _dataset_detail(c, w):
    ex = call(c, "GET", f"/api/datasets/{w.commit}")
    files = {f["path"]: f for f in ex.raw["files"]}
    assert files["coco/annotations.json"]["url"] and files["coco/annotations.json"]["missing"] is False
    assert files["parquet/objects.parquet"] == {"path": "parquet/objects.parquet", "url": None, "missing": True}
    assert "n_files" not in ex.raw, "the detail carries the files, not a count of them"
    return [ex]


# ---------------------------------------------------------------------------------------------- the tests

def _declared_routes() -> dict[str, APIRoute]:
    """Every operation whose response model is, or contains, an ApiOut."""
    from services.api.deps import ApiOut
    from services.api.main import app

    out: dict[str, APIRoute] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.response_model is None:
            continue
        if any(issubclass(m, ApiOut) for m in models_in(route.response_model)):
            for method in sorted(route.methods or ()):
                out[f"{method} {route.path}"] = route
    return out


@pytest.mark.parametrize("operation", sorted(CASES))
def test_the_response_is_what_the_handler_returned(operation, client, world):
    fn, empty_ok = CASES[operation]
    exchanges: list[Exchange] = fn(client, world)
    assert exchanges, f"{operation}: the case made no request"
    for ex in exchanges:
        assert ex.operation == operation, f"filed under {operation} but reached {ex.operation}"
        assert_nothing_lost(ex, empty_ok=empty_ok)


def test_every_declared_route_has_a_case():
    declared = _declared_routes()
    # A floor, so a scan that stops matching cannot report an empty difference as success.
    assert len(declared) >= 50, f"only {len(declared)} routes declare an ApiOut; the scan has stopped matching"
    missing = sorted(set(declared) - set(CASES))
    assert not missing, (
        "these routes declare a response model and nothing here proves the model loses no field. Add a "
        f"case that calls each with representative data, or do not declare it: {missing}")
    stale = sorted(set(CASES) - set(declared))
    assert not stale, f"cases for routes that no longer declare an ApiOut model: {stale}"


def test_declared_routes_send_only_the_keys_the_handler_set():
    """Without exclude_unset every optional field a handler left out is sent as an explicit null, which is
    a change to the endpoint: `"stopped" in answer` was false and becomes true."""
    loose = sorted(op for op, route in _declared_routes().items() if not route.response_model_exclude_unset)
    assert not loose, f"declare these with response_model_exclude_unset=True: {loose}"


def test_declared_models_publish_no_open_ended_schema():
    """ApiOut lets undeclared keys through at runtime and must not advertise that: an open object becomes
    an index signature in the TypeScript types, and a misspelt field then reads as `unknown` instead of
    failing to compile."""
    from services.api.deps import ApiOut

    models: set = set()
    for route in _declared_routes().values():
        models |= {m for m in models_in(route.response_model) if issubclass(m, ApiOut)}
    assert len(models) >= 50
    open_ended = sorted(m.__name__ for m in models if "additionalProperties" in m.model_json_schema())
    assert not open_ended, open_ended
    # And the half the schema hides: a key the model does not name survives validation and serialisation.
    sample = next(m for m in models if m.__name__ == "FirstFrameOut")
    kept = sample.model_validate({"frame_id": "f", "added_later": 1}).model_dump(mode="json", exclude_unset=True)
    assert kept == {"frame_id": "f", "added_later": 1}


def test_the_helper_catches_a_dropped_key_an_added_null_and_an_undeclared_key():
    """The proof is only as good as the checker, so the checker is shown each failure it exists to catch."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    from services.api.deps import ApiOut

    class Strict(BaseModel):                     # an ordinary model: filters what it does not name
        kept: int

    class Loose(ApiOut):
        kept: int
        sometimes: str | None = None

    app = FastAPI()

    @app.get("/dropped", response_model=Strict)
    async def dropped():
        return {"kept": 1, "lost": 2}

    @app.get("/nulled", response_model=Loose)
    async def nulled():
        return {"kept": 1}

    @app.get("/undeclared", response_model=Loose, response_model_exclude_unset=True)
    async def undeclared():
        return {"kept": 1, "nobody_declared_this": 2}

    @app.get("/empty", response_model=list[Loose], response_model_exclude_unset=True)
    async def empty():
        return []

    @app.get("/fine", response_model=list[Loose], response_model_exclude_unset=True)
    async def fine():
        return [{"kept": 1}, {"kept": 2, "sometimes": None}]

    c = TestClient(app)
    for path, complaint in (("/dropped", "differs from what the handler returned"),
                            ("/nulled", "differs from what the handler returned"),
                            ("/undeclared", "does not declare"),
                            ("/empty", "nothing was checked")):
        with pytest.raises(AssertionError, match=complaint):
            assert_nothing_lost(call(c, "GET", path))
    assert assert_nothing_lost(call(c, "GET", "/fine")) == [{"kept": 1}, {"kept": 2, "sometimes": None}]
    assert assert_nothing_lost(call(c, "GET", "/empty"), empty_ok=("$",)) == []
