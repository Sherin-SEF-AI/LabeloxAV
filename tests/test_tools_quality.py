"""Agent B's quality tools: consensus merge (#11), explained errors (#12), VLM reason per label (#13), rare
scenes (#14).

The pure parts (clustering votes across replicas, the accept/drop/dispute rule, parsing a VLM reply, the
cross-session k-NN, diverse picking) are tested without a database. The write paths run against the test
database; the VLM is a stub with the same `chat_json` contract as the Ollama client, and SigLIP's text tower
is replaced by a fixed vector, so nothing here loads a model. A real qwen2.5vl check is a smoke run, reported
separately.
"""

from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from core.timebase import now_ns, seconds_to_ns

# ---- pure: consensus -----------------------------------------------------------------------------------


def _v(voter, cls, box, human=True, cid=1, oid=None):
    from services.labelops.agreement import Vote

    return Vote(voter=voter, object_id=oid or uuid.uuid4().hex, bbox=[float(x) for x in box], class_id=cid,
                class_name=cls, conf=1.0 if human else 0.7, human=human)


def test_clusters_are_one_per_thing_with_at_most_one_box_per_voter():
    from services.labelops.agreement import cluster_votes

    a = [_v("r0", "car", [0, 0, 100, 100]), _v("r0", "car", [300, 300, 400, 400])]
    b = [_v("r1", "car", [5, 5, 105, 105]), _v("r1", "truck", [2, 2, 98, 98])]   # two boxes on one car
    m = [_v("model", "car", [302, 298, 401, 399], human=False)]
    cl = cluster_votes({"r0": a, "r1": b, "model": m}, 0.5)
    sizes = sorted(len(c) for c in cl)
    assert sizes == [1, 2, 2], "r1's second box on the same car opens its own cluster rather than double-voting"
    for c in cl:
        assert len({v.voter for v in c}) == len(c)


def test_decide_accepts_a_quorum_drops_a_model_only_box_and_disputes_the_rest():
    from services.labelops.agreement import decide_cluster

    voters = ["r0", "r1", "model"]
    agree = [_v("r0", "car", [0, 0, 100, 100]), _v("r1", "car", [4, 4, 104, 104]),
             _v("model", "truck", [0, 0, 100, 100], human=False, cid=2)]
    d = decide_cluster(agree, voters, quorum=2, iou_thresh=0.5)
    assert d["status"] == "accepted" and d["class_name"] == "car" and d["agree_n"] == 2
    assert d["bbox"] == [2.0, 2.0, 102.0, 102.0], "merged from the agreeing voters only"

    only_model = [_v("model", "car", [0, 0, 50, 50], human=False)]
    assert decide_cluster(only_model, voters, quorum=2, iou_thresh=0.5)["status"] == "dropped"

    split = [_v("r0", "car", [0, 0, 100, 100]), _v("r1", "truck", [0, 0, 100, 100], cid=2)]
    d = decide_cluster(split, voters, quorum=2, iou_thresh=0.5)
    assert d["status"] == "disputed" and "classes differ" in d["reason"]

    one_person = [_v("r0", "car", [0, 0, 100, 100])]
    d = decide_cluster(one_person, voters, quorum=2, iou_thresh=0.5)
    assert d["status"] == "disputed", "a person's box is never silently dropped"


def test_require_human_blocks_a_machine_only_quorum():
    from services.labelops.agreement import decide_cluster

    c = [_v("r0", "car", [0, 0, 100, 100]), _v("model", "truck", [0, 0, 100, 100], human=False, cid=2),
         _v("m2", "truck", [0, 0, 100, 100], human=False, cid=2)]
    assert decide_cluster(c, ["r0", "model", "m2"], quorum=2, iou_thresh=0.5)["status"] == "disputed"
    assert decide_cluster(c, ["r0", "model", "m2"], quorum=2, iou_thresh=0.5,
                          require_human=False)["status"] == "accepted"


# ---- pure: VLM reply -----------------------------------------------------------------------------------


def test_check_reply_parsing_refuses_anything_outside_the_contract():
    from services.autolabel.paths.path_c_vlm import parse_check_reply

    alts = ["e_auto", "tempo", "object_fallback"]
    v, r, a = parse_check_reply({"verdict": "incorrect", "reason": "It has three wheels. And a meter.",
                                 "alternative_class": "e_auto"}, "autorickshaw", alts)
    assert (v, r, a) == ("incorrect", "It has three wheels.", "e_auto")
    assert parse_check_reply({"verdict": "maybe"}, "car", alts)[0] == "unsure"
    assert parse_check_reply({"verdict": "correct", "alternative_class": "tempo"}, "car", alts)[2] is None
    assert parse_check_reply({"verdict": "incorrect", "alternative_class": "car"}, "car", ["car"])[0] == "unsure"
    assert parse_check_reply({"verdict": "incorrect", "alternative_class": "spaceship"}, "car", alts)[2] is None


class _FakeVlm:
    """Same chat_json contract as OllamaVlmClient; records what it was asked."""

    model = "fake-vlm"

    def __init__(self, reply: dict | None = None, fail: bool = False):
        self.reply = reply or {"verdict": "correct", "reason": "Four wheels and a boot.",
                               "alternative_class": "none"}
        self.fail = fail
        self.calls: list[dict] = []

    def chat_json(self, prompt, *, model=None, image_jpeg=None, temperature=0.0, schema=None, purpose="judge"):
        self.calls.append({"prompt": prompt, "schema": schema, "purpose": purpose, "image": bool(image_jpeg)})
        if self.fail:
            raise RuntimeError("connection refused")
        return self.reply

    def verify(self, *a, **k):  # pragma: no cover - chat_json is preferred
        raise AssertionError("verify should not be called when chat_json exists")


def test_check_label_asks_with_a_schema_and_reports_an_outage_as_not_ok():
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_c_vlm import VlmVerifier

    onto = get_ontology()
    img = np.full((200, 300, 3), 120, np.uint8)
    fake = _FakeVlm()
    res = VlmVerifier(fake, onto).check_label(img, (10, 10, 100, 100), onto.by_name("sedan").id)
    assert res.ok and res.verdict == "correct" and res.reason == "Four wheels and a boot."
    assert fake.calls[0]["purpose"] == "vlm_label_check", "the call is accounted in the ledger under its purpose"
    enum = fake.calls[0]["schema"]["properties"]["alternative_class"]["enum"]
    assert "sedan" not in enum and "none" in enum

    down = VlmVerifier(_FakeVlm(fail=True), onto).check_label(img, (10, 10, 100, 100), onto.by_name("sedan").id)
    assert not down.ok and down.error and down.verdict == "unsure"


# ---- pure: rare scenes ---------------------------------------------------------------------------------


def test_cross_session_knn_ignores_the_frames_own_drive():
    from services.intelligence.rare_scenes import cross_session_knn

    rng = np.random.default_rng(0)
    common = rng.normal(size=8)
    X = np.stack([common + 0.01 * rng.normal(size=8) for _ in range(6)]
                 + [rng.normal(size=8), rng.normal(size=8) * 0 + np.r_[1, -1, 1, -1, 1, -1, 1, -1]])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # Rows 0-2 drive A, 3-5 drive B (same scene), 6-7 drive C (an odd scene filmed twice).
    sess = ["A", "A", "A", "B", "B", "B", "C", "C"]
    score, nn, sim = cross_session_knn(X, sess, k=2, chunk=3)
    assert score[6] > score[0] and score[7] > score[3], "drive C's frames are near each other but rare"
    assert sess[nn[0]] != "A" and sess[nn[6]] != "C"
    s2, nn2, _ = cross_session_knn(X[:3], ["A", "A", "A"], k=2)
    assert np.isnan(s2).all() and (nn2 == -1).all(), "a lone drive is unmeasured, not infinitely rare"


def test_pick_diverse_caps_each_drive_and_cluster():
    from services.intelligence.rare_scenes import pick_diverse

    sess = ["A"] * 5 + ["B"] * 5
    clusters = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
    got = pick_diverse(list(range(10)), sess, clusters, top=10, per_session=2, per_cluster=1)
    assert got == [0, 3, 5, 8]


def test_pixel_screen_sets_aside_blank_and_flat_frames_but_not_a_photo():
    from services.intelligence.rare_scenes import pixel_screen

    grey = np.full((360, 640, 3), 128, np.uint8)
    grey[100:110, 200:220] = 30                               # a corrupt decode: grey with a smear
    assert "flat grey" in pixel_screen(grey)
    bars = np.zeros((360, 640, 3), np.uint8)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        bars[:, i * 160:(i + 1) * 160] = c
    assert "few distinct colours" in pixel_screen(bars)
    rng = np.random.default_rng(0)
    photo = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (9, 9), 0)
    assert pixel_screen(photo) is None
    assert pixel_screen(None) == "unreadable"


def test_why_rare_restates_measured_numbers():
    from services.intelligence.rare_scenes import why_rare

    w = why_rare({"nn_sim": 0.61, "cluster_size": 3, "cluster_sessions": 1, "n_objects": 0,
                  "scene": {"time_of_day": "night", "weather": "rain"}}, median_nn=0.93,
                 shares={"night + rain": 0.004})
    assert any("0.61" in s and "0.93" in s for s in w)
    assert any("3 frames from one drive" in s for s in w)
    assert any("Night + rain is 0.4% of the frames with a known scene" in s for s in w)
    assert "Nothing is labelled on it yet." in w


# ---- db ------------------------------------------------------------------------------------------------

def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _ensure_ontology(db):
    from db.models import OntologyClass, OntologyVersion
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    return onto


async def _session(db, n_frames: int, *, vehicle="QUAL-B", img_uri=None, origin="real"):
    from db.models import Frame
    from db.models import Session as DbSession

    onto = await _ensure_ontology(db)
    ts = now_ns()
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id=vehicle, start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(10),
                     city="BLR", sensors={}, ontology_version=onto.version, origin=origin))
    await db.flush()
    fids = []
    for i in range(n_frames):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f",
                     img_uri=img_uri or f"s3://missing/{fid}.jpg", width=640, height=480, quality=0.9,
                     scene={}, origin=origin))
        fids.append(fid)
    await db.flush()
    return sid, fids


def _obj(frame_id, cls, bbox, *, job=None, annotator=None, source="human", state="accepted", conf=1.0):
    from db.models import Object
    from services.autolabel.ontology import get_ontology

    return Object(object_id=uuid.uuid4(), frame_id=frame_id, class_id=get_ontology().by_name(cls).id,
                  bbox=[float(x) for x in bbox], conf=conf, source=source, state=state, attrs={}, provenance={},
                  version=1, job_id=job.job_id if job else None,
                  annotator_id=annotator.user_id if annotator else None)


async def _consensus_fixture(db):
    """A submitted two-replica task on one frame: an agreed car, a class conflict, a model-only box."""
    from db.models import LabelJob, LabelProject, User
    from services.labelops.jobs import create_task

    _sid, fids = await _session(db, 1)
    project = LabelProject(name=f"cons-{uuid.uuid4().hex[:8]}")
    db.add(project)
    await db.commit()
    out = await create_task(db, project_id=str(project.project_id), name="cons",
                            predicate={"frame_ids": [str(f) for f in fids]}, replicas=2)
    jobs = list((await db.execute(select(LabelJob).where(LabelJob.task_id == uuid.UUID(out["task_id"]))
                                  .order_by(LabelJob.replica_index))).scalars().all())
    users = [User(user_id=uuid.uuid4(), name=f"cons-{i}-{uuid.uuid4().hex[:6]}", role="annotator") for i in range(2)]
    db.add_all(users)
    await db.flush()
    for j in jobs:
        j.state = "submitted"
    f = fids[0]
    objs = {
        "a_car": _obj(f, "sedan", [10, 10, 110, 110], job=jobs[0], annotator=users[0]),
        "b_car": _obj(f, "sedan", [14, 12, 112, 108], job=jobs[1], annotator=users[1]),
        "a_rider": _obj(f, "rider", [300, 100, 340, 200], job=jobs[0], annotator=users[0]),
        "b_ped": _obj(f, "pedestrian", [301, 101, 341, 199], job=jobs[1], annotator=users[1]),
        "model_car": _obj(f, "sedan", [12, 10, 110, 112], source="fused", state="review", conf=0.8),
        "model_fp": _obj(f, "pole", [500, 10, 520, 200], source="fused", state="review", conf=0.4),
    }
    db.add_all(objs.values())
    await db.commit()
    return out["task_id"], jobs, f, objs


@pytest.mark.db
def test_migration_adds_consensus_and_keeps_every_other_source():
    from db.session import get_sessionmaker

    async def go():
        async with get_sessionmaker()() as db:
            return (await db.execute(text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname='ck_object_source'"))).scalar()

    defn = _run(go())
    for v in ("consensus", "described", "fused", "human", "vlm_review", "synthetic"):
        assert f"'{v}'" in defn


@pytest.mark.db
def test_merge_writes_consensus_routes_disputes_and_is_idempotent():
    from db.models import Issue, Object
    from db.session import get_sessionmaker
    from services.labelops import agreement as ag

    async def go():
        async with get_sessionmaker()() as db:
            task_id, jobs, f, objs = await _consensus_fixture(db)
            gid = str(jobs[0].replica_group)
            preview = await ag.merge_replica_group(db, gid, dry_run=True)
            n_after_preview = (await db.execute(select(Object).where(
                Object.frame_id == f, Object.source == "consensus"))).scalars().all()
            res = await ag.score_and_merge(db, gid)
            cons = list((await db.execute(select(Object).where(
                Object.frame_id == f, Object.source == "consensus"))).scalars().all())
            issues = list((await db.execute(select(Issue).where(
                Issue.frame_id == f, Issue.kind == "disagreement"))).scalars().all())
            again = await ag.merge_replica_group(db, gid)
            cons2 = list((await db.execute(select(Object).where(
                Object.frame_id == f, Object.source == "consensus"))).scalars().all())
            board = await ag.consensus_board(db, task_id)
            queue = await ag.task_disagreements(db, task_id)
            return preview, n_after_preview, res, cons, issues, again, cons2, board, queue, objs

    preview, n_after_preview, res, cons, issues, again, cons2, board, queue, objs = _run(go())
    assert preview["dry_run"] and not n_after_preview, "a preview writes nothing"
    assert preview["counts"]["accepted"] == 1 and preview["counts"]["disputed"] == 1
    assert preview["counts"]["dropped"] == 1, "the model's pole, which both annotators left out"

    c = res["counts"]
    assert (c["created"], c["disputed"], c["dropped"]) == (1, 1, 1)
    assert len(cons) == 1
    m = cons[0]
    assert m.state == "auto_accept" and m.job_id is None and m.annotator_id is None
    assert m.provenance["consensus"]["agree_n"] == 3, "both annotators and the model drew a sedan there"
    assert {i.object_id for i in issues} == {objs["a_rider"].object_id, objs["b_ped"].object_id}
    assert res["agreement"]["pairs"] == 1, "scored first, as before"

    assert again["counts"]["unchanged"] == 1 and again["counts"]["created"] == 0
    assert [o.object_id for o in cons2] == [m.object_id], "merging again replaces nothing"
    assert again["counts"]["issues_opened"] == 0, "and opens no second issue on the same box"

    assert board["heatmap"]["classes"] and board["groups"][0]["ready"] is True
    assert board["groups"][0]["consensus_objects"] == 1
    row = board["heatmap"]["cells"][board["heatmap"]["classes"].index("sedan")][0]
    assert row["accepted"] == 1 and row["rate"] == 1.0
    assert {q["object_id"] for q in queue} == {str(objs["a_rider"].object_id), str(objs["b_ped"].object_id)}
    assert all("classes differ" in q["reason"] for q in queue)


@pytest.mark.db
def test_a_stale_consensus_object_goes_but_an_edited_one_stays():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.labelops import agreement as ag

    async def go():
        async with get_sessionmaker()() as db:
            _task, jobs, f, _objs = await _consensus_fixture(db)
            gid = str(jobs[0].replica_group)
            await ag.merge_replica_group(db, gid)                       # people + model agree on the sedan
            # Without the model the agreeing set changes, so the old object no longer describes a vote.
            res = await ag.merge_replica_group(db, gid, include_model=False)
            left = list((await db.execute(select(Object).where(
                Object.frame_id == f, Object.source == "consensus"))).scalars().all())
            left[0].version = 2                                         # a person has edited it since
            await db.commit()
            res2 = await ag.merge_replica_group(db, gid)
            still = list((await db.execute(select(Object).where(
                Object.frame_id == f, Object.source == "consensus"))).scalars().all())
            return res, left, res2, still

    res, left, res2, still = _run(go())
    assert (res["counts"]["created"], res["counts"]["removed"]) == (1, 1)
    assert len(left) == 1 and left[0].provenance["consensus"]["include_model"] is False
    assert res2["counts"]["kept_edited"] == 1 and res2["counts"]["removed"] == 0
    assert len(still) == 2, "the edited object is a person's now and stays beside the new tally"


@pytest.mark.db
def test_merge_refuses_an_open_group_and_a_bad_quorum():
    from db.session import get_sessionmaker
    from services.labelops import agreement as ag

    async def go():
        async with get_sessionmaker()() as db:
            _task, jobs, _f, _objs = await _consensus_fixture(db)
            gid = str(jobs[0].replica_group)
            with pytest.raises(ag.AgreementError, match="quorum"):
                await ag.merge_replica_group(db, gid, quorum=4)
            jobs[1].state = "in_progress"
            await db.commit()
            with pytest.raises(ag.AgreementError, match="still open"):
                await ag.merge_replica_group(db, gid)

    _run(go())


@pytest.mark.db
def test_explanations_draw_the_box_a_duplicate_overlaps_and_the_track_it_breaks():
    from db.models import ErrorCandidate, Track
    from db.session import get_sessionmaker
    from services.errordetect.explain import explain_candidate, frame_explanations

    async def go():
        async with get_sessionmaker()() as db:
            sid, fids = await _session(db, 3)
            onto = await _ensure_ontology(db)
            tr = Track(track_id=uuid.uuid4(), session_id=sid, class_id=onto.by_name("sedan").id,
                       first_ts_ns=0, last_ts_ns=1)
            db.add(tr)
            await db.flush()
            keep = _obj(fids[0], "sedan", [10, 10, 110, 110], source="fused", state="review", conf=0.9)
            dup = _obj(fids[0], "sedan", [12, 11, 111, 109], source="fused", state="review", conf=0.5)
            mates = [_obj(f, "sedan", [10, 10, 110, 110], source="fused", state="review") for f in fids[1:]]
            flip = _obj(fids[0], "truck", [200, 200, 300, 300], source="fused", state="review")
            for o in [*mates, flip]:
                o.track_id = tr.track_id
            db.add_all([keep, dup, *mates, flip])
            await db.flush()
            c_dup = ErrorCandidate(object_id=dup.object_id, kind="reanalyze", score=0.7, proposed_label=None,
                                   detail={"rule": "duplicate", "reason": "quality review flagged 'duplicate'"})
            c_trk = ErrorCandidate(object_id=flip.object_id, kind="track_inconsistent", score=0.66,
                                   proposed_label={"class_id": onto.by_name("sedan").id, "class_name": "sedan"},
                                   detail={"track_id": str(tr.track_id), "given_class": "truck",
                                           "track_majority": "sedan", "track_len": 3})
            db.add_all([c_dup, c_trk])
            await db.commit()
            e_dup = await explain_candidate(db, str(c_dup.candidate_id))
            e_trk = await explain_candidate(db, str(c_trk.candidate_id))
            frame = await frame_explanations(db, str(fids[0]))
            return e_dup, e_trk, frame, keep, mates

    e_dup, e_trk, frame, keep, mates = _run(go())
    roles = {e["role"]: e for e in e_dup["evidence"]}
    assert set(roles) == {"subject", "overlaps"}
    assert roles["overlaps"]["object_id"] == str(keep.object_id)
    assert "covers the same object" in e_dup["reason"]
    assert e_dup["neighbours"][0]["crop_url"].endswith("/crop")
    assert e_trk["proposed"]["class_name"] == "sedan" and "sedan" in e_trk["reason"]
    assert {n["object_id"] for n in e_trk["neighbours"]} == {str(m.object_id) for m in mates}
    assert len(frame["items"]) == 2 and frame["width"] == 640


@pytest.mark.db
def test_vlm_check_stores_a_reason_replaces_it_on_recheck_and_marks_it_stale():
    from core.storage import get_object_store
    from db.models import MachineVerdict, Object
    from db.session import get_sessionmaker
    from services.intelligence.vlm_qa import CHECK_JUDGE, check_labels, latest_checks

    store = get_object_store()
    store.ensure_bucket()
    img = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (20, 200, 20), -1)
    uri = store.put_bytes(f"tests/quality/{uuid.uuid4().hex}.jpg", cv2.imencode(".jpg", img)[1].tobytes(),
                          "image/jpeg")

    async def go():
        async with get_sessionmaker()() as db:
            _sid, fids = await _session(db, 1, img_uri=uri)
            o = _obj(fids[0], "autorickshaw", [100, 100, 300, 300], source="fused", state="review")
            db.add(o)
            await db.commit()
            fake = _FakeVlm({"verdict": "incorrect", "reason": "It is a closed e-rickshaw with no meter. Maybe.",
                             "alternative_class": "e_auto"})
            r1 = await check_labels(db, [o.object_id], client=fake)
            fake2 = _FakeVlm({"verdict": "correct", "reason": "Three wheels and a canopy.",
                              "alternative_class": "none"})
            r2 = await check_labels(db, [o.object_id], client=fake2)
            n = (await db.execute(select(MachineVerdict).where(MachineVerdict.object_id == o.object_id,
                                                               MachineVerdict.judge == CHECK_JUDGE))).scalars().all()
            obj = await db.get(Object, o.object_id)
            from services.autolabel.ontology import get_ontology

            obj.class_id = get_ontology().by_name("e_auto").id
            await db.commit()
            stale = await latest_checks(db, [o.object_id])
            down = await check_labels(db, [o.object_id], client=_FakeVlm(fail=True))
            return o, r1, r2, n, stale, down

    o, r1, r2, n, stale, down = _run(go())
    first = r1["results"][str(o.object_id)]
    assert first["verdict"] == "incorrect" and first["alternative"] == "e_auto"
    assert first["reason"] == "It is a closed e-rickshaw with no meter."
    assert first["cost_usd"] is None or first["cost_usd"] == 0.0
    assert r2["results"][str(o.object_id)]["verdict"] == "correct"
    assert len(n) == 1, "one opinion per object per model, replaced on re-check"
    assert stale[str(o.object_id)]["stale"] is True, "the label moved after the model looked"
    assert down["checked"] == 0 and down["failed"] == 1, "an outage stores nothing"


@pytest.mark.db
def test_rare_scene_ranking_and_search_skip_synthetic_frames_and_explain():
    from db.models import FrameEmbedding
    from db.session import get_sessionmaker
    from services.intelligence import rare_scenes

    rng = np.random.default_rng(1)
    base = rng.normal(size=768)

    async def go():
        async with get_sessionmaker()() as db:
            await db.execute(text("DELETE FROM frame_embedding"))
            common = []
            for _ in range(3):        # three drives down the same road
                _s, f = await _session(db, 4)
                common += f
            _s, odd = await _session(db, 3)       # one drive somewhere nothing else looks like
            _s, synth = await _session(db, 2, origin="synthetic")
            other = rng.normal(size=768)
            for fid in common:
                v = base + 0.05 * rng.normal(size=768)
                db.add(FrameEmbedding(frame_id=fid, dino_vec=v.tolist(), siglip_vec=None, model_versions={}))
            for fid in odd:
                v = other + 0.05 * rng.normal(size=768)
                db.add(FrameEmbedding(frame_id=fid, dino_vec=v.tolist(), siglip_vec=None, model_versions={}))
            for fid in synth:
                db.add(FrameEmbedding(frame_id=fid, dino_vec=(-base).tolist(), siglip_vec=None, model_versions={}))
            await db.commit()
            ranked = await rare_scenes.rank_corpus(db, top=20, per_session=10, per_cluster=20,
                                                   screen_pixels=False)
            screened = await rare_scenes.rank_corpus(db, top=20, per_session=10, per_cluster=20)
            found = await rare_scenes.search(db, frame_id=str(odd[0]), k=5, rarest_first=True)
            return ranked, found, odd, synth, screened

    ranked, found, odd, synth, screened = _run(go())
    ids = [f["frame_id"] for f in ranked["frames"]]
    assert ranked["n_frames"] == 15 and ranked["excluded_not_real"] >= 2
    assert not {str(s) for s in synth} & set(ids), "synthetic frames are never ranked"
    assert set(ids[:3]) == {str(o) for o in odd}, "the drive nothing else resembles comes first"
    top = ranked["frames"][0]
    assert top["nn_sim"] < ranked["median_nn_sim"] and top["why"]
    assert found["mode"] == "example" and found["results"]
    # Its own drive's twin is the match (the diversity rerank keeps one of two near-identical frames).
    assert found["results"][0]["frame_id"] in {str(o) for o in odd[1:]}
    assert found["results"][0]["score"] is not None
    # These fixture frames have no pixels behind them, so the screen sets every one aside, with the reason.
    assert screened["frames"] == [] and screened["n_suspect"] == 15
    assert screened["suspect_frames"][0]["reason"] == "unreadable"


# ---- http -----------------------------------------------------------------------------------------------

@pytest.mark.db
def test_routes_merge_preview_why_and_task(monkeypatch):
    from _authutil import auth_headers

    from db.models import ErrorCandidate, LabelProject
    from db.session import get_sessionmaker
    from services.api.main import app
    from services.tools import registry

    real_enqueue = registry.enqueue

    async def no_start(*a, **k):
        k["start"] = False
        return await real_enqueue(*a, **k)

    monkeypatch.setattr(registry, "enqueue", no_start)

    async def seed():
        async with get_sessionmaker()() as db:
            task_id, jobs, f, objs = await _consensus_fixture(db)
            c = ErrorCandidate(object_id=objs["model_fp"].object_id, kind="reanalyze", score=0.6,
                               proposed_label=None, detail={"rule": "min_box_size", "reason": ""})
            p = LabelProject(name=f"rare-{uuid.uuid4().hex[:8]}")
            db.add_all([c, p])
            await db.commit()
            return task_id, str(jobs[0].replica_group), str(f), str(c.candidate_id), str(p.project_id)

    task_id, gid, fid, cid, pid = _run(seed())
    with TestClient(app) as c:
        h = auth_headers("reviewer")
        r = c.get("/api/tools/quality/consensus/tasks", headers=h)
        assert r.status_code == 200, r.text
        assert any(t["task_id"] == task_id and t["ready_groups"] == 1 for t in r.json()["tasks"])
        r = c.get(f"/api/tools/quality/consensus/tasks/{task_id}?quorum=3", headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["params"]["quorum"] == 3
        r = c.post(f"/api/tools/quality/consensus/groups/{gid}/preview", headers=h, json={"quorum": 2})
        assert r.status_code == 200 and r.json()["dry_run"] is True
        assert c.post(f"/api/tools/quality/consensus/groups/{gid}/preview", headers=h,
                      json={"quorum": 9}).status_code == 400
        r = c.post("/api/tools/quality/consensus/merge", headers=h, json={"task_id": task_id})
        assert r.status_code == 200 and r.json()["kind"] == "consensus_merge", r.text
        assert c.post("/api/tools/quality/consensus/merge", headers=h, json={}).status_code == 400
        r = c.get(f"/api/tools/quality/errors/{cid}/why", headers=h)
        assert r.status_code == 200 and r.json()["rule"] == "min_box_size"
        assert "minimum" in r.json()["reason"]
        assert c.get(f"/api/tools/quality/errors/{uuid.uuid4()}/why", headers=h).status_code == 404
        r = c.get(f"/api/tools/quality/frames/{fid}/why", headers=h)
        assert r.status_code == 200 and len(r.json()["items"]) == 1
        r = c.post("/api/tools/quality/rare-scenes/task", headers=h,
                   json={"project_id": pid, "name": "rare picks", "frame_ids": [fid, fid]})
        assert r.status_code == 200, r.text
        assert r.json()["n_frames"] == 1
        assert c.post("/api/tools/quality/vlm/check", headers=h,
                      json={"object_ids": [str(uuid.uuid4()) for _ in range(6)]}).status_code == 422
        r = c.post("/api/tools/quality/vlm/check-batch", headers=h, json={"object_ids": [str(uuid.uuid4())]})
        assert r.status_code == 200 and r.json()["kind"] == "vlm_label_check" and r.json()["status"] == "pending"
        r = c.post("/api/tools/quality/rare-scenes/rank", headers=h, json={"top": 50})
        assert r.status_code == 200 and r.json()["kind"] == "rare_scene_rank"
        assert c.get("/api/tools/quality/rare-scenes/latest", headers=h).json()["running"] is not None
        assert c.post("/api/tools/quality/rare-scenes/search", headers=h, json={}).status_code == 400


@pytest.mark.db
def test_writes_are_reviewer_only():
    from _authutil import auth_headers

    from core.config import get_settings
    from services.api.main import app

    if not get_settings().auth.enabled:
        pytest.skip("auth disabled in this posture")
    with TestClient(app) as c:
        h = auth_headers("annotator")
        assert c.post("/api/tools/quality/consensus/merge", headers=h,
                      json={"task_id": str(uuid.uuid4())}).status_code == 403
        assert c.post("/api/tools/quality/rare-scenes/rank", headers=h, json={}).status_code == 403
        assert c.post("/api/tools/quality/vlm/check-batch", headers=h,
                      json={"object_ids": [str(uuid.uuid4())]}).status_code == 403


# ---- job kinds through the queue -------------------------------------------------------------------------

@pytest.mark.db
def test_consensus_and_rank_kinds_run_to_done_through_the_queue(monkeypatch):
    from db.models import ToolJob
    from db.session import get_sessionmaker
    from services.tools.registry import enqueue, run_job

    async def go():
        async with get_sessionmaker()() as db:
            task_id, _jobs, _f, _objs = await _consensus_fixture(db)
            merge = await enqueue(db, "consensus_merge", {"task_id": task_id, "quorum": 2}, start=False)
            rank = await enqueue(db, "rare_scene_rank", {"top": 10}, start=False)
        s1 = await run_job(merge.job_id)
        s2 = await run_job(rank.job_id)
        async with get_sessionmaker()() as db:
            return s1, s2, await db.get(ToolJob, merge.job_id), await db.get(ToolJob, rank.job_id)

    s1, s2, merge, rank = _run(go())
    assert (s1, s2) == ("done", "done"), (merge.error, rank.error)
    assert merge.result["groups"] == 1 and merge.result["counts"]["created"] == 1
    assert merge.result["per_class"]["sedan"]["accepted"] == 1
    assert "frames" in rank.result


@pytest.mark.db
def test_vlm_batch_kind_checks_a_session_least_confident_first(monkeypatch):
    from db.models import ToolJob
    from db.session import get_sessionmaker
    from services.intelligence import vlm_qa
    from services.tools.registry import enqueue, run_job

    seen: list[list[str]] = []

    async def fake_check(db, ids, *, client=None, user_id=None, progress=None):
        seen.append([str(i) for i in ids])
        if progress:
            await progress(1.0)
        return {"requested": len(ids), "found": len(ids), "checked": len(ids), "failed": 0, "errors": [],
                "results": {str(i): {"verdict": "correct"} for i in ids}, "model": "fake"}

    monkeypatch.setattr(vlm_qa, "check_labels", fake_check)

    async def go():
        async with get_sessionmaker()() as db:
            sid, fids = await _session(db, 1)
            lo = _obj(fids[0], "sedan", [0, 0, 50, 50], source="fused", state="review", conf=0.2)
            hi = _obj(fids[0], "sedan", [60, 0, 110, 50], source="fused", state="review", conf=0.9)
            db.add_all([lo, hi])
            await db.commit()
            job = await enqueue(db, "vlm_label_check", {"session_id": str(sid), "limit": 1}, session_id=sid)
            assert job.gpu is True
        status = await run_job(job.job_id)
        async with get_sessionmaker()() as db:
            return status, await db.get(ToolJob, job.job_id), lo

    status, job, lo = _run(go())
    assert status == "done", job.error
    assert seen == [[str(lo.object_id)]]
    assert job.result["verdicts"] == {"correct": 1}
