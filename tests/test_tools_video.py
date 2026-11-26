"""Agent A's video and editor tools: SAM 2 video tracking (#1), describe (#2) and correction fan-out (#4).

The pure parts (prompt merging, propagation order, letterbox, box fusion, the confusion rule) are tested
without a database. The write paths run against the test database with the model replaced by a stub that
returns known masks, so what is asserted is the persistence contract: one track, interp_source, state,
re-prompt updates rather than duplicates, a person's label is never overwritten, and a fan-out is one undo.
The real SAM 2 video model is exercised in tests/test_tools_video_gpu.py.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.timebase import now_ns, seconds_to_ns

# ---- pure -----------------------------------------------------------------------------------------------


def test_merge_prompts_combines_one_frame_and_drops_empties():
    from services.temporal.sam2_video import Prompt, merge_prompts

    m = merge_prompts([Prompt(index=2, points=[[1, 2]], labels=[1]), Prompt(index=2, points=[[3, 4]], labels=[0]),
                       Prompt(index=5, box=[0, 0, 10, 10]), Prompt(index=7)])
    assert sorted(m) == [2, 5]
    assert m[2].points == [[1.0, 2.0], [3.0, 4.0]] and m[2].labels == [1, 0]
    assert m[5].box == [0.0, 0.0, 10.0, 10.0]


def test_merge_prompts_defaults_missing_labels_to_positive():
    from services.temporal.sam2_video import Prompt, merge_prompts

    assert merge_prompts([Prompt(index=0, points=[[1, 1], [2, 2]])])[0].labels == [1, 1]


def test_propagation_order_is_forward_from_earliest_prompt_then_backward():
    from services.temporal.sam2_video import propagation_order

    order = propagation_order(7, [2, 4])
    assert order == [(3, False), (5, False), (6, False), (1, True), (0, True)]
    assert propagation_order(3, []) == []


def test_letterbox_scales_to_the_long_side_and_pads_bottom_right():
    from services.temporal.sam2_video import IMGSZ, Prompt, letterbox, prompt_arrays

    img = np.zeros((540, 960, 3), np.uint8)
    arr, r = letterbox(img)
    assert arr.shape == (3, IMGSZ, IMGSZ) and r == pytest.approx(IMGSZ / 960)
    coords, labels = prompt_arrays(Prompt(index=0, points=[[100, 50]], labels=[1], box=[10, 20, 30, 40]), r)
    assert labels.tolist() == [[2, 3, 1]]
    assert coords[0, 2].tolist() == pytest.approx([100 * r, 50 * r])


def test_fuse_agreement_matches_by_iou_and_keeps_singletons():
    from services.autolabel.describe_session import fuse_agreement

    a = [([0, 0, 10, 10], 0.8), ([50, 50, 60, 60], 0.4)]
    b = [([1, 1, 11, 11], 0.6), ([200, 200, 210, 210], 0.5)]
    out = fuse_agreement(a, b)
    agreed = [d for d in out if d["agreed"]]
    assert len(agreed) == 1 and agreed[0]["conf"] == 0.8
    assert agreed[0]["bbox"][0] == pytest.approx((0 * 0.8 + 1 * 0.6) / 1.4)
    assert {d["from"] for d in out if not d["agreed"]} == {"a", "b"}
    assert fuse_agreement([], []) == []


def test_new_proposals_drops_duplicates_of_existing_labels():
    from services.autolabel.describe import Proposal
    from services.autolabel.describe_session import new_proposals

    props = [Proposal(bbox=[0, 0, 10, 10], conf=0.9, polygons=[]), Proposal(bbox=[100, 100, 120, 120], conf=0.5,
                                                                           polygons=[])]
    keep = new_proposals(props, [[1, 1, 10, 10]])
    assert [p.conf for p in keep] == [0.5]
    assert new_proposals(props, []) == props


def test_confusion_match_reads_the_labels_own_votes():
    from services.api.routers.tools_video import confusion_match

    prov = {"proposals": [{"class_name": "autorickshaw"}, {"class_name": "truck"}],
            "reasoning": {"suggested_class": "truck"}}
    assert confusion_match(prov, "truck")
    assert not confusion_match(prov, "bus")
    assert confusion_match({"reasoning": {"suggested_class": "bus"}}, "bus")
    assert not confusion_match(None, "bus")


# ---- db -------------------------------------------------------------------------------------------------

SIZE, BOX = (240, 320), 40


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _square_at(i: int) -> tuple[int, int]:
    return 40 + 8 * i, 80


def _seed(n: int = 6) -> dict:
    """A session of n frames on one camera with a textured square moving right, and an object on frame 2."""
    from core.storage import get_object_store
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()
    rng = np.random.default_rng(3)
    tex = rng.integers(0, 255, size=(BOX, BOX, 3), dtype=np.uint8)

    async def go():
        sid, ts = uuid.uuid4(), now_ns()
        out = {"session_id": str(sid), "frames": []}
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="SAM2VID", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(n),
                             city="BLR", sensors={}, ontology_version=onto.version))
            await db.flush()
            for i in range(n):
                img = np.full((*SIZE, 3), 90, np.uint8)
                x, y = _square_at(i)
                img[y:y + BOX, x:x + BOX] = tex
                ok, buf = cv2.imencode(".jpg", img)
                t = ts + seconds_to_ns(i) // 10
                uri = store.put_bytes(f"frames/{sid}/cam_front/{t}.jpg", buf.tobytes(), "image/jpeg")
                fid = uuid.uuid4()
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t, cam_id="cam_front", img_uri=uri,
                             width=SIZE[1], height=SIZE[0], quality=0.9, scene={}))
                out["frames"].append(str(fid))
            await db.commit()
        return out

    return _run(go())


class _StubTracker:
    """Yields the square's true mask on every frame, except the frames listed as `gone`."""

    def __init__(self, gone: set[int] | None = None):
        self.gone = gone or set()
        self.calls: list = []

    def track(self, load_image, n_frames, prompts):
        from services.temporal.sam2_video import FrameMask, merge_prompts

        self.calls.append(merge_prompts(prompts))
        for i in range(n_frames):
            load_image(i)
            if i in self.gone:
                yield FrameMask(index=i, mask=None, score=0.0)
                continue
            m = np.zeros(SIZE, bool)
            x, y = _square_at(i)
            m[y:y + BOX, x:x + BOX] = True
            yield FrameMask(index=i, mask=m, score=0.93)


async def _objects(track_id: str) -> list:
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        return list((await db.execute(select(Object).where(Object.track_id == uuid.UUID(track_id)))).scalars())


@pytest.mark.db
def test_track_from_a_click_writes_one_tool_object_per_frame_on_one_track():
    from core.storage import get_object_store
    from services.temporal.sam2_video import run_track

    s = _seed(6)
    stub = _StubTracker()
    res = _run(run_track({"frame_id": s["frames"][2], "class_name": "sedan", "before": 5, "after": 5,
                          "prompts": [{"frame_id": s["frames"][2], "points": [[60, 100]], "labels": [1]}]},
                         tracker_factory=lambda: stub))
    assert res["method"] == "sam2_video" and len(res["created"]) == 6 and res["frames"] == 6
    objs = _run(_objects(res["track_id"]))
    assert len(objs) == 6
    assert {o.interp_source for o in objs} == {"sam2_video"}
    assert {o.state for o in objs} == {"annotate"} and {o.source for o in objs} == {"propagated"}
    by_frame = {str(o.frame_id): o for o in objs}
    x, y = _square_at(4)
    assert by_frame[s["frames"][4]].bbox == [x, y, x + BOX, y + BOX]
    mask = json.loads(get_object_store().get_bytes(by_frame[s["frames"][4]].mask_uri))
    assert mask["polygons"] and mask["width"] == SIZE[1]
    assert by_frame[s["frames"][2]].provenance["prompted"] is True


@pytest.mark.db
def test_reprompt_updates_its_own_objects_keeps_a_persons_and_drops_what_it_no_longer_sees():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.temporal.sam2_video import run_track

    s = _seed(6)
    first = _run(run_track({"frame_id": s["frames"][0], "class_name": "sedan", "before": 0, "after": 5,
                            "prompts": [{"frame_id": s["frames"][0], "box": [40, 80, 80, 120]}]},
                           tracker_factory=lambda: _StubTracker()))
    tid = first["track_id"]

    # A person relabels frame 3's object: it is theirs now and a re-run must not touch it.
    async def humanise():
        async with get_sessionmaker()() as db:
            for o in await _objects(tid):
                if str(o.frame_id) == s["frames"][3]:
                    row = await db.get(Object, o.object_id)
                    row.source, row.state = "human", "accepted"
                    row.class_id = get_ontology().by_name("suv").id
            await db.commit()

    _run(humanise())
    stub = _StubTracker(gone={5})
    second = _run(run_track({"frame_id": s["frames"][0], "track_id": tid, "before": 0, "after": 5,
                             "prompts": [{"frame_id": s["frames"][4], "points": [[80, 100]], "labels": [1]}]},
                            tracker_factory=lambda: stub))
    assert second["created"] == []
    assert len(second["updated"]) == 4          # frames 0, 1, 2, 4
    assert len(second["removed"]) == 1          # frame 5: the corrected run no longer sees it
    assert len(second["kept"]) == 1             # frame 3: the person's label
    # The first run's prompt was replayed alongside the new one.
    assert sorted(stub.calls[0]) == [0, 4]
    objs = _run(_objects(tid))
    assert len(objs) == 5
    human = [o for o in objs if o.source == "human"]
    assert len(human) == 1 and human[0].class_id == get_ontology().by_name("suv").id


@pytest.mark.db
def test_tracking_an_existing_label_uses_its_box_and_leaves_it_alone():
    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.temporal.sam2_video import run_track

    s = _seed(4)
    oid = uuid.uuid4()

    async def add():
        async with get_sessionmaker()() as db:
            db.add(Object(object_id=oid, frame_id=uuid.UUID(s["frames"][1]), class_id=get_ontology().by_name("sedan").id,
                          bbox=[48.0, 80.0, 88.0, 120.0], conf=1.0, source="human", state="accepted", attrs={},
                          provenance={}))
            await db.commit()

    _run(add())
    stub = _StubTracker()
    res = _run(run_track({"frame_id": s["frames"][1], "object_id": str(oid), "before": 3, "after": 3},
                         tracker_factory=lambda: stub))
    assert stub.calls[0][1].box == [48.0, 80.0, 88.0, 120.0]
    assert res["kept"] == [str(oid)] and len(res["created"]) == 3
    objs = _run(_objects(res["track_id"]))
    assert str(oid) in {str(o.object_id) for o in objs}  # the label joined the new track, unchanged otherwise
    anchor = next(o for o in objs if o.object_id == oid)
    assert anchor.source == "human" and anchor.bbox == [48.0, 80.0, 88.0, 120.0]


@pytest.mark.db
def test_a_failing_video_model_falls_back_to_flow_propagation():
    from services.temporal.sam2_video import run_track

    s = _seed(5)

    def broken():
        raise RuntimeError("no video model here")

    # refine=False keeps the fallback on CPU (flow only), so this test needs no GPU.
    res = _run(run_track({"frame_id": s["frames"][0], "class_name": "sedan", "before": 0, "after": 4,
                          "refine": False, "prompts": [{"frame_id": s["frames"][0], "box": [40, 80, 80, 120]}]},
                         tracker_factory=broken))
    assert res["method"] == "sam_propagate_fallback"
    assert "no video model here" in res["fallback_reason"]


@pytest.mark.db
def test_a_prompt_is_required():
    from services.temporal.sam2_video import run_track

    s = _seed(2)
    with pytest.raises(ValueError):
        _run(run_track({"frame_id": s["frames"][0], "class_name": "sedan", "prompts": []},
                       tracker_factory=lambda: _StubTracker()))


@pytest.mark.db
def test_describe_session_writes_review_proposals_and_skips_duplicates():
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.session import get_sessionmaker
    from services.autolabel.describe import Proposal
    from services.autolabel.describe_session import run_describe_session
    from services.autolabel.ontology import get_ontology

    s = _seed(3)
    cid = get_ontology().by_name("autorickshaw").id

    async def existing():
        async with get_sessionmaker()() as db:
            db.add(Object(frame_id=uuid.UUID(s["frames"][0]), class_id=cid, bbox=[10.0, 10.0, 50.0, 50.0], conf=1.0,
                          source="human", state="accepted", attrs={}, provenance={}))
            await db.commit()

    _run(existing())
    calls: list = []

    def fake_propose(img, phrase, *, min_conf):
        calls.append((phrase, min_conf))
        return [Proposal(bbox=[12.0, 10.0, 50.0, 52.0], conf=0.7, polygons=[[12, 10, 50, 10, 50, 52, 12, 52]]),
                Proposal(bbox=[200.0, 100.0, 260.0, 160.0], conf=0.4, polygons=[])]

    res = _run(run_describe_session({"phrase": "  Auto  Rickshaw ", "class_name": "autorickshaw"},
                                    session_id=uuid.UUID(s["session_id"]), propose_fn=fake_propose))
    assert calls[0] == ("auto rickshaw", 0.25)
    assert res["frames"] == 3 and res["written"] == 5 and res["skipped_duplicates"] == 1
    assert "Florence-2" in res["second_opinion"]

    async def described():
        async with get_sessionmaker()() as db:
            return list((await db.execute(select(Object).join(Frame).where(
                Frame.session_id == uuid.UUID(s["session_id"]), Object.source == "described"))).scalars())

    rows = _run(described())
    assert len(rows) == 5 and {o.state for o in rows} == {"review"}
    assert {o.provenance["described_as"] for o in rows} == {"auto rickshaw"}
    # Idempotent: a re-run finds every proposal already labelled.
    again = _run(run_describe_session({"phrase": "auto rickshaw", "class_name": "autorickshaw"},
                                      session_id=uuid.UUID(s["session_id"]), propose_fn=fake_propose))
    assert again["written"] == 0


@pytest.mark.db
def test_describe_session_dry_run_writes_nothing():
    from services.autolabel.describe import Proposal
    from services.autolabel.describe_session import run_describe_session

    s = _seed(2)
    res = _run(run_describe_session(
        {"phrase": "cart", "class_name": "autorickshaw", "dry_run": True}, session_id=uuid.UUID(s["session_id"]),
        propose_fn=lambda img, phrase, min_conf: [Proposal(bbox=[0, 0, 5, 5], conf=0.9, polygons=[])]))
    assert res["proposed"] == 2 and res["written"] == 0 and res["dry_run"] is True


# ---- http -----------------------------------------------------------------------------------------------

@pytest.mark.db
def test_track_route_queues_a_gpu_job_and_validates():
    from _authutil import auth_headers

    from services.api.main import app

    s = _seed(2)
    with TestClient(app) as c:
        h = auth_headers("annotator")
        r = c.post("/api/tools/video/track", headers=h, json={
            "frame_id": s["frames"][0], "class_name": "sedan",
            "prompts": [{"frame_id": s["frames"][0], "points": [[50, 90]], "labels": [1]}]})
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "sam2_video" and r.json()["status"] == "pending"
        assert c.post("/api/tools/video/track", headers=h, json={
            "frame_id": s["frames"][0], "class_name": "sedan", "prompts": []}).status_code == 400
        assert c.post("/api/tools/video/track", headers=h, json={
            "frame_id": s["frames"][0], "class_name": "not_a_class",
            "prompts": [{"frame_id": s["frames"][0], "points": [[1, 1]]}]}).status_code == 400
        assert c.post("/api/tools/video/track", headers=h, json={
            "frame_id": str(uuid.uuid4()), "class_name": "sedan", "prompts": []}).status_code == 404


@pytest.mark.db
def test_describe_session_route_is_reviewer_only():
    from _authutil import auth_headers

    from core.config import get_settings
    from services.api.main import app

    s = _seed(1)
    body = {"session_id": s["session_id"], "phrase": "hand cart", "class_name": "autorickshaw"}
    with TestClient(app) as c:
        r = c.post("/api/tools/video/describe/session", headers=auth_headers("reviewer"), json=body)
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "describe_session"
        if get_settings().auth.enabled:
            assert c.post("/api/tools/video/describe/session", headers=auth_headers("annotator"),
                          json=body).status_code == 403


@pytest.mark.db
def test_describe_accept_writes_described_objects():
    from _authutil import auth_headers

    from services.api.main import app

    s = _seed(1)
    with TestClient(app) as c:
        h = auth_headers("annotator")
        r = c.post(f"/api/tools/video/frames/{s['frames'][0]}/describe/accept", headers=h, json={
            "phrase": "hand cart", "class_name": "autorickshaw",
            "proposals": [{"bbox": [10, 10, 60, 60], "conf": 0.31, "polygons": [[10, 10, 60, 10, 60, 60, 10, 60]]}]})
        assert r.status_code == 200, r.text
        oid = r.json()["created"][0]
        got = c.get(f"/api/objects/{oid}", headers=h).json()
        assert got["source"] == "described"
        assert got["mask_polygons"]


def _seed_fanout() -> dict:
    """A track of 4 machine 'autorickshaw' boxes (one of them human), plus session confusion candidates."""
    from db.models import Object, Track
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    s = _seed(4)
    onto = get_ontology()
    auto, truck = onto.by_name("autorickshaw").id, onto.by_name("truck").id

    async def go():
        async with get_sessionmaker()() as db:
            tid = uuid.uuid4()
            db.add(Track(track_id=tid, session_id=uuid.UUID(s["session_id"]), class_id=auto, first_ts_ns=0,
                         last_ts_ns=1))
            await db.flush()
            ids = []
            for i, fid in enumerate(s["frames"]):
                oid = uuid.uuid4()
                ids.append(str(oid))
                db.add(Object(object_id=oid, frame_id=uuid.UUID(fid), track_id=tid, class_id=auto,
                              bbox=[10.0, 10.0, 60.0, 60.0], conf=0.8, source="human" if i == 3 else "fused",
                              state="accepted" if i == 3 else "review", attrs={}, provenance={}))
            voted = uuid.uuid4()
            db.add(Object(object_id=voted, frame_id=uuid.UUID(s["frames"][1]), class_id=auto,
                          bbox=[100.0, 100.0, 150.0, 150.0], conf=0.6, source="fused", state="review", attrs={},
                          provenance={"proposals": [{"class_name": "autorickshaw"}, {"class_name": "truck"}]}))
            unvoted = uuid.uuid4()
            db.add(Object(object_id=unvoted, frame_id=uuid.UUID(s["frames"][2]), class_id=auto,
                          bbox=[200.0, 100.0, 250.0, 150.0], conf=0.6, source="fused", state="review", attrs={},
                          provenance={"proposals": [{"class_name": "autorickshaw"}]}))
            await db.commit()
            return {**s, "track_id": str(tid), "track": ids, "voted": str(voted), "unvoted": str(unvoted),
                    "truck": truck}

    return _run(go())


@pytest.mark.db
def test_fanout_preview_apply_and_undo():
    from _authutil import auth_headers

    from db.models import Object, Track
    from db.session import get_sessionmaker
    from services.api.main import app
    from services.autolabel.ontology import get_ontology

    f = _seed_fanout()
    auto = get_ontology().by_name("autorickshaw").id
    with TestClient(app) as c:
        h = auth_headers("reviewer")
        r = c.post("/api/tools/video/fanout/preview", headers=h, json={
            "object_id": f["track"][0], "from_class": "autorickshaw", "to_class": "truck"})
        assert r.status_code == 200, r.text
        p = r.json()
        track = {o["object_id"]: o for o in p["track"]["objects"]}
        assert set(track) == set(f["track"][1:])
        assert track[f["track"][3]]["selectable"] is False          # the person's label
        assert [o["object_id"] for o in p["confusion"]["objects"]] == [f["voted"]]
        assert p["lookalikes"]["objects"] == [] and p["lookalikes"]["reason"]  # nothing embedded here

        r = c.post("/api/tools/video/fanout/apply", headers=h, json={
            "object_id": f["track"][0], "to_class": "truck",
            "track_object_ids": f["track"][1:], "confusion_object_ids": [f["voted"]]})
        assert r.status_code == 200, r.text
        a = r.json()
        assert a["track"] == 2 and a["others"] == 1 and a["skipped_human"] == [f["track"][3]]
        run_id = a["run_id"]

        async def state():
            async with get_sessionmaker()() as db:
                objs = {str(o): (await db.get(Object, uuid.UUID(o))) for o in [*f["track"], f["voted"]]}
                return ({k: (v.class_id, v.source) for k, v in objs.items()},
                        (await db.get(Track, uuid.UUID(f["track_id"]))).class_id)

        objs, tclass = _run(state())
        assert objs[f["track"][1]] == (f["truck"], "propagated")
        assert objs[f["voted"]] == (f["truck"], "human")
        assert objs[f["track"][3]][0] == auto
        assert tclass == f["truck"]

        r = c.post(f"/api/tools/video/fanout/{run_id}/undo", headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["reverted"] == 3
        objs, tclass = _run(state())
        assert objs[f["track"][1]] == (auto, "fused") and objs[f["voted"]] == (auto, "fused")
        assert tclass == auto
        assert c.post(f"/api/tools/video/fanout/{run_id}/undo", headers=h).status_code == 409
