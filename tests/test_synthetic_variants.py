"""Synthetic variants (#16): rain / dusk / night / fog / blur copies of real labelled frames, labels carried over,
and proof that the quarantine still holds now that perturbed frames carry objects.

Until this tool a `perturbed` frame never had objects (services/verdyx/counterfactual.py scored perturbations in
memory against the source frame's labels). Variants now store perturbed frames with copied labels, so every
reader that must not see them is asked here, on rows, the question it answers in production; each pairs with
a positive assertion on the real session so a reader that returned nothing at all would fail rather than pass.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest
from sqlalchemy import func, select

from core.origin import PERTURBED, REAL, SYNTHETIC_SOURCE, SYNTHETIC_STATE

EPOCH = "variants-model-A"
CLASS = "pedestrian"
STATES = ("review", "auto_accept", "accepted", "settled")


# ---- pure --------------------------------------------------------------------------------------------------

def test_only_photometric_perturbations_are_offered():
    from services.synth.variants import check_perturbations
    from services.verdyx.perturb import DEFAULT_STRENGTHS, PHOTOMETRIC

    assert set(PHOTOMETRIC) <= set(DEFAULT_STRENGTHS) and "occlude" not in PHOTOMETRIC
    assert check_perturbations(["rain", "night", "rain"]) == ["rain", "night"]
    with pytest.raises(ValueError, match="photometric"):
        check_perturbations(["occlude"])
    with pytest.raises(ValueError):
        check_perturbations([])


def test_night_darkens_keeps_shape_and_blooms_lights():
    from services.verdyx.perturb import apply

    img = np.full((60, 80, 3), 140, np.uint8)
    img[20:24, 30:34] = 255                                    # a lamp
    out = apply("night", img, seed=3)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[5:15, 5:15].mean() < img[5:15, 5:15].mean() * 0.6
    assert out[22, 32].mean() > out[5, 5].mean() + 60            # the lamp still reads as a light
    assert out[19, 32].mean() > out[5, 5].mean()                 # and glows into its neighbours
    assert np.array_equal(out, apply("night", img, seed=3))
    assert out[..., 2].mean() > out[..., 0].mean()               # colder under street lighting (BGR: B > R)


# ---- database ----------------------------------------------------------------------------------------------

def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _seed_real(db, *, n_frames: int = 2, states=("accepted", "settled"), extra_state: str | None = None) -> dict:
    """A real session whose frames have real images and labels in the given states."""
    import cv2

    from core.storage import get_object_store
    from core.timebase import now_ns
    from db.models import Frame, Object, Track
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = onto.by_name(CLASS).id
    store = get_object_store()
    store.ensure_bucket()
    sid, t0 = uuid.uuid4(), now_ns()
    db.add(DbSession(session_id=sid, vehicle_id="VAR-T", start_ts_ns=t0, end_ts_ns=t0 + 10**9, city="BLR",
                     route=f"variants-{uuid.uuid4().hex[:6]}", sensors={}, ontology_version=onto.version))
    await db.flush()
    tr = Track(session_id=sid, class_id=cid, first_ts_ns=t0, last_ts_ns=t0 + 10**9)
    db.add(tr)
    await db.flush()
    img = np.zeros((90, 160, 3), np.uint8)
    img[:, :, 1] = np.linspace(40, 220, 160).astype(np.uint8)[None, :]
    ok, jpg = cv2.imencode(".jpg", img)
    out = {"session_id": sid, "frames": [], "objects": [], "cid": cid, "track_id": tr.track_id}
    for i in range(n_frames):
        ts = t0 + i * 10**8
        uri = store.put_bytes(f"frames/{sid}/front/{ts}.jpg", jpg.tobytes(), "image/jpeg")
        f = Frame(session_id=sid, ts_ns=ts, cam_id="front", img_uri=uri, width=160, height=90, quality=1.0)
        db.add(f)
        await db.flush()
        out["frames"].append(f.frame_id)
        for st in states:
            o = Object(frame_id=f.frame_id, track_id=tr.track_id, class_id=cid, bbox=[10.0, 20.0, 60.0, 80.0],
                       conf=0.9, source="human" if st == "accepted" else "fused", state=st, rot_deg=7.5,
                       polyline=[[10, 20], [60, 80]], attrs={"occluded": "partial"},
                       mask_uri=f"s3://labeloxav/masks/{sid}/{uuid.uuid4()}.json", mask_encoding="polygon",
                       provenance={"proposals": [{"model_version": EPOCH, "class_name": CLASS}]})
            db.add(o)
            await db.flush()
            out["objects"].append(o.object_id)
        if extra_state:
            db.add(Object(frame_id=f.frame_id, class_id=cid, bbox=[1, 1, 5, 5], conf=0.5, source="fused",
                          state=extra_state))
    await db.commit()
    return out


@pytest.mark.db
def test_variants_copy_labels_exactly_and_quarantine_them():
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.synth.variants import build_variants

    async def seed():
        async with get_sessionmaker()() as db:
            return await _seed_real(db)

    fx = _run(seed())
    res = _run(build_variants(fx["frames"], ["rain", "night"], strengths={"rain": 0.7}, seed=5))
    assert res["frames"] == 4 and res["objects"] == 8 and res["per_perturbation"] == {"rain": 2, "night": 2}

    async def rows():
        async with get_sessionmaker()() as db:
            frames = (await db.execute(select(Frame).where(Frame.frame_id.in_(
                [uuid.UUID(f) for f in res["frame_ids"]])))).scalars().all()
            objs = (await db.execute(select(Object).where(Object.frame_id.in_([f.frame_id for f in frames]))
                                     )).scalars().all()
            sess = await db.get(DbSession, uuid.UUID(res["session_id"]))
            src = (await db.execute(select(Object).where(Object.object_id.in_(fx["objects"])))).scalars().all()
            return frames, objs, sess, src

    frames, objs, sess, src = _run(rows())
    assert sess.origin == PERTURBED and sess.vehicle_id == "perturbed"
    assert all(f.origin == PERTURBED and f.source_frame_id in fx["frames"] for f in frames)
    assert {t for f in frames for t in f.tags if t.startswith("perturb:")} == {"perturb:rain", "perturb:night"}
    by_src = {o.object_id: o for o in src}
    for o in objs:
        s = by_src[uuid.UUID(o.provenance["synthetic"]["source_object_id"])]
        assert o.state == SYNTHETIC_STATE and o.source == SYNTHETIC_SOURCE and o.track_id is None
        # photometric: the geometry is the source's, byte for byte
        assert list(o.bbox) == list(s.bbox) and o.polyline == s.polyline and o.rot_deg == s.rot_deg
        assert o.mask_uri == s.mask_uri and o.attrs == s.attrs and o.class_id == s.class_id
        assert o.provenance["synthetic"]["source_state"] == s.state
    rain = [o for o in objs if o.provenance["synthetic"]["perturbation"] == "rain"]
    assert all(o.provenance["synthetic"]["strength"] == 0.7 for o in rain)


@pytest.mark.db
def test_eligibility_refuses_unfinished_label_sets():
    from db.session import get_sessionmaker
    from services.synth.variants import build_variants, eligibility

    async def go():
        async with get_sessionmaker()() as db:
            ok = await _seed_real(db, n_frames=1)
            rev = await _seed_real(db, n_frames=1, extra_state="review")
            ann = await _seed_real(db, n_frames=1, extra_state="annotate")
            empty = await _seed_real(db, n_frames=1, states=("rejected",))
            ids = [ok["frames"][0], rev["frames"][0], ann["frames"][0], empty["frames"][0], uuid.uuid4()]
            strict = await eligibility(db, ids)
            loose = await eligibility(db, ids, include_review=True)
            return ids, strict, loose

    ids, strict, loose = _run(go())
    assert strict["eligible"] == [str(ids[0])]
    assert "review" in strict["refused"][str(ids[1])] and "annotate" in strict["refused"][str(ids[2])]
    assert strict["refused"][str(ids[3])] == "no trusted labels to carry over"
    assert strict["refused"][str(ids[4])] == "frame not found"
    assert set(loose["eligible"]) == {str(ids[0]), str(ids[1])}
    # a variant of a variant is refused: only real frames are varied
    res = _run(build_variants([ids[0]], ["fog"]))
    again = _run(build_variants([uuid.UUID(res["frame_ids"][0])], ["fog"]))
    assert again["frames"] == 0 and "only real frames" in next(iter(again["refused"].values()))


@pytest.mark.db
def test_revert_removes_exactly_what_the_run_made():
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.synth.variants import build_variants, revert_variants

    async def seed():
        async with get_sessionmaker()() as db:
            return await _seed_real(db)

    fx = _run(seed())
    res = _run(build_variants(fx["frames"], ["dusk"]))

    async def revert():
        async with get_sessionmaker()() as db:
            # a real frame id slipped into the list is never touched
            out = await revert_variants(db, res["frame_ids"] + [str(fx["frames"][0])], res["session_id"])
        async with get_sessionmaker()() as db:
            left = (await db.execute(select(func.count()).select_from(Frame).where(
                Frame.frame_id.in_([uuid.UUID(f) for f in res["frame_ids"]])))).scalar_one()
            real = (await db.execute(select(func.count()).select_from(Object).where(
                Object.object_id.in_(fx["objects"])))).scalar_one()
            sess = await db.get(DbSession, uuid.UUID(res["session_id"]))
            return out, left, real, sess

    out, left, real, sess = _run(revert())
    assert out == {"removed": 2, "skipped": 1, "objects_left": 0}
    assert left == 0 and real == len(fx["objects"]) and sess is None


# ---- the quarantine, reader by reader -----------------------------------------------------------------------

async def _seed_pair(db) -> dict:
    """A real session and a perturbed one built from it, the perturbed objects exactly as variants writes them."""
    from core.timebase import now_ns
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = onto.by_name(CLASS).id
    tag = uuid.uuid4().hex[:6]
    out: dict = {"cid": cid, "frames": {REAL: [], PERTURBED: []}, "objects": {REAL: [], PERTURBED: []}}
    sids = {REAL: uuid.uuid4(), PERTURBED: uuid.uuid4()}
    for origin, sid in sids.items():
        db.add(DbSession(session_id=sid, vehicle_id=f"Q-{tag}", start_ts_ns=0, end_ts_ns=1, city="BLR",
                         route=f"quarantine-{tag}", sensors={}, ontology_version=onto.version, origin=origin))
        out[f"{origin}_session"] = sid
    await db.flush()
    prov = {"proposals": [{"verdict": "agree", "class_name": CLASS, "model_version": EPOCH, "conf": 0.8,
                           "path": "path_a"}]}
    for i in range(4):
        rf = Frame(frame_id=uuid.uuid4(), session_id=sids[REAL], ts_ns=now_ns() + i, cam_id="front", width=1920,
                   height=1080, img_uri=f"s3://x/{tag}-real-{i}.jpg", origin=REAL)
        pf = Frame(frame_id=uuid.uuid4(), session_id=sids[PERTURBED], ts_ns=rf.ts_ns, cam_id="front", width=1920,
                   height=1080, img_uri=f"s3://x/{tag}-rain-{i}.jpg", origin=PERTURBED, source_frame_id=rf.frame_id,
                   tags=["perturbed", "perturb:rain"])
        db.add_all([rf, pf])
        await db.flush()
        out["frames"][REAL].append(rf.frame_id)
        out["frames"][PERTURBED].append(pf.frame_id)
        for st in STATES:
            o = Object(frame_id=rf.frame_id, class_id=cid, bbox=[100, 100, 400, 400], conf=0.9,
                       source="human" if st == "accepted" else "fused", state=st, provenance=prov, attrs={})
            db.add(o)
            await db.flush()
            out["objects"][REAL].append(o.object_id)
            c = Object(frame_id=pf.frame_id, class_id=cid, bbox=[100, 100, 400, 400], conf=0.9,
                       source=SYNTHETIC_SOURCE, state=SYNTHETIC_STATE,
                       provenance={**prov, "synthetic": {"generator": "variants", "perturbation": "rain",
                                                         "source_object_id": str(o.object_id), "source_state": st}},
                       attrs={})
            db.add(c)
            await db.flush()
            out["objects"][PERTURBED].append(c.object_id)
    await db.commit()
    return out


@pytest.mark.db
class TestPerturbedQuarantine:
    async def test_settlement_population(self):
        from db.models import Frame, Object
        from db.session import get_sessionmaker
        from services.labelops.settlement import stratum_population

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            pop = await stratum_population(db, fx["cid"], CLASS, EPOCH)
            real_review = (await db.execute(select(func.count()).select_from(Object).join(Frame).where(
                Frame.origin == REAL, Object.state == "review", Object.class_id == fx["cid"]))).scalar_one()
        assert 4 <= pop <= real_review

    async def test_control_sample_seed(self):
        from db.models import ControlSample
        from db.session import get_sessionmaker
        from services.govern.control_sample import seed_from_recent_auto_accepts

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            await seed_from_recent_auto_accepts(db, limit=100_000, rate=1.0)
            mirrored = set((await db.execute(select(ControlSample.object_id))).scalars().all())
        assert not (mirrored & set(fx["objects"][PERTURBED])) and mirrored & set(fx["objects"][REAL])

    async def test_precision_draw(self):
        from db.models import Object
        from db.session import get_sessionmaker
        from services.labelops.precision_batch import build_precision_batch

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            res = await build_precision_batch(db, classes=[CLASS], target=100_000, min_side_px=1)
            stamped = set((await db.execute(select(Object.object_id).where(
                Object.provenance["flywheel"]["cycle_id"].astext == res["batch_id"]))).scalars().all())
        assert stamped and not (stamped & set(fx["objects"][PERTURBED]))

    async def test_gold_build_and_so_the_counterfactual_never_score_a_copy(self):
        """counterfactual_eval scores a gold set's objects; gold is built by `_fetch_gold`. A copy that cannot
        enter gold cannot be scored as truth by the counterfactual either."""
        from db.session import get_sessionmaker
        from services.training.gold import GoldSpec, _fetch_gold

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
        assert len(await _fetch_gold(GoldSpec(session_id=str(fx[f"{REAL}_session"])))) == 4
        assert await _fetch_gold(GoldSpec(session_id=str(fx[f"{PERTURBED}_session"]))) == []

    async def test_trainset_excludes_perturbed_by_default(self):
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.training.dataset_builder import BuildSpec, _select

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            route = (await db.get(DbSession, fx[f"{REAL}_session"])).route
        off = await _select(BuildSpec(route_prefix=route, conf_floor=0.0))
        assert {c["frame_id"] for c in off} == {str(f) for f in fx["frames"][REAL]}

    async def test_trainset_opt_in_takes_variants_and_never_validates_on_them(self):
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.training.dataset_builder import BuildSpec, _partition, _select

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            route = (await db.get(DbSession, fx[f"{REAL}_session"])).route
        on = await _select(BuildSpec(route_prefix=route, conf_floor=0.0, include_synthetic=True))
        pert = {str(f) for f in fx["frames"][PERTURBED]}
        assert pert <= {c["frame_id"] for c in on}
        by_frame: dict[str, list[dict]] = {}
        for c in on:
            by_frame.setdefault(c["frame_id"], []).append(c)
        for i in range(8):
            by_frame[f"extra-{i}"] = [{"frame_id": f"extra-{i}", "session_id": f"extra-session-{i % 2}",
                                       "origin": REAL, "source_frame_id": None, "gold": False, "class_id": fx["cid"]}]
        val, kept, dropped = _partition(by_frame, set(), BuildSpec(val_frac=0.5, seed=1))
        assert not (val & pert) and kept + dropped == 4

    async def test_embedding_queue(self):
        from db.models import Frame, Object
        from db.session import get_sessionmaker
        from services.intelligence.embed.pending import (
            frame_needs_embedding,
            object_needs_embedding,
        )

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            f_p = (await db.execute(select(func.count()).select_from(Frame).where(
                Frame.frame_id.in_(fx["frames"][PERTURBED]), frame_needs_embedding()))).scalar_one()
            o_p = (await db.execute(select(func.count()).select_from(Object).where(
                Object.object_id.in_(fx["objects"][PERTURBED]), object_needs_embedding()))).scalar_one()
            f_r = (await db.execute(select(func.count()).select_from(Frame).where(
                Frame.frame_id.in_(fx["frames"][REAL]), frame_needs_embedding()))).scalar_one()
        assert f_p == 0 and o_p == 0 and f_r == 4

    async def test_explorer_predicate(self):
        from db.models import Frame
        from db.session import get_sessionmaker
        from services.explore.query import frame_clauses

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            sid = str(fx[f"{PERTURBED}_session"])
            hidden = (await db.execute(select(func.count()).select_from(Frame).where(
                *frame_clauses({"session_id": sid})))).scalar_one()
            shown = (await db.execute(select(func.count()).select_from(Frame).where(
                *frame_clauses({"session_id": sid, "include_synthetic": True})))).scalar_one()
        assert hidden == 0 and shown == 4

    async def test_blind_audit_never_picks_a_variant(self):
        from db.models import InferenceRun, LabelProject, ModelRegistry, Prediction
        from db.session import get_sessionmaker
        from services.verdyx.blind_audit import audit_frame_ids, seed_audit

        async with get_sessionmaker()() as db:
            fx = await _seed_pair(db)
            mv = f"variants-{uuid.uuid4().hex[:8]}"
            db.add(ModelRegistry(model_version=mv, task="detection"))
            await db.flush()
            run = InferenceRun(model_version=mv, gold_id=None, status="complete", frame_count=8, params={},
                               code_sha="0" * 40)
            db.add(run)
            await db.flush()
            for fid in fx["frames"][REAL] + fx["frames"][PERTURBED]:
                db.add(Prediction(run_id=run.run_id, frame_id=fid, class_id=fx["cid"], bbox=[0.0, 0.0, 100.0, 100.0],
                                  conf=0.9))
            project = LabelProject(name=f"variants-{uuid.uuid4().hex[:6]}", modality="image")
            db.add(project)
            await db.commit()
            res = await seed_audit(db, run_id=str(run.run_id), n_frames=100, project_id=str(project.project_id))
            assert "error" not in res, res
            picked = set(await audit_frame_ids(db, uuid.UUID(res["audit_id"])))
        assert picked == set(fx["frames"][REAL])

    async def test_a_person_cannot_promote_a_copy_out_of_the_synthetic_state(self):
        from services.review_policy import ReviewStateError, state_for

        with pytest.raises(ReviewStateError):
            state_for(None, SYNTHETIC_STATE, "admin", "review")


# ---- routes -----------------------------------------------------------------------------------------------

@pytest.mark.db
def test_variant_routes_preview_build_runs_and_revert():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from db.models import ErrorCandidate
    from db.session import get_sessionmaker
    from services.api.main import app
    from services.tools.registry import run_job

    async def seed():
        async with get_sessionmaker()() as db:
            fx = await _seed_real(db, n_frames=2)
            db.add(ErrorCandidate(object_id=fx["objects"][0], kind="confident_learning", score=0.9,
                                  status="confirmed_error", detail={}))
            await db.commit()
            return fx

    fx = _run(seed())
    f0 = str(fx["frames"][0])
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        names = [p["name"] for p in c.get("/api/tools/sensors/variants/perturbations", headers=ann).json()
                 ["perturbations"]]
        assert "night" in names and "occlude" not in names
        hard = c.get("/api/tools/sensors/variants/hard-frames?source=errors&limit=500", headers=ann).json()["frames"]
        mine = [h for h in hard if h["frame_id"] == f0]
        assert mine and mine[0]["eligible"] and mine[0]["why"] == "confirmed label error"
        pv = c.get(f"/api/tools/sensors/variants/preview.jpg?frame_id={f0}&perturbation=fog&strength=0.8",
                   headers=ann)
        assert pv.status_code == 200 and pv.content[:2] == b"\xff\xd8"
        assert c.get(f"/api/tools/sensors/variants/preview.jpg?frame_id={f0}&perturbation=occlude",
                     headers=ann).status_code == 422
        body = {"frame_ids": [str(f) for f in fx["frames"]], "perturbations": ["rain", "motion_blur"]}
        assert c.post("/api/tools/sensors/variants", headers=ann, json=body).status_code == 403
        r = c.post("/api/tools/sensors/variants", headers=rev, json=body)
        assert r.status_code == 200 and r.json()["variants"] == 4, r.text
        jid = r.json()["job_id"]
    _run(run_job(uuid.UUID(jid)))
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        runs = c.get("/api/tools/sensors/variants/runs", headers=ann).json()["runs"]
        run = next(x for x in runs if x["job_id"] == jid)
        assert run["status"] == "done" and run["frames"] == 4 and run["objects"] == 8
        summ = c.get("/api/tools/sensors/variants/summary", headers=ann).json()
        assert summ["frames"] >= 4 and summ["by_perturbation"].get("motion_blur", 0) >= 2
        assert c.post(f"/api/tools/sensors/variants/runs/{jid}/revert", headers=ann).status_code == 403
        rv = c.post(f"/api/tools/sensors/variants/runs/{jid}/revert", headers=rev)
        assert rv.status_code == 200 and rv.json()["removed"] == 4
        assert c.post(f"/api/tools/sensors/variants/runs/{jid}/revert", headers=rev).status_code == 409


@pytest.mark.db
def test_variants_of_two_projects_land_in_two_sessions():
    """A members-only project's variants must not share a session with another project's."""
    from db.models import LabelProject
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.synth.variants import build_variants, revert_variants

    async def seed():
        async with get_sessionmaker()() as db:
            a = await _seed_real(db, n_frames=1)
            b = await _seed_real(db, n_frames=1)
            pa, pb = LabelProject(name=f"va-{uuid.uuid4().hex[:6]}", modality="image"), \
                LabelProject(name=f"vb-{uuid.uuid4().hex[:6]}", modality="image")
            db.add_all([pa, pb])
            await db.flush()
            (await db.get(DbSession, a["session_id"])).project_id = pa.project_id
            (await db.get(DbSession, b["session_id"])).project_id = pb.project_id
            await db.commit()
            return a["frames"][0], b["frames"][0], pa.project_id, pb.project_id

    fa, fb, pa, pb = _run(seed())
    res = _run(build_variants([fa, fb], ["fog"]))
    assert res["frames"] == 2 and len(res["session_ids"]) == 2

    async def check():
        async with get_sessionmaker()() as db:
            projects = {(await db.get(DbSession, uuid.UUID(s))).project_id for s in res["session_ids"]}
            out = await revert_variants(db, res["frame_ids"], res["session_ids"])
            left = [await db.get(DbSession, uuid.UUID(s)) for s in res["session_ids"]]
            return projects, out, left

    projects, out, left = _run(check())
    assert projects == {pa, pb} and out["removed"] == 2 and left == [None, None]
