"""A copy-paste build end to end on the test database and the object store: plan, build, quarantine, revert.

The fixture is one real session with two frames: a donor frame carrying a human-accepted pedestrian with a
polygon mask, and a background frame with a drivable mask and one auto-accepted label. The build must
compose from exactly those, write everything as synthetic, and revert to nothing.
"""

from __future__ import annotations

import json
import uuid

import cv2
import numpy as np
import pytest
from sqlalchemy import delete, func, select

from core.config import get_settings
from core.origin import REAL, SYNTHETIC, SYNTHETIC_SOURCE, SYNTHETIC_STATE
from core.storage import get_object_store
from db.models import AgentRun, DrivableMask, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.synth.copy_paste import BATCH_KIND, KIND, build, plan_build

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")

W, H = 640, 480


def _jpeg(img: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return enc.tobytes()


async def _fixture(db, *, backgrounds: int = 3) -> dict:
    onto = get_ontology()
    ped = onto.by_name("pedestrian").id
    sedan = onto.by_name("sedan").id
    store = get_object_store()
    tag = uuid.uuid4().hex[:8]
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id=f"SYN-{tag}", start_ts_ns=0, end_ts_ns=1, city="BLR",
                     route=f"synth-fixture-{tag}", sensors={}, ontology_version=onto.version, origin=REAL))
    await db.flush()

    # Donor: a bright disc on a dark frame, human-accepted, polygon mask in the store.
    dimg = np.full((H, W, 3), 30, dtype=np.uint8)
    cv2.circle(dimg, (320, 300), 50, (220, 220, 230), -1)
    d_fid, d_oid = uuid.uuid4(), uuid.uuid4()
    d_uri = store.put_bytes(f"frames/{sid}/front/{1}.jpg", _jpeg(dimg), "image/jpeg")
    circle = [[float(320 + 50 * np.cos(t)), float(300 + 50 * np.sin(t))] for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    poly = [v for pt in circle for v in pt]
    m_uri = store.put_bytes(f"masks/{sid}/{d_fid}/{d_oid}.json",
                            json.dumps({"encoding": "polygon", "polygons": [poly], "height": H, "width": W}).encode(),
                            "application/json")
    db.add(Frame(frame_id=d_fid, session_id=sid, ts_ns=1, cam_id="front", img_uri=d_uri, width=W, height=H,
                 origin=REAL))
    await db.flush()
    db.add(Object(object_id=d_oid, frame_id=d_fid, class_id=ped, bbox=[270.0, 250.0, 370.0, 350.0], conf=1.0,
                  source="human", state="accepted", mask_uri=m_uri, mask_encoding="polygon", attrs={},
                  provenance={}))

    # Backgrounds: a gradient road with a drivable trapezoid and one auto-accepted sedan up in the corner.
    bg_ids = []
    for i in range(backgrounds):
        bimg = np.zeros((H, W, 3), dtype=np.uint8)
        bimg[..., 0] = np.linspace(60, 180, W, dtype=np.uint8)[None, :]
        bimg[..., 1] = np.linspace(80, 140, H, dtype=np.uint8)[:, None]
        bimg[..., 2] = 100
        fid = uuid.uuid4()
        uri = store.put_bytes(f"frames/{sid}/front/{10 + i}.jpg", _jpeg(bimg), "image/jpeg")
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=10 + i, cam_id="front", img_uri=uri, width=W, height=H,
                     origin=REAL, selected=True))
        await db.flush()
        drivable = [[200.0, 250.0, 440.0, 250.0, 640.0, 470.0, 0.0, 470.0]]
        dm_uri = store.put_bytes(f"masks/drivable/{sid}/{fid}.json",
                                 json.dumps({"classes": {"drivable": drivable, "non_drivable": [], "fallback": []},
                                             "width": W, "height": H}).encode(), "application/json")
        db.add(DrivableMask(frame_id=fid, mask_uri=dm_uri, coverage={"drivable": 0.4}, source="proposed",
                            model_version="fixture"))
        db.add(Object(frame_id=fid, class_id=sedan, bbox=[10.0, 10.0, 90.0, 60.0], conf=0.9,
                      source="auto_accept", state="auto_accept", attrs={}, provenance={}))
        bg_ids.append(fid)
    await db.commit()
    return {"session_id": sid, "donor_object": d_oid, "backgrounds": bg_ids, "ped": ped, "sedan": sedan}


@pytest.fixture(autouse=True)
async def _clean_slate():
    """The build draws backgrounds from the whole test database, so a fixture left behind by an
    earlier failed run would be a background for this one. Purge fixtures and their synthetic output
    on both sides of every test; frame, object and agent rows follow the session by cascade."""
    async def purge():
        async with get_sessionmaker()() as db:
            await db.execute(delete(DbSession).where(
                (DbSession.route.like("synth-fixture-%")) | (DbSession.origin == SYNTHETIC)))
            await db.execute(delete(AgentRun).where(AgentRun.kind.in_((KIND, BATCH_KIND)),
                                                    AgentRun.created_by == "test"))
            await db.commit()
    await purge()
    yield
    await purge()


@requires_infra
class TestBuild:
    async def test_plan_names_donors_backgrounds_and_refusals(self):
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            plan = await plan_build(db, class_names=["pedestrian", "cattle", "not_a_class"], n_frames=10)
        by = {c["class_name"]: c for c in plan["classes"]}
        assert by["pedestrian"]["donors"] >= 1
        assert by["not_a_class"]["donors"] == 0 and "ontology" in by["not_a_class"]["reason"]
        assert plan["backgrounds"] >= len(fx["backgrounds"])
        assert plan["feasible"] and plan["batches"] == 1

    async def test_build_writes_only_synthetic_rows_and_reverts_to_nothing(self):
        from services.agent.runs import revert_run

        async with get_sessionmaker()() as db:
            fx = await _fixture(db, backgrounds=3)
            run_id = uuid.uuid4()
            db.add(AgentRun(run_id=run_id, kind=KIND, scope={}, status="running", policy={}, counts={},
                            changes={}, critic={}, created_by="test"))
            await db.commit()

        report = await build(run_id, class_names=["pedestrian"], n_frames=3, seed="t", created_by="test")
        assert report["status"] == "committed", report
        assert report["frames"] >= 1, report
        assert report["pasted"] == {"pedestrian": report["frames"]}
        synth_sid = uuid.UUID(report["session_id"])

        async with get_sessionmaker()() as db:
            s = await db.get(DbSession, synth_sid)
            assert s.origin == SYNTHETIC and s.vehicle_id == "synthetic"
            frames = (await db.execute(select(Frame).where(Frame.session_id == synth_sid))).scalars().all()
            assert len(frames) == report["frames"]
            for f in frames:
                assert f.origin == SYNTHETIC and f.source_frame_id in fx["backgrounds"]
                assert f.img_uri.endswith(f"/{synth_sid}/front/{f.ts_ns}.jpg")
            objs = (await db.execute(select(Object).join(Frame).where(Frame.session_id == synth_sid))).scalars().all()
            # Every object is synthetic in both columns; each frame carries the copied sedan and the paste.
            assert objs and all(o.state == SYNTHETIC_STATE and o.source == SYNTHETIC_SOURCE for o in objs)
            pasted = [o for o in objs if o.provenance["synthetic"]["pasted"]]
            copied = [o for o in objs if not o.provenance["synthetic"]["pasted"]]
            assert len(pasted) == len(frames) and len(copied) == len(frames)
            assert all(o.class_id == fx["ped"] and o.mask_encoding == "polygon" for o in pasted)
            assert all(o.class_id == fx["sedan"] and o.provenance["synthetic"]["source_state"] == "auto_accept"
                       for o in copied)
            # The paste stands on the drivable trapezoid, below its top row.
            assert all(o.bbox[3] > 250 for o in pasted)
            # The pasted mask blob is readable and describes the same box.
            store = get_object_store()
            blob = json.loads(store.get_bytes(pasted[0].mask_uri))
            assert blob["encoding"] == "polygon" and blob["polygons"]
            img = cv2.imdecode(np.frombuffer(store.get_bytes(frames[0].img_uri), np.uint8), cv2.IMREAD_COLOR)
            assert img.shape == (H, W, 3)

            batches = (await db.execute(select(AgentRun).where(AgentRun.kind == BATCH_KIND,
                                                              AgentRun.scope["parent_run_id"].astext == str(run_id)))).scalars().all()
            assert len(batches) == 1 and batches[0].status == "committed"
            assert set(batches[0].changes["frame_ids"]) == {str(f.frame_id) for f in frames}
            parent = await db.get(AgentRun, run_id)
            assert parent.status == "committed" and parent.changes["child_runs"] == [str(batches[0].run_id)]

            # The real session is untouched.
            real_n = (await db.execute(select(func.count()).select_from(Frame)
                                       .where(Frame.session_id == fx["session_id"]))).scalar_one()
            assert real_n == 1 + len(fx["backgrounds"])

            res = await revert_run(db, run_id)
            assert res["children"] == 1 and res["reverted"] == len(frames)
            assert (await db.get(DbSession, synth_sid)) is None
            left = (await db.execute(select(func.count()).select_from(Frame)
                                     .where(Frame.session_id == synth_sid))).scalar_one()
            assert left == 0
            assert not store.exists(frames[0].img_uri)
            assert not store.exists(pasted[0].mask_uri)
            # The donor's own mask blob is shared by reference and must survive the revert.
            donor = await db.get(Object, fx["donor_object"])
            assert store.exists(donor.mask_uri)

    async def test_build_refuses_a_class_without_donors(self):
        async with get_sessionmaker()() as db:
            await _fixture(db, backgrounds=1)
            run_id = uuid.uuid4()
            db.add(AgentRun(run_id=run_id, kind=KIND, scope={}, status="running", policy={}, counts={},
                            changes={}, critic={}, created_by="test"))
            await db.commit()
        report = await build(run_id, class_names=["not_a_class"], n_frames=3, seed="t", created_by="test")
        assert report["status"] == "refused" and "donor" in report["reason"]
        async with get_sessionmaker()() as db:
            run = await db.get(AgentRun, run_id)
            assert run.status == "refused"
            n = (await db.execute(select(func.count()).select_from(DbSession)
                                  .where(DbSession.origin == SYNTHETIC,
                                         DbSession.sensors["synthetic"]["agent_run_id"].astext == str(run_id)))).scalar_one()
            assert n == 0
