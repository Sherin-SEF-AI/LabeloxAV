"""Annotation-agent tests. Pure-function coverage of the policy, all five critic checks (calibration and
LiDAR deps monkeypatched so they are deterministic without infra), and the NL parser; plus DB-backed
integration tests for plan/commit/revert, the human-safety invariant, and the temporal critic firing on a
real class-flipping track. Mirrors the run_async + TestClient pattern used across the suite.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


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


# ---- fakes for the pure critic tests -------------------------------------------------------------

class _Cls:
    def __init__(self, name, l1):
        self.name = name
        self.l1 = l1


class _Onto:
    _M = {1: ("sedan", "four_wheeler"), 2: ("rider", "vru"), 3: ("pedestrian", "vru"),
          4: ("motorcycle", "two_wheeler"), 5: ("sky", "region")}

    def by_id(self, i):
        n, l1 = self._M[int(i)]
        return _Cls(n, l1)


class _Obj:
    def __init__(self, oid, cid, bbox, tid=None):
        self.object_id = oid
        self.class_id = cid
        self.bbox = bbox
        self.track_id = tid


def _ctx(objs, **kw):
    from services.agent.critic import CriticContext

    return CriticContext(onto=_Onto(), cam_id="cam_front", width=1920, height=1080,
                         frame_objects=objs, dynamics=kw.get("dynamics", {}),
                         track_history=kw.get("track_history", {}), cloud_xyz=kw.get("cloud_xyz"))


# ---- policy ---------------------------------------------------------------------------------------

def test_policy_bands():
    from services.agent.policy import PolicyThresholds, decide

    th = PolicyThresholds()
    assert decide(0.97, True, True, True, th).action == "auto_accept"
    assert decide(0.97, False, True, True, th).action == "review"   # no agreement
    assert decide(0.97, True, True, False, th).action == "review"   # critic veto
    assert decide(0.97, True, False, True, th).action == "review"   # quality veto
    assert decide(0.80, True, True, True, th).action == "review"    # mid band
    assert decide(0.40, True, True, True, th).action == "annotate"  # below floor


def test_policy_agreement_optional():
    from services.agent.policy import PolicyThresholds, decide

    th = PolicyThresholds(require_agreement=False)
    assert decide(0.97, False, True, True, th).action == "auto_accept"


# ---- critic ---------------------------------------------------------------------------------------

def test_critic_relationship():
    from services.agent.critic import critique_frame

    rider_alone = _Obj("r1", 2, [100, 500, 140, 700])
    rider_ok = _Obj("r2", 2, [300, 500, 340, 700])
    moto = _Obj("m1", 4, [295, 620, 350, 760])
    v = critique_frame(_ctx([rider_alone, rider_ok, moto]))
    assert v["r1"].checks["relationship"] == "flag"
    assert v["r2"].checks["relationship"] == "pass"   # a two-wheeler under it


def test_critic_motion():
    from services.agent.critic import critique_frame

    ped_fast = _Obj("p1", 3, [500, 900, 540, 1040])
    ped_ok = _Obj("p2", 3, [600, 900, 640, 1040])
    car_flying = _Obj("c1", 1, [700, 900, 900, 1040])
    v = critique_frame(_ctx([ped_fast, ped_ok, car_flying],
                            dynamics={"p1": {"speed_kmh": 60.0}, "p2": {"speed_kmh": 4.0},
                                      "c1": {"speed_kmh": 260.0}}))
    assert v["p1"].checks["motion"] == "flag"   # 60 km/h VRU
    assert v["p2"].checks["motion"] == "pass"   # 4 km/h VRU
    assert v["c1"].checks["motion"] == "flag"   # 260 km/h anything


def test_critic_temporal():
    from services.agent.critic import critique_frame

    flip = _Obj("t1", 1, [10, 10, 50, 50], tid="TK")
    steady = _Obj("t2", 1, [10, 10, 50, 50], tid="ST")
    hist = {"TK": [(100, 1, 20, 20), (200, 4, 25, 25)],           # sedan -> motorcycle
            "ST": [(100, 1, 20, 20), (200, 1, 24, 24)]}           # stable
    v = critique_frame(_ctx([flip, steady], track_history=hist))
    assert v["t1"].checks["temporal"] == "flag"
    assert v["t2"].checks["temporal"] == "pass"


def test_critic_temporal_teleport():
    from services.agent.critic import critique_frame

    tp = _Obj("t1", 1, [10, 10, 50, 50], tid="TK")
    # jump ~full frame diagonal between consecutive frames
    hist = {"TK": [(100, 1, 20, 20), (200, 1, 1900, 1060)]}
    v = critique_frame(_ctx([tp], track_history=hist))
    assert v["t1"].checks["temporal"] == "flag"


def test_critic_geometric(monkeypatch):
    import services.lidar.project as proj

    # a vehicle looking up (dz>0, oz>0) -> no ground point -> flag
    monkeypatch.setattr(proj, "camera_ray_to_ego",
                        lambda u, v, cam, w, h: {"origin": np.array([0, 0, 1.5]), "direction": np.array([0, 0, 0.1])})
    from services.agent.critic import critique_frame

    car = _Obj("c1", 1, [800, 100, 1000, 300])
    stuff = _Obj("s1", 5, [0, 0, 200, 200])   # region class -> geometric skip
    v = critique_frame(_ctx([car, stuff]))
    assert v["c1"].checks["geometric"] == "flag"
    assert v["s1"].checks["geometric"] == "skip"

    # looking down -> ground point ahead -> pass
    monkeypatch.setattr(proj, "camera_ray_to_ego",
                        lambda u, v, cam, w, h: {"origin": np.array([0, 0, 1.5]), "direction": np.array([0.3, 0, -0.5])})
    v2 = critique_frame(_ctx([_Obj("c2", 1, [800, 900, 1000, 1040])]))
    assert v2["c2"].checks["geometric"] == "pass"


def test_critic_cross_modal(monkeypatch):
    import services.lidar.detect3d.lift as lift

    monkeypatch.setattr(lift, "frustum_indices", lambda cloud, bbox, cam, w, h, pad=0: np.array([], dtype=int))
    from services.agent.critic import critique_frame

    big_empty = _Obj("c1", 1, [400, 400, 900, 900])   # >2% of frame, 0 returns -> flag
    small = _Obj("c2", 1, [10, 10, 40, 40])           # tiny box -> pass (few returns expected)
    cloud = np.zeros((10, 3), dtype=float)
    v = critique_frame(_ctx([big_empty, small], cloud_xyz=cloud))
    assert v["c1"].checks["cross_modal"] == "flag"
    assert v["c2"].checks["cross_modal"] == "pass"


def test_critic_skips_without_inputs():
    from services.agent.critic import critique_frame

    # no dynamics / no track / no cloud -> those checks skip, object is not vetoed by them
    o = _Obj("o1", 1, [800, 900, 1000, 1040])
    v = critique_frame(_ctx([o]))
    assert v["o1"].checks["motion"] == "skip"
    assert v["o1"].checks["cross_modal"] == "skip"
    assert v["o1"].checks["temporal"] == "skip"


# ---- NL parser (uses the real ontology; no infra) -------------------------------------------------

def test_nl_parse_actions_and_scope():
    from services.agent.nl import parse_command
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    i = parse_command("auto-accept the two-wheelers above 0.9", onto)
    assert i.action == "accept" and i.conf_min == 0.9 and len(i.class_ids) >= 5
    assert parse_command("how many pedestrians here", onto).action == "find"
    assert parse_command("double-check all riders", onto).action == "reconcile"
    assert parse_command("undo that", onto).action == "revert"
    assert parse_command("what would you do with vehicles over 95%", onto).conf_min == 0.95


def test_nl_plurals_and_l1_groups():
    from services.agent.nl import parse_command
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    peds = parse_command("count pedestrians", onto)
    assert "pedestrian" in peds.class_names          # plural matched singular class
    veh = parse_command("accept all vehicles", onto)
    assert len(veh.class_ids) > 20                   # every vehicle l1, not a hardcoded few


# ---- DB integration -------------------------------------------------------------------------------

async def _seed_ontology(db):
    """Mirror the file ontology into the DB so object.class_id FKs resolve (the fresh test DB has none)."""
    from db.models import OntologyClass, OntologyVersion
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is not None:
        return
    db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
    await db.flush()
    for c in onto.classes:
        db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                             india=c.india, map_to={}))
    await db.flush()


def _vehicle_ids(n=2):
    from services.autolabel.ontology import get_ontology

    return [c.id for c in get_ontology().classes if c.l1 == "four_wheeler"][:n]


async def _seed_frame_with_objects(objects: list[dict]):
    """Seed a session + one frame + machine objects. objects: list of {class_id, conf, state, source,
    provenance, bbox}. Returns (session_id, frame_id, [object_ids])."""
    import cv2

    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    start = now_ns()
    img = np.random.default_rng(3).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{start}.jpg", buf.tobytes(), "image/jpeg")
    oids = []
    async with maker() as db:
        await _seed_ontology(db)
        db.add(DbSession(session_id=sid, vehicle_id="AGENT-01", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri=uri,
                     width=640, height=480, quality=0.9))
        for o in objects:
            oid = uuid.uuid4()
            oids.append(oid)
            db.add(Object(object_id=oid, frame_id=fid, class_id=o["class_id"], bbox=o["bbox"],
                          conf=o["conf"], source=o["source"], state=o["state"],
                          provenance=o.get("provenance", {}), attrs={}, version=1))
        await db.commit()
    return sid, fid, oids


async def _obj_state(oid):
    from db.models import Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        o = await db.get(Object, oid)
        return (o.state, o.source)


@requires_infra
def test_plan_commit_revert_cycle():
    from services.agent.frame_agent import commit_frame, plan_frame
    from services.agent.policy import PolicyThresholds
    from services.agent.runs import revert_run
    from db.session import get_sessionmaker

    # two confident, agreeing machine objects
    prov = {"agreement": True, "quality_flags": []}
    vid = _vehicle_ids(1)[0]
    _sid, fid, oids = run_async(_seed_frame_with_objects([
        {"class_id": vid, "conf": 0.98, "state": "review", "source": "fused", "provenance": prov, "bbox": [10, 300, 60, 460]},
        {"class_id": vid, "conf": 0.97, "state": "review", "source": "fused", "provenance": prov, "bbox": [80, 300, 130, 460]},
    ]))

    async def _flow():
        async with get_sessionmaker()() as db:
            plan = await plan_frame(db, fid, PolicyThresholds())
            assert plan["counts"]["auto_accept"] == 2
        async with get_sessionmaker()() as db:
            run = await commit_frame(db, fid, PolicyThresholds(), created_by="tester")
            assert run["applied"] == 2
        st = [await _obj_state(o) for o in oids]
        assert all(s == ("auto_accept", "auto_accept") for s in st)
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(run["run_id"]))
            assert rev["reverted"] == 2
        st2 = [await _obj_state(o) for o in oids]
        assert all(s == ("review", "fused") for s in st2)

    run_async(_flow())


@requires_infra
def test_agent_never_touches_human():
    from services.agent.frame_agent import commit_frame
    from services.agent.policy import PolicyThresholds
    from db.session import get_sessionmaker

    _sid, fid, oids = run_async(_seed_frame_with_objects([
        {"class_id": _vehicle_ids(1)[0], "conf": 0.99, "state": "accepted", "source": "human",
         "provenance": {"agreement": True}, "bbox": [10, 300, 60, 460]},
    ]))

    async def _flow():
        async with get_sessionmaker()() as db:
            run = await commit_frame(db, fid, PolicyThresholds(require_agreement=False), created_by="t")
            assert run["applied"] == 0   # human object out of scope
        assert (await _obj_state(oids[0])) == ("accepted", "human")

    run_async(_flow())


@requires_infra
def test_temporal_critic_on_real_track():
    """A real track that flips class across two frames must be flagged by the DB-backed temporal check."""
    import cv2

    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.agent.frame_agent import plan_frame

    store = get_object_store()
    store.ensure_bucket()

    async def _flow():
        maker = get_sessionmaker()
        sid, f1, f2, tid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        start = now_ns()
        img = np.random.default_rng(5).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
        _ok, buf = cv2.imencode(".jpg", img)
        uri = store.put_bytes(f"frames/{sid}/cam_f/{start}.jpg", buf.tobytes(), "image/jpeg")
        ts2 = start + seconds_to_ns(1) // 10
        cid_a, cid_b = _vehicle_ids(2)
        async with maker() as db:
            await _seed_ontology(db)
            db.add(DbSession(session_id=sid, vehicle_id="TRK-01", start_ts_ns=start,
                             end_ts_ns=start + seconds_to_ns(1), city="BLR", sensors={},
                             ontology_version="labelox-in-0.1.0"))
            for f, ts in ((f1, start), (f2, ts2)):
                db.add(Frame(frame_id=f, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri,
                             width=640, height=480, quality=0.9))
            from db.models import Track
            db.add(Track(track_id=tid, session_id=sid, class_id=cid_a, first_ts_ns=start, last_ts_ns=ts2))
            # same track, class flips between the two frames
            db.add(Object(object_id=uuid.uuid4(), frame_id=f1, track_id=tid, class_id=cid_a, bbox=[10, 300, 60, 460],
                          conf=0.9, source="fused", state="review", provenance={"agreement": True}, attrs={}, version=1))
            db.add(Object(object_id=uuid.uuid4(), frame_id=f2, track_id=tid, class_id=cid_b, bbox=[12, 302, 62, 462],
                          conf=0.9, source="fused", state="review", provenance={"agreement": True}, attrs={}, version=1))
            await db.commit()
        async with maker() as db:
            plan = await plan_frame(db, f2)
        item = next(i for i in plan["items"] if i["class_name"] and not i["critic_ok"])
        assert any("class flip" in r for r in item["critic_reasons"])

    run_async(_flow())
