"""M-F.2 behavior/intent annotation: trajectory-derived geometric intents are proposed from the stored
trajectory, an unclear track stays unknown (no proposal), and a human can confirm or set an intent from the
closed vocabulary (an out-of-vocabulary intent is rejected). The VLM contextual path is exercised operationally
(it needs the VLM), not here. Single asyncio.run so the cached engine binds to one loop."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.intelligence.intent import (
    VEHICLE_INTENTS,
    VRU_INTENTS,
    propose_from_trajectory,
    vocab,
)


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


class _T:  # a lightweight stand-in for the Track ORM row (propose_from_trajectory only reads two fields)
    def __init__(self, class_id, trajectory):
        self.class_id = class_id
        self.trajectory = trajectory


def _vehicle_class_id(onto):
    for c in onto.classes:
        if c.l1 == "four_wheeler":
            return c.id
    raise AssertionError("no four_wheeler class in ontology")


def test_vocab_is_closed():
    v = vocab()
    assert set(v["vehicle"]) == set(VEHICLE_INTENTS) and set(v["vru"]) == set(VRU_INTENTS)
    assert "cut_in" in v["trajectory_vehicle"] and "looking_at_vehicle" in v["vlm_vru"]


def test_trajectory_proposes_cut_in_and_leaves_unclear_unknown():
    onto = get_ontology()
    cfg = get_settings()
    vcid = _vehicle_class_id(onto)

    # a cut-in: area grows sharply while the box sits in the ego column (cx near centre, 1920 wide frame)
    pts = [{"cx": 900, "by": 500, "area": 5000, "ts_ns": i * 10 ** 8, "ego_speed": 10.0} for i in range(3)]
    pts[-1]["cx"] = 960
    cut_in_traj = {"points": pts, "summary": {"n": 6, "approaching": True, "area_growth": 1.6,
                                              "duration_ns": 6 * 10 ** 8, "net_disp_px": 40.0,
                                              "x_drift_frac": 0.02, "mean_speed_px": 3.0}}
    props = propose_from_trajectory(_T(vcid, cut_in_traj), onto, cfg, flow_sign=0.0, frame_width=1920.0)
    assert any(p["intent"] == "cut_in" and p["source"] == "trajectory" and p["status"] == "proposed" for p in props)

    # a near-stationary vehicle mid-frame: no clear geometric intent -> unknown (empty)
    still = {"points": [{"cx": 950, "by": 500, "area": 5000, "ts_ns": i * 10 ** 8, "ego_speed": 10.0} for i in range(4)],
             "summary": {"n": 4, "approaching": False, "area_growth": 1.0, "duration_ns": 4 * 10 ** 8,
                         "net_disp_px": 0.5, "x_drift_frac": 0.0, "mean_speed_px": 0.2}}
    assert propose_from_trajectory(_T(vcid, still), onto, cfg, flow_sign=0.0, frame_width=1920.0) == []


@requires_infra
def test_set_and_confirm_intent():
    from sqlalchemy import select
    from db.models import Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.intelligence.intent import propose_track, set_intent

    onto = get_ontology()
    sid = uuid.uuid4()
    tid = uuid.uuid4()
    vcid = _vehicle_class_id(onto)

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="INTENT", start_ts_ns=0, end_ts_ns=1,
                             ontology_version=onto.version))
            await db.flush()
            traj = {"points": [{"cx": 900 + i * 20, "by": 500, "area": 4000 + i * 1500, "ts_ns": i * 10 ** 8,
                                "ego_speed": 10.0} for i in range(4)],
                    "summary": {"n": 4, "approaching": True, "area_growth": 1.7, "duration_ns": 4 * 10 ** 8,
                                "net_disp_px": 40.0, "x_drift_frac": 0.02, "mean_speed_px": 3.0}}
            db.add(Track(track_id=tid, session_id=sid, class_id=vcid, first_ts_ns=0, last_ts_ns=1, trajectory=traj))
            await db.commit()

        # trajectory proposes cut_in
        r = await propose_track(tid)
        assert "cut_in" in r["proposed"]

        # an out-of-vocabulary intent is rejected
        bad = await set_intent(tid, "teleporting", "vehicle")
        assert "error" in bad

        # a human confirms cut_in: it becomes source=human, status=confirmed, and the machine proposal is marked confirmed
        ok = await set_intent(tid, "cut_in", "vehicle")
        assert "error" not in ok
        human = [r for r in ok["intents"] if r["source"] == "human"]
        assert human and human[0]["intent"] == "cut_in" and human[0]["status"] == "confirmed"
        assert all(r["status"] == "confirmed" for r in ok["intents"] if r["intent"] == "cut_in")

        async with maker() as db:
            await db.delete(await db.get(DbSession, sid))  # cascades track
            await db.commit()

    asyncio.run(run())
