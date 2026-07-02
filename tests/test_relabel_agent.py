"""Relabel agent (reasoning-layer class correction) and the enhanced keyframe interpolation.

The relabel decision is tested with a stubbed independent model so the margin logic is exercised
deterministically; the DB plan/commit/revert path runs against real infra with the model stubbed. The
interpolation tests are pure: they pin the shape-preserving spline (no overshoot) and the confidence decay.
"""

from __future__ import annotations

import asyncio
import uuid

import cv2
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


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


# --- the reasoning: margin over the independent model's distribution --------------------------------------

def test_decide_relabels_only_on_clear_margin():
    from services.agent import relabel_agent

    crop = np.zeros((10, 10, 3), dtype=np.uint8)

    # Suggested class beats current by a clear margin and clears the floor -> a decisive relabel (keep).
    relabel_agent.classify_crop = lambda c, topk=20: [  # type: ignore[attr-defined]
        {"class_id": 7, "class_name": "autorickshaw", "conf": 0.72},
        {"class_id": 3, "class_name": "car", "conf": 0.10},
    ]
    import services.autolabel.classify_crop as cc
    cc.classify_crop = relabel_agent.classify_crop  # _decide imports from the module at call time

    d = relabel_agent._decide(crop, current_id=3, min_conf=0.45, margin=0.15, strong_conf=0.60, strong_margin=0.30)
    assert d is not None and d[0] == 7 and d[3] == "relabel_keep"

    # Same top class but the current class is close behind -> margin not met -> leave it alone.
    cc.classify_crop = lambda c, topk=20: [
        {"class_id": 7, "class_name": "autorickshaw", "conf": 0.52},
        {"class_id": 3, "class_name": "car", "conf": 0.48},
    ]
    assert relabel_agent._decide(crop, 3, min_conf=0.45, margin=0.15, strong_conf=0.60, strong_margin=0.30) is None

    # Model already agrees with the current label -> no proposal.
    cc.classify_crop = lambda c, topk=20: [{"class_id": 3, "class_name": "car", "conf": 0.9}]
    assert relabel_agent._decide(crop, 3, min_conf=0.45, margin=0.15, strong_conf=0.60, strong_margin=0.30) is None

    # Clears the margin but not the strong bar -> applied, but routed to review.
    cc.classify_crop = lambda c, topk=20: [
        {"class_id": 7, "class_name": "autorickshaw", "conf": 0.50},
        {"class_id": 3, "class_name": "car", "conf": 0.20},
    ]
    d = relabel_agent._decide(crop, 3, min_conf=0.45, margin=0.15, strong_conf=0.60, strong_margin=0.30)
    assert d is not None and d[3] == "relabel_review"


# --- enhanced interpolation: shape-preserving, no overshoot, confidence decays -----------------------------

def _boxes(widths, cx=100.0, cy=100.0, h=40.0):
    return np.asarray([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for w in widths], dtype=float)


def test_cubic_interpolation_does_not_overshoot():
    from services.temporal.interpolate import build_box_interpolator

    # A near-flat run then a sharp rise: an ordinary cubic dips below the flat part (Runge). PCHIP must not.
    kf_ts = [0, 10, 20]
    box_at, src = build_box_interpolator(kf_ts, _boxes([10.0, 11.0, 40.0]), "cubic")
    assert src == "cubic"
    widths = []
    for ts in range(0, 21):
        x1, _, x2, _ = box_at(float(ts))
        assert x2 > x1                                   # never inverts
        widths.append(x2 - x1)
    assert min(widths) >= 10.0 - 1e-6                    # no undershoot below the data floor
    assert max(widths) <= 40.0 + 1e-6                    # no overshoot above the data ceiling


def test_cubic_falls_back_to_linear_with_two_keyframes():
    from services.temporal.interpolate import build_box_interpolator

    box_at, src = build_box_interpolator([0, 10], _boxes([10.0, 20.0]), "cubic")
    assert src == "linear"
    x1, _, x2, _ = box_at(5.0)
    assert abs((x2 - x1) - 15.0) < 1e-6                  # midpoint of a linear ramp


def test_interp_confidence_decays_from_anchors():
    from services.temporal.interpolate import _interp_conf

    kf = [0, 100]
    assert _interp_conf(0.0, kf) == pytest.approx(0.55, abs=1e-6)     # at an anchor
    assert _interp_conf(50.0, kf) == pytest.approx(0.30, abs=1e-6)    # mid-gap, least certain
    assert _interp_conf(50.0, kf) < _interp_conf(10.0, kf)           # monotone toward the middle


# --- DB path: commit a relabel, then revert it exactly ----------------------------------------------------

async def _seed_frame_with_car():
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, fid, oid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(1).integers(20, 230, size=(240, 320, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
    car = next(c.id for c in onto.classes if c.name == "sedan")
    auto = next(c.id for c in onto.classes if c.name == "autorickshaw")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=320, height=240,
                     quality=0.9, scene={}))
        db.add(Object(object_id=oid, frame_id=fid, class_id=car, bbox=[10.0, 10.0, 120.0, 120.0], conf=0.9,
                      source="fused", state="review", attrs={}, provenance={}, version=1))
        await db.commit()
    return str(fid), str(oid), car, auto


@requires_infra
def test_commit_and_revert_relabel():
    import services.autolabel.classify_crop as cc
    from db.models import Object
    from db.session import get_sessionmaker
    from services.agent.relabel_agent import commit_relabel
    from services.agent.runs import revert_run

    fid, oid, car, auto = run_async(_seed_frame_with_car())
    # Force the independent model to decisively call the car an autorickshaw.
    cc.classify_crop = lambda c, topk=20: [
        {"class_id": auto, "class_name": "autorickshaw", "conf": 0.8},
        {"class_id": car, "class_name": "car", "conf": 0.05},
    ]

    async def _flow():
        async with get_sessionmaker()() as db:
            res = await commit_relabel(db, uuid.UUID(fid))
        assert res["relabeled"] == 1
        async with get_sessionmaker()() as db:
            obj = await db.get(Object, uuid.UUID(oid))
            assert obj.class_id == auto and obj.provenance.get("agent_run_id") == res["run_id"]
        async with get_sessionmaker()() as db:
            rev = await revert_run(db, uuid.UUID(res["run_id"]))
        assert rev["reverted"] == 1
        async with get_sessionmaker()() as db:
            obj = await db.get(Object, uuid.UUID(oid))
            assert obj.class_id == car                      # restored exactly
            assert "agent_run_id" not in (obj.provenance or {})

    run_async(_flow())
