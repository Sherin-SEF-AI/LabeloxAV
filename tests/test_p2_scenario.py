"""P2 scenario mining: tracking, trajectories, event detectors, NL parse (pure units), plus the
mine_session pipeline and scenario API (DB integration)."""

from __future__ import annotations

import uuid

import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.intelligence.events import detect_events
from services.intelligence.nlsearch import parse_query
from services.intelligence.tracking import Det, track_camera
from services.intelligence.trajectory import FrameCtx, build_trajectory

NS = 1_000_000_000


def _synth(boxes, class_id, w=640, h=480, fps=5):
    dets, ctx = [], {}
    for i, b in enumerate(boxes):
        fid = uuid.uuid4()
        ts = i * int(NS / fps)
        dets.append(Det(object_id=uuid.uuid4(), frame_id=fid, ts_ns=ts, cam_id="cam_f", bbox=tuple(b), class_id=class_id))
        ctx[fid] = FrameCtx(width=w, height=h, ego_speed=None, lat=12.97, lon=77.59)
    return dets, ctx


def _track_one(boxes, class_id):
    dets, ctx = _synth(boxes, class_id)
    _, tracks = track_camera(dets)
    assert len(tracks) == 1
    tr = tracks[0]
    return tr, {str(tr.track_id): build_trajectory(tr, ctx)}, ctx


# --- tracking -----------------------------------------------------------------


def test_tracker_groups_consecutive_overlapping_dets_into_one_track():
    boxes = [[100, 100, 200, 200], [110, 105, 210, 205], [122, 110, 222, 210], [134, 116, 234, 216]]
    dets, _ = _synth(boxes, 6)
    assignment, tracks = track_camera(dets)
    assert len(tracks) == 1
    assert len(set(assignment.values())) == 1  # all objects -> same track id
    assert len(assignment) == 4


def test_tracker_splits_disjoint_objects():
    # two well-separated objects per frame -> two tracks
    dets = []
    ctx = {}
    for i in range(4):
        fid = uuid.uuid4()
        ts = i * int(NS / 5)
        ctx[fid] = FrameCtx(640, 480, None, None, None)
        dets.append(Det(uuid.uuid4(), fid, ts, "cam_f", (10 + i * 5, 10, 80 + i * 5, 80), 11))
        dets.append(Det(uuid.uuid4(), fid, ts, "cam_f", (500 + i * 5, 300, 580 + i * 5, 400), 6))
    _, tracks = track_camera(dets)
    assert len(tracks) == 2


# --- trajectory ---------------------------------------------------------------


def test_trajectory_detects_closing_and_drift():
    # box grows (closing) and drifts right
    boxes = [[200, 150, 280, 230], [220, 150, 320, 250], [240, 145, 360, 275], [260, 140, 410, 300]]
    tr, trajs, _ = _track_one(boxes, 6)
    s = trajs[str(tr.track_id)].summary
    assert s["area_growth"] > 1.3
    assert s["approaching"] is True
    assert s["x_drift_frac"] > 0


# --- event detectors ----------------------------------------------------------


def test_hard_brake_from_ego_series():
    onto = get_ontology()
    ego = [(0, 15.0), (int(NS * 0.3), 4.0)]  # ~-36 m/s^2
    evs = detect_events([], {}, {}, ego, onto)
    assert any(e.type == "hard_brake" for e in evs)


def test_cut_in_and_near_miss_for_closing_vehicle_in_path():
    onto = get_ontology()
    # autorickshaw moving into the ego column (center) while growing fast
    boxes = [[120, 180, 200, 260], [180, 180, 290, 290], [240, 175, 380, 320], [270, 170, 430, 350]]
    tr, trajs, ctx = _track_one(boxes, 6)
    evs = detect_events([tr], trajs, ctx, [], onto)
    types = {e.type for e in evs}
    assert "cut_in" in types
    assert "near_miss" in types
    nm = next(e for e in evs if e.type == "near_miss")
    assert 0.0 < nm.criticality <= 1.0


def test_animal_on_road_low_in_frame():
    onto = get_ontology()
    # cattle (id 31) sitting low in the frame (carriageway region)
    boxes = [[250, 360, 330, 440], [252, 362, 332, 442], [254, 360, 334, 440], [256, 361, 336, 441]]
    tr, trajs, ctx = _track_one(boxes, 31)
    evs = detect_events([tr], trajs, ctx, [], onto)
    assert any(e.type == "animal_on_road" for e in evs)


def test_wrong_side_against_majority_flow():
    onto = get_ontology()
    # two vehicles, constant size (no closing); A drifts right, B drifts left
    a_boxes = [[100, 200, 180, 280], [130, 200, 210, 280], [160, 200, 240, 280], [190, 200, 270, 280], [220, 200, 300, 280]]
    b_boxes = [[520, 200, 600, 280], [490, 200, 570, 280], [460, 200, 540, 280], [430, 200, 510, 280], [400, 200, 480, 280]]
    da, ca = _synth(a_boxes, 11)
    db_, cb = _synth(b_boxes, 6)
    _, ta = track_camera(da)
    _, tb = track_camera(db_)
    ctx = {**ca, **cb}
    tracks = ta + tb
    trajs = {str(t.track_id): build_trajectory(t, ctx) for t in tracks}
    evs = detect_events(tracks, trajs, ctx, [], onto)
    assert any(e.type == "wrong_side" for e in evs)


# --- NL parse -----------------------------------------------------------------


def test_nl_parse_extracts_event_class_and_tags():
    p = parse_query("wrong-side autorickshaw cutting in at night on a wet road")
    assert "wrong_side" in p.types
    assert "cut_in" in p.types
    assert "autorickshaw" in p.actor_classes
    assert "night" in p.tags
    assert "wet" in p.tags


# --- DB + API integration -----------------------------------------------------


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")
pytestmark_async = pytest.mark.asyncio


@requires_infra
@pytest.mark.asyncio
async def test_mine_session_persists_tracks_and_scenarios():
    from core.timebase import seconds_to_ns
    from db.models import Frame, Object, Scenario, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from sqlalchemy import func, select
    from services.intelligence.run import mine_session

    maker = get_sessionmaker()
    sid = uuid.uuid4()
    start = 1_700_000_000 * NS
    # an autorickshaw closing into the ego column across 6 frames
    boxes = [[120, 180, 200, 260], [170, 180, 280, 290], [220, 175, 360, 320], [250, 172, 410, 340],
             [270, 170, 450, 360], [290, 168, 500, 380]]

    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-07", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(2), city="BLR", sensors={},
                         ontology_version="labelox-in-0.1.0"))
        await db.flush()
        for i, b in enumerate(boxes):
            fid = uuid.uuid4()
            ts = start + seconds_to_ns(i / 5)
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                         img_uri=f"s3://x/{fid}.jpg", width=640, height=480, ego_speed=8.0, quality=0.9))
            db.add(Object(frame_id=fid, class_id=6, bbox=b, conf=0.8, attrs={}, source="fused",
                          state="auto_accept", provenance={}))
        await db.commit()

    summary = await mine_session(sid)
    assert summary["tracks"] >= 1
    assert summary["objects_tracked"] >= 4
    assert summary["scenarios"] >= 1

    async with maker() as db:
        tracks = (await db.execute(select(func.count()).select_from(Track).where(Track.session_id == sid))).scalar_one()
        scen = (await db.execute(select(func.count()).select_from(Scenario).where(Scenario.session_id == sid))).scalar_one()
        tracked = (await db.execute(
            select(func.count()).select_from(Object).join(Frame).where(Frame.session_id == sid, Object.track_id.isnot(None))
        )).scalar_one()
        assert tracks >= 1 and scen >= 1 and tracked >= 4

    # NL search finds the mined scenario
    from services.intelligence.nlsearch import search_scenarios

    async with maker() as db:
        results = await search_scenarios(db, "autorickshaw cutting in", session_id=str(sid))
    assert any(r["type"] in {"cut_in", "near_miss"} for r in results)
