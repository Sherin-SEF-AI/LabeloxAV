"""Agent D's driving tools: lanes in a top-down view (#7), paths and intent (#8), camera-LiDAR alignment (#9).

Pure geometry runs without a database: splines, the ground lift and ego-pose transfer, topology with a person's
hints, the Lanelet2 and OpenDRIVE output, the CTRV baseline, and the alignment search on a ray-cast street whose
true calibration is known. The write paths run against the test database on generated sessions.
"""

from __future__ import annotations

import asyncio
import math
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from _driving_scene import lidar_scan, render, true_calibration, yaw_quat  # noqa: E402


def _nominal(w=1920, h=1080, height=1.5, pitch=0.0):
    from services.calibration.resolve import Calibration

    return Calibration(cam_id="cam_front", model="pinhole", fx=1000.0, fy=1000.0, cx=w / 2, cy=h / 2,
                       rpy_deg=(0.0, pitch, 0.0), xyz_m=(0.0, 0.0, height), source="dataset", quality=0.9)


# ---- #7 geometry --------------------------------------------------------------------------------------------------

def test_catmull_rom_passes_through_its_points_and_keeps_two_point_lines_straight():
    from services.hdmap.bev_lanes import catmull_rom

    pts = [[0, 0], [10, 5], [20, 0], [30, 8]]
    out = catmull_rom(pts, 8)
    assert len(out) == 3 * 8 + 1
    for i, p in enumerate(pts):
        assert out[i * 8] == pytest.approx(p)
    assert catmull_rom([[0, 0], [5, 5]], 8) == [[0.0, 0.0], [5.0, 5.0]]


def test_ground_lift_round_trips_and_left_is_positive():
    from services.hdmap.bev_lanes import ground_to_image, image_to_ground

    cal = _nominal()
    g = [[8.0, 1.75], [20.0, -1.75], [35.0, 0.0]]
    img = ground_to_image(g, cal)
    assert img[0][0] < 960 < img[1][0]          # left of the camera is left in the image
    back = image_to_ground(img, cal)
    assert np.allclose(back, g, atol=0.02)
    assert image_to_ground([[960, 100]], cal) == []  # above the horizon: dropped, not clamped


def test_lane_carried_by_ego_pose_lands_where_the_road_is():
    """A straight lane 1.75 m left of a vehicle driving 5 m forward: the carried lane is the same line, 5 m
    nearer; and after a 10 degree turn the image position follows the geometry, not the old pixels."""
    from services.hdmap.bev_lanes import carry_lane, ego_transform, ground_to_image, image_to_ground

    class P:
        def __init__(self, x, y, yaw):
            self.x, self.y, self.z = x, y, 0.0
            self.qw, self.qx, self.qy, self.qz = yaw_quat(yaw)

    cal = _nominal()
    lane_img = ground_to_image([[6.0, 1.75], [15.0, 1.75], [30.0, 1.75]], cal)
    t0 = ego_transform(P(0, 0, 0), None, cal)
    t1 = ego_transform(P(5, 0, 0), None, cal)
    moved = carry_lane(lane_img, cal, cal, t0, t1, 1920, 1080)
    g = image_to_ground(moved, cal)
    assert all(abs(p[1] - 1.75) < 0.02 for p in g)
    assert g[0][0] == pytest.approx(10.0, abs=0.05)          # the point now 1 m ahead is under the bonnet
    # a left turn of 10 degrees: the lane (fixed in the world) swings to the right in the vehicle frame
    t2 = ego_transform(P(5, 0, math.radians(10)), None, cal)
    turned = image_to_ground(carry_lane(lane_img, cal, cal, t0, t2, 1920, 1080), cal)
    assert turned[-1][1] < 1.75 - 1.0


def test_topology_hints_force_pairs_and_successors():
    from services.hdmap.topology import build_lanelets

    def line(x0, y0, x1, y1):
        return [(x0 * 1e-5, y0 * 1e-5), (x1 * 1e-5, y1 * 1e-5)]

    b = [{"id": "a", "points": line(0, 0, 0, 30), "lane_type": "solid", "confidence": 1},
         {"id": "b", "points": line(4, 30, 4, 0), "lane_type": "dashed", "confidence": 1},  # drawn backwards
         {"id": "c", "points": line(0, 32, 0, 60), "lane_type": "solid", "confidence": 1},
         {"id": "d", "points": line(4, 32, 4, 60), "lane_type": "dashed", "confidence": 1}]
    res = build_lanelets(b, forced_pairs=[("a", "b"), ("c", "d")], forced_successors=[("a", "c")])
    assert len(res["lanelets"]) == 2
    assert res["lanelets"][0].source_elements == ("a", "b")
    assert (0, 1) in res["successors"]
    # the reversed right boundary was turned to run with the left one
    r = res["lanelets"][0].right
    assert r[0][1] < r[-1][1]


def test_lanelet2_and_opendrive_carry_the_drawn_regulation():
    from services.hdmap.bev_lanes import hints_from_relations, wkt_line
    from services.hdmap.export import to_lanelet2_osm, to_opendrive

    def ll(x, y0, y1):
        return wkt_line([(x * 1e-5, y0 * 1e-5), (x * 1e-5, y1 * 1e-5)])

    fused = [{"element_id": "L", "kind": "lane", "wkt": ll(0, 0, 30), "attrs": {"lane_type": "solid"}, "confidence": 1},
             {"element_id": "R", "kind": "lane", "wkt": ll(3.5, 0, 30), "attrs": {"lane_type": "dashed"},
              "confidence": 1},
             {"element_id": "S", "kind": "stop_line", "wkt": wkt_line([(0, 30e-5), (3.5e-5, 30e-5)]),
              "attrs": {"lane_type": "stop_line"}, "confidence": 1}]
    rels = [{"lane_id": "L", "relation": "right_neighbour", "target_lane_id": "R"},
            {"lane_id": "L", "relation": "stops_at", "target_lane_id": "S"},
            {"lane_id": "L", "relation": "controlled_by", "target_lane_id": None, "target_object_id": "sig"}]
    hints = hints_from_relations(rels)
    assert hints["pairs"] == [("L", "R")]
    reg = hints["regulatory"][0]
    reg.pop("signal_ref")
    reg["signal"] = (1.7e-5, 30e-5)
    osm = ET.fromstring(to_lanelet2_osm(fused, hints))
    rels_el = osm.findall("relation")
    kinds = [{t.get("k"): t.get("v") for t in r.findall("tag")} for r in rels_el]
    assert any(k.get("subtype") == "traffic_light" for k in kinds)
    lanelet = next(r for r, k in zip(rels_el, kinds, strict=True) if k.get("type") == "lanelet")
    assert any(m.get("role") == "regulatory_element" for m in lanelet.findall("member"))
    assert any({t.get("k"): t.get("v") for t in w.findall("tag")}.get("type") == "stop_line" for w in osm.findall("way"))
    xodr = ET.fromstring(to_opendrive(fused, (0.0, 0.0), hints))
    sigs = xodr.findall(".//signal")
    assert {s.get("name") for s in sigs} == {"traffic_light", "stop_line"}


# ---- #8 geometry --------------------------------------------------------------------------------------------------

def test_ctrv_recovers_a_constant_turn_and_measures_its_own_error():
    from services.temporal.trajectory import baseline_error, ctrv_forecast, estimate_ctrv_state

    v, w = 8.0, 0.2
    ts = np.arange(0, 3.01, 0.1)
    th = w * ts
    xs, ys = v / w * np.sin(th), v / w * (1 - np.cos(th))
    st = estimate_ctrv_state(ts.tolist(), np.c_[xs, ys].tolist())
    assert st["speed_mps"] == pytest.approx(v, rel=0.03)
    assert st["yaw_rate_rps"] == pytest.approx(w, abs=0.03)
    assert st["heading_rad"] == pytest.approx(w * 3.0, abs=0.1)
    fc = ctrv_forecast(st, horizon_s=2.0, dt_s=0.25)
    assert fc[0] == [0.0, pytest.approx(xs[-1], abs=1e-3), pytest.approx(ys[-1], abs=1e-3)]
    t2 = np.arange(0.1, 2.01, 0.1)
    th2 = w * (3.0 + t2)
    truth = np.c_[t2, v / w * np.sin(th2), v / w * (1 - np.cos(th2))]
    err = baseline_error(fc, truth.tolist())
    assert err["ade_m"] < 0.6 and err["n"] == len(t2)
    # a straight line has no yaw rate and forecasts straight
    st2 = estimate_ctrv_state([0, 0.5, 1.0], [[0, 0], [5, 0], [10, 0]])
    assert st2["yaw_rate_rps"] == pytest.approx(0, abs=1e-6)
    assert ctrv_forecast(st2, 1.0, 0.5)[-1] == [1.0, pytest.approx(20.0), pytest.approx(0.0)]


# ---- #9 alignment -------------------------------------------------------------------------------------------------

def test_alignment_projection_matches_the_shared_projector():
    from services.lidar.calib.align import project
    from services.lidar.project import project_to_camera

    cal = true_calibration()
    xyz, _ = lidar_scan()
    uv, ins, _ = project(xyz, cal, 1242, 375)
    sel = np.flatnonzero(ins)[:4000]
    ref = project_to_camera(xyz[sel], "cam_front", 1242, 375, cal)
    assert ref["in_image"].all()
    assert np.abs(ref["uv"] - uv[sel]).max() < 0.01


def test_auto_refine_recovers_a_perturbed_extrinsic():
    from services.lidar.calib.align import (
        prepare_view,
        projection_shift,
        refine,
        score,
        with_extrinsics,
    )

    truth = true_calibration()
    xyz, inten = lidar_scan()
    view = prepare_view(xyz, inten, render(truth))
    bad = with_extrinsics(truth, (1.5, -0.2, 1.1), (0.37, -0.14, -0.03))
    shift0 = projection_shift(xyz, truth, bad, 1242, 375)
    assert shift0["mean_px"] > 20
    s_true, s_bad = score([view], truth), score([view], bad)
    assert s_true["edge"] > s_bad["edge"] + 0.1 and s_true["mi"] > s_bad["mi"]
    res = refine([view], bad)
    assert res["improved"] and res["after"]["objective"] > res["before"]["objective"]
    fixed = with_extrinsics(truth, res["rpy_deg"], res["xyz_m"])
    shift1 = projection_shift(xyz, truth, fixed, 1242, 375)
    assert shift1["mean_px"] < 4.0, shift1
    assert np.allclose(res["rpy_deg"], truth.rpy_deg, atol=0.3)          # from 1.5 degrees off


def test_point_pairs_solve_the_extrinsic():
    from services.lidar.calib.align import pairs_error, project, solve_from_pairs, with_extrinsics

    truth = true_calibration()
    xyz, _ = lidar_scan()
    uv, ins, _ = project(xyz, truth, 1242, 375)
    idx = np.flatnonzero(ins)[:: max(1, int(ins.sum()) // 12)][:12]
    bad = with_extrinsics(truth, (1.0, 1.0, -1.0), (0.4, 0.0, 0.0))
    assert pairs_error(xyz[idx], uv[idx], bad, 1242, 375)["rms"] > 10
    sol = solve_from_pairs(xyz[idx], uv[idx], bad)
    assert pairs_error(xyz[idx], uv[idx], with_extrinsics(truth, sol["rpy_deg"], sol["xyz_m"]),
                       1242, 375)["rms"] < 0.05
    with pytest.raises(ValueError):
        solve_from_pairs(xyz[idx[:3]], uv[idx[:3]], bad)


# ---- database -----------------------------------------------------------------------------------------------------

def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _seed_drive(n: int = 6, with_cloud: bool = False, with_track: bool = True) -> dict:
    """A session with n frames 0.5 s apart, an ego pose per frame (driving +x at 10 m/s), a camera calibration,
    one sedan track drifting left, a traffic signal box, and optionally a LiDAR scan per frame of the ray-cast
    street with its rendered image."""
    import cv2

    from core.storage import get_object_store
    from core.timebase import now_ns
    from db.models import CameraCalibration, EgoPose, Frame, Object, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.ingest.normalize import Cloud
    from services.lidar.ingest.store import store_cloud

    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()
    truth = true_calibration()
    w, h = (1242, 375) if with_cloud else (1920, 1080)
    img = render(truth) if with_cloud else np.full((h, w, 3), 90, np.uint8)
    _ok, jpg = cv2.imencode(".jpg", img)
    cal_fields = (dict(fx=truth.fx, fy=truth.fy, cx=truth.cx, cy=truth.cy, ref_width=1242,
                       rpy_deg=list(truth.rpy_deg), xyz_m=list(truth.xyz_m))
                  if with_cloud else dict(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, ref_width=1920,
                                          rpy_deg=[0.0, 0.0, 0.0], xyz_m=[0.0, 0.0, 1.5]))

    async def go():
        sid, t0 = uuid.uuid4(), now_ns()
        out = {"session_id": sid, "frames": [], "ts": []}
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="DRIVING-T", start_ts_ns=t0, end_ts_ns=t0 + 10**10,
                             city="Bengaluru", ontology_version=onto.version,
                             sensors={"ego_frame": "velodyne", "lidar_height_m": 1.73} if with_cloud else {}))
            await db.flush()
            db.add(CameraCalibration(session_id=sid, cam_id="cam_front", model="pinhole", dist=[], source="dataset",
                                     quality=0.95, **cal_fields))
            for i in range(n):
                ts = t0 + i * 500_000_000
                fid = uuid.uuid4()
                uri = store.put_bytes(f"frames/{sid}/cam_front/{ts}.jpg", jpg.tobytes(), "image/jpeg")
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_front", img_uri=uri, width=w,
                             height=h, quality=0.9, scene={}))
                out["frames"].append(fid)
                out["ts"].append(ts)
            await db.flush()
            for i, ts in enumerate(out["ts"]):
                qw, qx, qy, qz = yaw_quat(0.0)
                db.add(EgoPose(session_id=sid, ts_ns=ts, frame_id=out["frames"][i], x=5.0 * i, y=0.0, z=0.0,
                               qw=qw, qx=qx, qy=qy, qz=qz, source="gnss_imu", quality=0.9, measured=True))
            if with_track:
                from services.hdmap.bev_lanes import ground_to_image

                cal = _nominal()
                sedan = onto.by_name("sedan").id
                tid = uuid.uuid4()
                db.add(Track(track_id=tid, session_id=sid, class_id=sedan, first_ts_ns=out["ts"][0],
                             last_ts_ns=out["ts"][-1], intents=[]))
                await db.flush()
                for i, (fid, _ts) in enumerate(zip(out["frames"], out["ts"], strict=True)):
                    # world position: x = 20 + 12 t (faster than ego), y drifting left 0.5 m/s
                    t = i * 0.5
                    wx, wy = 20 + 12 * t, -1.5 + 0.5 * t
                    fwd, left = wx - 5.0 * i, wy
                    (u, v), = ground_to_image([[fwd, left]], cal)
                    db.add(Object(frame_id=fid, track_id=tid, class_id=sedan, bbox=[u - 40, v - 60, u + 40, v],
                                  source="human", state="accepted", conf=1.0))
                sig = onto.by_name("traffic_signal").id
                db.add(Object(frame_id=out["frames"][0], class_id=sig, bbox=[900, 200, 930, 260], source="human",
                              state="accepted", conf=1.0))
                out["track_id"] = tid
            await db.commit()
        if with_cloud:
            xyz, inten = lidar_scan()
            for ts in out["ts"]:
                await store_cloud(Cloud(xyz=xyz, intensity=inten, ts_ns=ts, source="lidar"), sid, source="lidar")
        return out

    return _run(go())



@pytest.mark.db
def test_bev_lane_editor_relations_propagation_and_export():
    from _authutil import auth_headers

    from services.api.main import app

    s = _seed_drive(6)
    f0 = str(s["frames"][0])
    with TestClient(app) as c:
        ann = auth_headers("annotator")
        st = c.get(f"/api/tools/driving/frames/{f0}/bev-lanes", headers=ann)
        assert st.status_code == 200, st.text
        v = st.json()["view"]
        assert st.json()["has_pose"] and st.json()["next_frames"] == 5
        assert len(st.json()["signals"]) == 1

        def bev(fwd, left):
            return [(-left + v["half_width_m"]) * v["px_per_m"], (v["far_m"] - fwd) * v["px_per_m"]]

        mk = lambda left, typ: c.put(f"/api/tools/driving/frames/{f0}/bev-lanes", headers=ann,  # noqa: E731
                                     json={"bev_points": [bev(6, left), bev(14, left), bev(24, left)],
                                           "lane_type": typ})
        a, b = mk(1.75, "solid"), mk(-1.75, "dashed")
        stop = c.put(f"/api/tools/driving/frames/{f0}/bev-lanes", headers=ann,
                     json={"bev_points": [bev(24.5, 1.75), bev(24.5, -1.75)], "lane_type": "stop_line"})
        assert a.status_code == 200 and b.status_code == 200 and stop.status_code == 200, a.text
        la, lb, ls = a.json()["lane_id"], b.json()["lane_id"], stop.json()["lane_id"]
        assert c.put(f"/api/tools/driving/frames/{f0}/bev-lanes", headers=ann,
                     json={"bev_points": [bev(6, 0)], "lane_type": "solid"}).status_code == 400
        assert c.put(f"/api/tools/driving/frames/{f0}/bev-lanes", headers=ann,
                     json={"bev_points": [bev(6, 0), bev(9, 0)], "lane_type": "zigzag"}).status_code == 400
        sig = st.json()["signals"][0]["object_id"]
        r = c.post(f"/api/tools/driving/lanes/{la}/relations", headers=ann,
                   json={"relation": "right_neighbour", "target_lane_id": lb})
        assert r.status_code == 201, r.text
        assert c.post(f"/api/tools/driving/lanes/{la}/relations", headers=ann,
                      json={"relation": "stops_at", "target_lane_id": lb}).status_code == 400
        assert c.post(f"/api/tools/driving/lanes/{la}/relations", headers=ann,
                      json={"relation": "stops_at", "target_lane_id": ls}).status_code == 201
        assert c.post(f"/api/tools/driving/lanes/{la}/relations", headers=ann,
                      json={"relation": "controlled_by", "target_object_id": sig}).status_code == 201
        dup = c.post(f"/api/tools/driving/lanes/{la}/relations", headers=ann,
                     json={"relation": "right_neighbour", "target_lane_id": lb})
        assert dup.json()["relation_id"] == r.json()["relation_id"]
        st2 = c.get(f"/api/tools/driving/frames/{f0}/bev-lanes", headers=ann).json()
        assert len(st2["lanes"]) == 3 and len(st2["relations"]) == 3
        got = next(x for x in st2["lanes"] if x["lane_id"] == la)
        assert np.allclose(got["bev_points"][0], bev(6, 1.75), atol=0.5)

        p = c.post(f"/api/tools/driving/frames/{f0}/lanes/propagate-pose", headers=ann, json={"frames": 3})
        assert p.status_code == 200, p.text
        assert p.json()["queued"] is False and p.json()["created"] >= 4
        f2 = str(s["frames"][2])
        st3 = c.get(f"/api/tools/driving/frames/{f2}/bev-lanes", headers=ann).json()
        carried = [x for x in st3["lanes"] if x["method"] == "ego_pose"]
        assert carried
        # 10 m further on, the lane drawn 6..24 m ahead now starts about 10 m nearer (clipped at 2 m) and
        # stays 1.75 m left: the road did not move
        left_lane = min(carried, key=lambda x: x["bev_points"][0][0])
        xs = [q[0] for q in left_lane["bev_points"]]
        assert np.allclose(xs, bev(10, 1.75)[0], atol=1.0)
        again = c.post(f"/api/tools/driving/frames/{f0}/lanes/propagate-pose", headers=ann, json={"frames": 3})
        st4 = c.get(f"/api/tools/driving/frames/{f2}/bev-lanes", headers=ann).json()
        assert again.status_code == 200 and len(st4["lanes"]) == len(st3["lanes"])     # replaced, not duplicated

        ex = c.post("/api/tools/driving/lanes/export", headers=ann, json={"frame_ids": [f0]})
        assert ex.status_code == 200, ex.text
        e = ex.json()
        assert e["lanelets"] == 1 and e["stop_lines"] == 1 and e["regulatory"] == 1
        assert e["georeferenced"] is False and e["note"]          # the session has poses but no GNSS fix
        osm = ET.fromstring(e["lanelet2"])
        assert any(t.get("v") == "traffic_light" for t in osm.iter("tag"))
        assert ET.fromstring(e["opendrive"]).find(".//signal") is not None
        # two frames of the same lanes: carried copies keep the relations through track_ref
        ex2 = c.post("/api/tools/driving/lanes/export", headers=ann, json={"frame_ids": [f0, f2]}).json()
        assert ex2["lanelets"] == 2 and ex2["hints"]["pairs"] == 2

        assert c.delete(f"/api/tools/driving/lane-relations/{r.json()['relation_id']}", headers=ann).status_code == 200


@pytest.mark.db
def test_paths_forecast_futures_ego_and_intents():
    from _authutil import auth_headers

    from services.api.main import app

    s = _seed_drive(6)
    sid, tid = str(s["session_id"]), str(s["track_id"])
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        lst = c.get("/api/tools/driving/paths/sessions", headers=ann)
        assert lst.status_code == 200 and any(x["session_id"] == sid and x["ego_poses"] == 6 for x in lst.json())
        v = c.get(f"/api/tools/driving/paths/sessions/{sid}", headers=ann).json()
        assert v["coord_frame"] == "world" and len(v["ego"]["poses"]) == 6
        tr = next(t for t in v["tracks"] if t["track_id"] == tid)
        # in the world frame the sedan runs x = 20 + 12 t, y = -1.5 + 0.5 t: ego motion removed
        xy = np.array(tr["path"])[:, 1:]
        assert np.allclose(np.diff(xy[:, 0]), 6.0, atol=0.3) and np.allclose(np.diff(xy[:, 1]), 0.25, atol=0.1)

        anchor = tr["frames"][3]
        fc = c.get(f"/api/tools/driving/paths/tracks/{tid}/forecast", headers=ann,
                   params={"anchor_frame_id": anchor, "horizon_s": 1.0, "dt_s": 0.25})
        assert fc.status_code == 200, fc.text
        f = fc.json()
        assert f["model"] == "ctrv" and "baseline" in f["note"].lower()
        assert f["state"]["speed_mps"] == pytest.approx(math.hypot(12, 0.5), rel=0.08)
        assert f["error_vs_observed"]["ade_m"] < 0.5 and f["error_vs_observed"]["n"] == 2
        assert c.get(f"/api/tools/driving/paths/tracks/{tid}/forecast", headers=ann,
                     params={"anchor_frame_id": tr["frames"][0]}).status_code == 400

        pts = [[p[0], p[1], p[2] + 0.3] for p in f["forecast"]]
        sv = c.put(f"/api/tools/driving/paths/tracks/{tid}/future", headers=ann,
                   json={"anchor_frame_id": anchor, "points": pts, "source": "human", "seed": {"model": "ctrv"}})
        assert sv.status_code == 200 and sv.json()["coord_frame"] == "world"
        sv2 = c.put(f"/api/tools/driving/paths/tracks/{tid}/future", headers=ann,
                    json={"anchor_frame_id": anchor, "points": pts[:3], "source": "human"})
        assert sv2.json()["path_id"] == sv.json()["path_id"]          # one future per anchor, updated

        sm = c.post(f"/api/tools/driving/paths/sessions/{sid}/ego/smooth", headers=ann, json={"window": 5})
        assert sm.status_code == 200 and len(sm.json()["points"]) == 6
        eg = c.put(f"/api/tools/driving/paths/sessions/{sid}/ego", headers=ann,
                   json={"points": sm.json()["points"], "source": "smoothed"})
        assert eg.status_code == 200 and eg.json()["kind"] == "ego_path"

        it = c.get(f"/api/tools/driving/paths/tracks/{tid}/intents", headers=ann).json()
        assert "cut_in" in it["quick"] and "jaywalking" not in it["quick"]          # a sedan cannot jaywalk
        bad = c.post(f"/api/tools/driving/paths/tracks/{tid}/intents", headers=ann,
                     json={"event_type": "jaywalking", "start_frame_id": tr["frames"][0],
                           "end_frame_id": tr["frames"][2]})
        assert bad.status_code == 400
        ok = c.post(f"/api/tools/driving/paths/tracks/{tid}/intents", headers=ann,
                    json={"event_type": "cut_in", "start_frame_id": tr["frames"][4], "end_frame_id": tr["frames"][1]})
        assert ok.status_code == 201, ok.text
        assert ok.json()["state"] == "proposed" and ok.json()["mirrored_to_intents"]
        ok2 = c.post(f"/api/tools/driving/paths/tracks/{tid}/intents", headers=rev,
                     json={"event_type": "overtaking", "start_frame_id": tr["frames"][0],
                           "end_frame_id": tr["frames"][5]})
        assert ok2.json()["state"] == "accepted"
        it = c.get(f"/api/tools/driving/paths/tracks/{tid}/intents", headers=ann).json()
        assert len(it["events"]) == 2 and {i["intent"] for i in it["intents"]} == {"cut_in", "overtaking"}
        assert c.delete(f"/api/tools/driving/paths/intents/{ok2.json()['event_id']}", headers=ann).status_code == 403
        assert c.delete(f"/api/tools/driving/paths/intents/{ok.json()['event_id']}", headers=ann).status_code == 200

        sc = c.get(f"/api/tools/driving/paths/sessions/{sid}/scenario", headers=ann)
        assert sc.status_code == 200, sc.text
        doc = sc.json()["json"]
        assert doc["format"] == "labeloxav.paths_intents/1" and doc["ego"]["source"] == "path_label"
        actor = next(a for a in doc["actors"] if a["track_id"] == tid)
        assert actor["futures"] and actor["events"][0]["type"] == "overtaking"
        osc = ET.fromstring(sc.json()["openscenario"])
        assert osc.find(".//FollowTrajectoryAction") is not None
        assert c.delete(f"/api/tools/driving/paths/labels/{sv.json()['path_id']}", headers=ann).status_code == 200


@pytest.mark.db
def test_alignment_view_refine_and_manual_save_beside_the_dataset_row():
    from _authutil import auth_headers
    from sqlalchemy import select

    from db.models import CameraCalibration
    from db.session import get_sessionmaker
    from services.api.main import app
    from services.tools.registry import enqueue, run_job

    s = _seed_drive(3, with_cloud=True, with_track=False)
    sid = str(s["session_id"])
    truth = true_calibration()
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        ss = c.get("/api/tools/driving/align/sessions", headers=ann)
        assert ss.status_code == 200 and any(x["session_id"] == sid for x in ss.json())
        fr = c.get(f"/api/tools/driving/align/sessions/{sid}/frames", headers=ann).json()["frames"]
        assert len(fr) == 3
        f0 = fr[0]["frame_id"]
        v = c.get(f"/api/tools/driving/align/frames/{f0}/view", headers=ann)
        assert v.status_code == 200, v.text
        vj = v.json()
        pts = np.frombuffer(__import__("base64").b64decode(vj["points_b64"]), "<f4").reshape(-1, 5)
        assert len(pts) == vj["point_count"] and (pts[:, 4] > 0).sum() > 50
        assert vj["current"]["source"] == "dataset" and not vj["has_manual"]
        assert vj["edge_png"].startswith("data:image/png;base64,")

        bad_rpy = [truth.rpy_deg[0] + 1.2, truth.rpy_deg[1] - 0.8, truth.rpy_deg[2] + 1.0]
        bad_xyz = [truth.xyz_m[0] + 0.1, truth.xyz_m[1] - 0.08, truth.xyz_m[2]]
        sc = c.post("/api/tools/driving/align/score", headers=ann,
                    json={"frame_ids": [f0], "rpy_deg": bad_rpy, "xyz_m": bad_xyz}).json()
        assert sc["candidate"]["objective"] < sc["current"]["objective"]
        q = c.post("/api/tools/driving/align/refine", headers=ann,
                   json={"frame_ids": [f0], "rpy_deg": bad_rpy, "xyz_m": bad_xyz})
        assert q.status_code == 200 and q.json()["kind"] == "driving_calib_refine"

    async def enq():
        async with get_sessionmaker()() as db:
            job = await enqueue(db, "driving_calib_refine",
                                {"frame_ids": [f0, fr[1]["frame_id"]], "rpy_deg": bad_rpy, "xyz_m": bad_xyz},
                                session_id=s["session_id"], start=False)
            return job.job_id

    jid = _run(enq())
    assert _run(run_job(jid)) == "done"

    async def result():
        from db.models import ToolJob

        async with get_sessionmaker()() as db:
            return (await db.get(ToolJob, jid)).result

    res = _run(result())
    assert res["improved"] and res["shift_from_base"]["mean_px"] < 4.0, res["shift_from_base"]

    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        body = {"frame_ids": [f0], "rpy_deg": res["rpy_deg"], "xyz_m": res["xyz_m"], "method": "auto"}
        assert c.post("/api/tools/driving/align/save", headers=ann, json=body).status_code == 403
        sv = c.post("/api/tools/driving/align/save", headers=rev, json=body)
        assert sv.status_code in (200, 409), sv.text
        saved = sv.status_code == 200

    async def rows():
        async with get_sessionmaker()() as db:
            return [(r.source, list(r.rpy_deg)) for r in (await db.execute(select(CameraCalibration).where(
                CameraCalibration.session_id == s["session_id"]))).scalars().all()]

    got = _run(rows())
    # the dataset row is never touched
    assert ("dataset", [float(x) for x in truth.rpy_deg]) in [(a, [float(x) for x in b]) for a, b in got]
    if saved:
        assert {a for a, _ in got} == {"dataset", "manual"}
        with TestClient(app) as c:
            rev = auth_headers("reviewer")
            v2 = c.get(f"/api/tools/driving/align/frames/{f0}/view", headers=auth_headers("annotator")).json()
            assert v2["has_manual"] and v2["current"]["source"] == "manual" and v2["base"]["source"] == "dataset"
            h = c.get(f"/api/tools/driving/align/sessions/{sid}/history", headers=rev).json()["alignments"]
            assert any(x["saved"] for x in h) and any(x["method"] == "auto" and not x["saved"] for x in h)
            assert c.delete(f"/api/tools/driving/align/sessions/{sid}/cameras/cam_front/manual",
                            headers=rev).status_code == 200
        assert {a for a, _ in _run(rows())} == {"dataset"}
    else:
        # the shared resolver predates 0131 and the save undid itself rather than break the session
        assert {a for a, _ in got} == {"dataset"}


async def _manual_aware_resolve(session_id, cam_id, img_w, img_h):
    """The resolver as 0131 needs it (the change requested of the shared file): the manual row first."""
    from sqlalchemy import select

    from db.models import CameraCalibration
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.calibration.resolve import (
        calibration_from_row,
        ego_height_from_sensors,
        nominal_calibration,
    )

    async with get_sessionmaker()() as db:
        row = (await db.execute(select(CameraCalibration).where(
            CameraCalibration.session_id == session_id, CameraCalibration.cam_id == cam_id)
            .order_by((CameraCalibration.source == "manual").desc()).limit(1))).scalar_one_or_none()
        sensors = (await db.execute(select(DbSession.sensors).where(DbSession.session_id == session_id))).scalar()
    if row is None:
        return nominal_calibration(cam_id, img_w, img_h)
    cal = calibration_from_row(row, img_w, img_h)
    cal.ego_height_m = ego_height_from_sensors(sensors)
    return cal


@pytest.mark.db
def test_manual_save_with_a_manual_aware_resolver(monkeypatch):
    from _authutil import auth_headers

    import services.calibration.resolve as resolve
    from services.api.main import app

    monkeypatch.setattr(resolve, "resolve_calibration", _manual_aware_resolve)
    s = _seed_drive(2, with_cloud=True, with_track=False)
    sid = str(s["session_id"])
    truth = true_calibration()
    with TestClient(app) as c:
        rev = auth_headers("reviewer")
        f0 = c.get(f"/api/tools/driving/align/sessions/{sid}/frames", headers=rev).json()["frames"][0]["frame_id"]
        moved = [truth.rpy_deg[0], truth.rpy_deg[1] + 0.5, truth.rpy_deg[2]]
        sv = c.post("/api/tools/driving/align/save", headers=rev,
                    json={"frame_ids": [f0], "rpy_deg": moved, "xyz_m": list(truth.xyz_m), "method": "manual"})
        assert sv.status_code == 200, sv.text
        out = sv.json()
        # a worse alignment can be saved (a person may know better than the score) but it is reported as worse
        assert out["improved"] is False and out["resolver_source"] == "manual"
        v = c.get(f"/api/tools/driving/align/frames/{f0}/view", headers=rev).json()
        assert v["current"]["rpy_deg"][1] == pytest.approx(moved[1]) and v["base"]["source"] == "dataset"
        # saving again updates the one manual row
        sv2 = c.post("/api/tools/driving/align/save", headers=rev,
                     json={"frame_ids": [f0], "rpy_deg": list(truth.rpy_deg), "xyz_m": list(truth.xyz_m)})
        assert sv2.json()["calibration_id"] == out["calibration_id"]
        assert c.delete(f"/api/tools/driving/align/sessions/{sid}/cameras/cam_front/manual",
                        headers=rev).status_code == 200
        assert c.get(f"/api/tools/driving/align/frames/{f0}/view", headers=rev).json()["current"]["source"] == "dataset"
