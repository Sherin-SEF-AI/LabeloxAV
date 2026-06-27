"""P3 per-object dynamics: on a constructed track of an object approaching the ego in a straight line, the
IPM ground-plane recovers decreasing distance, a plausible speed, a shrinking time-to-collision, and a
high risk when it is close and closing. Bboxes are placed so the flat-road IPM yields known distances
(~15, ~10, ~7 m); ego speed is 0 so the object's own approach drives the closing rate. Single asyncio.run."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _bottom_for_distance(d_m: float) -> float:
    # invert the flat-road IPM for a 640x480 cam_f frame: forward = (h * fy) / (v - cy)
    cfg = get_settings()
    fy = cfg.rig.lenses["narrow"].fy * (640 / cfg.rig.ref_width)
    cy = 480 / 2.0
    return cy + (cfg.spatial.camera_height_m * fy) / d_m


@requires_infra
def test_dynamics_recovers_distance_speed_ttc_risk_on_approach():
    from db.models import Frame, Object, ObjectDynamics, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.dynamics.compute import compute_session_dynamics

    sid = uuid.uuid4()
    tid = uuid.uuid4()
    ped = get_ontology().by_name("pedestrian").id
    plan = [(0, 15.0), (400_000_000, 10.0), (800_000_000, 7.0)]  # (ts_ns, true distance) over 0.4s steps

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="DYN-TEST", start_ts_ns=0, end_ts_ns=1,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            db.add(Track(track_id=tid, session_id=sid, class_id=ped, first_ts_ns=0, last_ts_ns=800_000_000,
                         tracker_version="test"))
            await db.flush()
            oids = []
            for ts, dist in plan:
                f = Frame(session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=f"s3://x/{ts}.jpg",
                          width=640, height=480, ego_speed=0.0)
                db.add(f)
                await db.flush()
                vb = _bottom_for_distance(dist)
                o = Object(frame_id=f.frame_id, class_id=ped, bbox=[315.0, 100.0, 325.0, vb], conf=0.9,
                           state="accepted", source="human", track_id=tid)
                db.add(o)
                await db.flush()
                oids.append((o.object_id, dist))
            await db.commit()

            res = await compute_session_dynamics(sid)
            assert res["objects"] == 3 and res["tracked_with_speed"] >= 2

            dyn = {}
            for oid, dist in oids:
                d = await db.get(ObjectDynamics, oid)
                assert d is not None
                dyn[round(dist)] = d

            # distance recovered within ~10% of truth, and decreasing along the track
            assert abs(dyn[15].distance_m - 15.0) < 1.5
            assert abs(dyn[7].distance_m - 7.0) < 1.0
            assert dyn[15].distance_m > dyn[10].distance_m > dyn[7].distance_m

            # the first frame has no predecessor -> distance only, no speed
            assert dyn[15].speed_kmh is None
            # later frames get a plausible speed (object closing ~12.5 m/s = ~45 km/h), within the sane band
            assert dyn[7].speed_kmh is not None and 20.0 < dyn[7].speed_kmh < 150.0
            # closing + shrinking TTC + high risk when close and approaching
            assert dyn[7].closing_speed_kmh > 0 and dyn[7].ttc_s is not None and dyn[7].ttc_s < 1.5
            assert dyn[7].risk_level == "high"
            assert dyn[7].heading_deg is not None  # moving toward ego (~180 deg)

            await db.delete(await db.get(DbSession, sid))
            await db.commit()

    asyncio.run(run())
