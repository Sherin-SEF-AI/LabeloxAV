"""Per-object dynamics (P3). Lift each object's ground-contact point (bbox bottom-centre) to a metric
position with the Phase 3 IPM ground-plane, then finite-difference along its M2.0 track, with the ego CAN
speed, to derive: distance, lateral offset, ground speed, heading, closing speed, time-to-collision, and a
risk level. This turns the dataset from perception into something that also supports planning + prediction.

Monocular and IPM-based: there is no LiDAR, so distance is a flat-road estimate and every row records its
method + a confidence. Reuses services.hdmap.georef.ipm_pixel_to_vehicle, the rig intrinsics + camera
height from config, the ontology (for VRU risk), and the ego speed already ingested. ego_speed is m/s.
"""

from __future__ import annotations

import math
from collections import defaultdict
from uuid import UUID

from sqlalchemy import delete, select

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Object, ObjectDynamics
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.domain import safety_l1
from services.hdmap.georef import ipm_pixel_to_vehicle

log = get_logger("dynamics")

MPS_TO_KMH = 3.6
TTC_HIGH_S = 1.5      # below this is an imminent-collision risk
TTC_MED_S = 4.0
MAX_SPEED_KMH = 150.0  # physically-plausible ceiling; a larger delta is a tracking/IPM artifact, not a speed
DT_MIN_S, DT_MAX_S = 0.05, 2.0  # usable frame gap for a finite difference

# How far a flat-road IPM estimate can be believed, and why there has to be a limit at all.
#
# The ray through a pixel below the horizon meets the ground plane at exactly one point, so the lift always
# returns an answer. Near the horizon that answer is worthless: the ray is nearly parallel to the road, and
# the intersection races away. Run unbounded over the real corpus it produced distances with a 95th
# percentile of 410 m and a maximum of 11,978 m, and lateral offsets spanning 5.5 km, on dashcam frames.
#
# The bound is derived rather than picked. For camera height h, an object at forward distance f sits at a
# ray angle below the horizon of theta = atan(h/f), which for f >> h is about h/f. One pixel of vertical
# error is d(theta) = 1/fy, so the relative distance error is
#
#     df/f = d(theta)/theta = f / (fy * h)
#
# and for an error budget tau over a box-bottom precision of dv pixels,
#
#     f_max = tau * fy * h / dv
#
# A detector's box bottom is not pixel-exact; it moves several pixels frame to frame on the same object,
# which is what BOX_BOTTOM_JITTER_PX encodes. Beyond f_max the estimate is reported as unmeasured rather
# than as a number, which is the same rule this codebase already applies to unmeasured confidence: absent
# evidence must not arrive looking like evidence.
IPM_ERROR_BUDGET = 0.25        # a box-bottom jitter may move the estimate by at most this fraction
BOX_BOTTOM_JITTER_PX = 5.0     # observed frame-to-frame movement of a detector's box bottom

# A time to collision is only meaningful while the gap is actually closing at a rate that means something.
# The guard was closing_mps > 1e-3, one millimetre per second, which divided a large distance by nearly zero
# and produced times to collision of up to 36 days. Below this the honest answer is that nothing is closing.
MIN_CLOSING_MPS = 0.5          # about 1.8 km/h of range rate
MAX_TTC_S = 60.0               # beyond a minute it is not a collision estimate, it is arithmetic


def ipm_max_range_m(fy: float, camera_height_m: float) -> float:
    """Farthest forward distance this camera's flat-road lift can support. See the derivation above."""
    return IPM_ERROR_BUDGET * fy * camera_height_m / BOX_BOTTOM_JITTER_PX


def _risk(ttc: float | None, distance: float | None, is_vru: bool, closing_mps: float | None) -> str:
    if ttc is not None and ttc < TTC_HIGH_S:
        return "high"
    if is_vru and distance is not None and distance < 5.0:
        return "high"
    if ttc is not None and ttc < TTC_MED_S:
        return "medium"
    if distance is not None and (distance < 10.0 or (is_vru and distance < 12.0)):
        return "medium"
    return "low"


async def compute_session_dynamics(session_id: UUID) -> dict:
    cfg = get_settings()
    onto = get_ontology()
    rig, sp = cfg.rig, cfg.spatial
    pitch = math.radians(sp.camera_pitch_deg)
    maker = get_sessionmaker()

    async with maker() as db:
        rows = (await db.execute(
            select(Object.object_id, Object.track_id, Object.class_id, Object.bbox,
                   Frame.frame_id, Frame.ts_ns, Frame.cam_id, Frame.width, Frame.height, Frame.ego_speed)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == session_id))).all()
        if not rows:
            return {"objects": 0, "reason": "no objects in this session"}

        # 1. lift every object's ground-contact point to a metric (forward, lateral) position
        recs: dict = {}
        for oid, tid, cid, bbox, fid, ts, cam, w, h, ego in rows:
            lens = rig.camera_lens.get(cam, "narrow")
            K = rig.lenses[lens]
            scale = (w or rig.ref_width) / rig.ref_width
            fx, fy, cx, cy = K.fx * scale, K.fy * scale, (w or rig.ref_width) / 2.0, (h or 1080) / 2.0
            u, v = (bbox[0] + bbox[2]) / 2.0, bbox[3]   # bottom-centre = where the object meets the road
            fl = ipm_pixel_to_vehicle(u, v, fx, fy, cx, cy, sp.camera_height_m, pitch,
                                      dist=K.dist, fisheye=K.model == "fisheye")
            forward, lateral, dist = (None, None, None)
            if fl is not None:
                f_max = ipm_max_range_m(fy, sp.camera_height_m)
                cand_fwd, cand_lat = fl
                # Behind the camera is not a distance, and beyond the usable range the lift is arithmetic
                # rather than measurement. Both are dropped to None so a consumer sees missing evidence
                # instead of a confident number.
                if cand_fwd > 0 and math.hypot(cand_fwd, cand_lat) <= f_max:
                    forward, lateral = cand_fwd, cand_lat
                    dist = math.hypot(forward, lateral)
            recs[oid] = {"oid": oid, "tid": tid, "cid": cid, "fid": fid, "ts": ts, "ego": float(ego or 0.0),
                         "forward": forward, "lateral": lateral, "dist": dist}

        # 2. finite-difference along each track (ordered by time) for speed / heading / closing / ttc
        by_track: dict = defaultdict(list)
        for r in recs.values():
            if r["tid"] is not None and r["forward"] is not None:
                by_track[r["tid"]].append(r)
        dyn: dict = {}
        for items in by_track.values():
            items.sort(key=lambda x: x["ts"])
            for i, r in enumerate(items):
                speed = heading = closing = ttc = None
                if i > 0:
                    p = items[i - 1]
                    dt = (r["ts"] - p["ts"]) / 1e9
                    if DT_MIN_S <= dt <= DT_MAX_S:
                        # object ground forward speed = relative range rate + ego ground speed over the
                        # interval. Use the interval-average ego (not just the endpoint) so the conversion
                        # matches the averaging the finite difference already implies. Straight-line
                        # assumption: no ego yaw-rate model, so turns add some phantom lateral speed.
                        ego_avg = (r["ego"] + p["ego"]) / 2.0
                        fwd_speed = (r["forward"] - p["forward"]) / dt + ego_avg
                        lat_speed = (r["lateral"] - p["lateral"]) / dt
                        cand = math.hypot(fwd_speed, lat_speed) * MPS_TO_KMH
                        if cand <= MAX_SPEED_KMH:  # reject implausible jumps (bad track / IPM artifact)
                            speed = cand
                            heading = math.degrees(math.atan2(lat_speed, fwd_speed))
                            closing_mps = -(r["dist"] - p["dist"]) / dt
                            closing = closing_mps * MPS_TO_KMH
                            if closing_mps >= MIN_CLOSING_MPS:
                                cand_ttc = r["dist"] / closing_mps
                                # A time to collision beyond the horizon is not a collision estimate. It is
                                # a large distance divided by an almost-still gap, and reporting it invites
                                # a downstream rule to treat it as a measurement of safety.
                                ttc = cand_ttc if cand_ttc <= MAX_TTC_S else None
                dyn[r["oid"]] = {"speed": speed, "heading": heading, "closing": closing, "ttc": ttc}

        # 3. write one row per object (distance always when liftable; speed/etc for tracked frames)
        #
        # Deleted through a subquery on the session's frames rather than an IN list of every object id.
        # asyncpg caps a statement at 32,767 parameters, so a dense session exceeded it and failed outright:
        # one of 215 real sessions produced no dynamics at all for that reason. A subquery binds one
        # parameter regardless of how many objects the session holds.
        await db.execute(delete(ObjectDynamics).where(
            ObjectDynamics.frame_id.in_(select(Frame.frame_id).where(Frame.session_id == session_id))))
        tracked = 0
        for oid, r in recs.items():
            d = dyn.get(oid, {})
            try:
                is_vru = onto.by_id(r["cid"]).l1 in safety_l1()
            except Exception:  # noqa: BLE001 -- an OOD class_id must not crash the whole session
                is_vru = False
            closing_mps = (d["closing"] / MPS_TO_KMH) if d.get("closing") is not None else None
            risk = _risk(d.get("ttc"), r["dist"], is_vru, closing_mps) if r["dist"] is not None else None
            if d.get("speed") is not None:
                tracked += 1
            db.add(ObjectDynamics(
                object_id=oid, track_id=r["tid"], frame_id=r["fid"], ts_ns=r["ts"],
                distance_m=round(r["dist"], 2) if r["dist"] is not None else None,
                lateral_m=round(r["lateral"], 2) if r["lateral"] is not None else None,
                speed_kmh=round(d["speed"], 1) if d.get("speed") is not None else None,
                closing_speed_kmh=round(d["closing"], 1) if d.get("closing") is not None else None,
                heading_deg=round(d["heading"], 1) if d.get("heading") is not None else None,
                ttc_s=round(d["ttc"], 2) if d.get("ttc") is not None else None,
                risk_level=risk, method="ipm_mono_v1",
                confidence=0.6 if r["dist"] is not None else 0.2))
        await db.commit()

    out = {"session_id": str(session_id), "objects": len(recs), "tracked_with_speed": tracked,
           "with_distance": sum(1 for r in recs.values() if r["dist"] is not None)}
    log.info("dynamics.computed", **out)
    return out
