"""Turn stored dynamics into clip-level maneuvers, so a track can be searched by the shape of its motion.

`services/sievyx/maneuver.py` has recognised maneuvers from a trajectory since M12, and `ClipManeuver` has
had exactly the right columns for as long. Neither was ever joined to the corpus: the recogniser only ran on
a trajectory handed to it in a request body, and the table has never had a writer. So "find me every cut-in"
was a function call nobody could make and a table nobody could query.

What was missing was metric positions per track, and those now exist. `ObjectDynamics` carries a lateral
offset and a distance per detection, which is the (x, y) the recogniser wants once forward is recovered from
the two. Lateral and forward in metres, not pixels, is the whole reason this is worth doing: a lane change
is a couple of metres sideways whatever the object's distance, and in image space it is a number that means
something different in every row of the frame.

Only tracks whose geometry survived the range bound are described. A track seen only near the horizon has no
believable metric path, and inventing one would put a confident maneuver label on arithmetic.
"""

from __future__ import annotations

import math
from uuid import UUID

from sqlalchemy import delete, select

from core.config import get_settings
from core.logging import get_logger
from db.models import ClipManeuver, Frame, Object, ObjectDynamics
from services.dynamics.compute import ipm_max_trajectory_range_m
from services.sievyx.maneuver import recognize

log = get_logger("sievyx.maneuver_run")

# Below this a path is two points and a guess: `trajectory_features` needs three to have a heading at each
# end, and a heading from one segment is the segment itself rather than a turn.
MIN_POINTS = 5

# A track whose metric path is a few centimetres has not manoeuvred, it has jittered. Describing it would
# fill the table with "straight" rows carrying no information and bury the ones that mean something.
MIN_PATH_M = 2.0

# The largest turn a road user completes inside one tracked clip. A U-turn is 180 degrees and is the most
# there is; a vehicle does not exceed it and then keep going in a few seconds of dashcam footage.
#
# This is a gate on the trajectory, not on the maneuver. Monocular IPM positions on this corpus are noisy
# enough that differencing them produces headings that swing without bound: even after a 23 m trajectory
# range bound, a 25 cm step floor and turn accumulated as wrapped per-step deltas, 67.5% of tracks still
# came out as U-turns. Those are not U-turns and no threshold on the classifier can make them into
# something true, because the input is not a path. A track that reports more turn than physically happens
# is evidence that its positions cannot support a maneuver label at all, so none is written and the count
# of refusals is reported instead. The alternative is 1,743 confident wrong labels.
MAX_PLAUSIBLE_TURN_DEG = 200.0


async def compute_session_maneuvers(db, session_id: UUID | str) -> dict:
    """Recognise and store one maneuver per track in this session. Idempotent."""
    sid = session_id if isinstance(session_id, UUID) else UUID(str(session_id))

    rows = (await db.execute(
        select(ObjectDynamics.track_id, ObjectDynamics.ts_ns,
               ObjectDynamics.distance_m, ObjectDynamics.lateral_m)
        .join(Object, Object.object_id == ObjectDynamics.object_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id == sid,
               ObjectDynamics.track_id.isnot(None),
               ObjectDynamics.distance_m.isnot(None),
               ObjectDynamics.lateral_m.isnot(None))
        .order_by(ObjectDynamics.track_id, ObjectDynamics.ts_ns))).all()

    # The trajectory bound, not the reading bound. A distance at 150 m with 25% error is a usable rough
    # statement; differencing two of them for a heading keeps the error and discards the signal. Without
    # this, 79.6% of every track in the corpus classified as a U-turn, because objects at 60 to 200 m jumped
    # over 100 m between consecutive frames and a heading fitted to that swings through a full circle.
    cfg = get_settings()
    max_traj_m = ipm_max_trajectory_range_m(
        cfg.rig.lenses[cfg.rig.camera_lens.get("cam_f", "narrow")].fy, cfg.spatial.camera_height_m)

    by_track: dict = {}
    dropped_far = 0
    for tid, ts, dist, lat in rows:
        if float(dist) > max_traj_m:
            dropped_far += 1
            continue
        # Forward is recovered from the distance and the lateral offset, since the lift stores the hypotenuse
        # rather than the leg. Clamped at zero: floating point can make the two disagree by a hair when an
        # object is almost directly abeam, and a negative under the root is not a position behind the camera.
        fwd = math.sqrt(max(0.0, float(dist) ** 2 - float(lat) ** 2))
        by_track.setdefault(str(tid), []).append({"t": int(ts or 0), "x": float(lat), "y": fwd})

    # Rewritten per session rather than appended, so re-running after a dynamics change corrects the table
    # instead of doubling it. By subquery, because a session can hold more object ids than asyncpg will bind.
    await db.execute(delete(ClipManeuver).where(ClipManeuver.session_id == sid))

    kept = skipped_short = skipped_still = skipped_noisy = 0
    counts: dict = {}
    for tid, traj in by_track.items():
        if len(traj) < MIN_POINTS:
            skipped_short += 1
            continue
        res = recognize(traj)
        if res["features"]["path_len"] < MIN_PATH_M:
            skipped_still += 1
            continue
        if abs(res["features"]["net_turn_deg"]) > MAX_PLAUSIBLE_TURN_DEG:
            # More turn than a road user performs. The path is noise, so it gets no label at all.
            skipped_noisy += 1
            continue
        db.add(ClipManeuver(
            session_id=sid, track_id=UUID(tid), t_in_ns=traj[0]["t"], t_out_ns=traj[-1]["t"],
            maneuver=res["maneuver"], confidence=res["confidence"], features=res["features"]))
        counts[res["maneuver"]] = counts.get(res["maneuver"], 0) + 1
        kept += 1
    await db.commit()

    log.info("sievyx.maneuvers", session=str(sid), tracks=len(by_track), stored=kept,
             too_short=skipped_short, too_still=skipped_still, too_noisy=skipped_noisy,
             points_beyond_range=dropped_far, by_maneuver=counts)
    return {"session_id": str(sid), "tracks": len(by_track), "stored": kept,
            "skipped_too_short": skipped_short, "skipped_not_moving": skipped_still,
            # Refused rather than mislabelled: the loudest number in this result, and it should be.
            "skipped_path_too_noisy": skipped_noisy,
            # Reported so the coverage cost of the trajectory bound is visible rather than inferred.
            "points_beyond_trajectory_range": dropped_far,
            "trajectory_range_m": round(max_traj_m, 1), "by_maneuver": counts}
