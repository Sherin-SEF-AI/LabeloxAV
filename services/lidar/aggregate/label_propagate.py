"""M-4D.1: one-shot label in the aggregated scene propagates to every clip frame, the 4D labor multiplier.

The aggregation stores a per-scan pose that maps each scan's ego points INTO the common map frame. So a
cuboid drawn once in the aggregated (static) scene maps back into every scan's ego frame by that scan's
inverse pose: one label becomes a cuboid on every frame of the clip, threaded onto a single 3D track with
one consistent size across the whole trajectory (the Auto4D size-lock, for free, since the dims come from
the single label). The propagated boxes land in review (state=annotate), never auto-accepted.
"""

from __future__ import annotations

import math

import numpy as np

from core.logging import get_logger

log = get_logger("aggregate_label")


def _pose_yaw(pose) -> float:
    """The z-axis rotation (yaw) of a 4x4 pose."""
    return math.atan2(pose[1][0], pose[0][0])


def map_cuboid_to_frame(center_map, yaw_map: float, pose) -> tuple[list[float], float]:
    """A cuboid in the aggregated-map frame -> a scan's ego frame, given the scan's pose (which maps the
    scan's ego points INTO the map). The center transforms by the inverse pose; the yaw subtracts the pose's
    own yaw."""
    inv = np.linalg.inv(np.asarray(pose, dtype=np.float64))
    c = inv @ np.array([center_map[0], center_map[1], center_map[2], 1.0])
    return [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)], round(yaw_map - _pose_yaw(pose), 5)


async def propagate_aggregate_label(agg_id, center, dims, yaw: float, class_id: int,
                                    source: str = "human") -> dict:
    """Create one 3D track from a single aggregate-frame cuboid: a propagated Object3D in every scan of the
    map, each transformed into that scan's ego frame, all sharing the one labeled size."""
    from sqlalchemy import select

    from db.models import AggregatedMap, Frame, Object3D, PointCloud, Track3D
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        agg = await db.get(AggregatedMap, agg_id)
        if agg is None:
            return {"error": "aggregated map not found"}
        poses = (agg.pose_graph or {}).get("poses") or []
        sids = list(agg.session_ids or [])
        if not poses or not sids:
            return {"error": "map has no poses or sessions"}
        clouds = (await db.execute(select(PointCloud).where(PointCloud.session_id.in_(sids))
                                   .order_by(PointCloud.ts_ns))).scalars().all()
        n = min(len(poses), len(clouds))
        if n == 0:
            return {"error": "no clouds for the map sessions"}

        tr = Track3D(session_id=clouds[0].session_id, class_id=class_id,
                     first_ts_ns=clouds[0].ts_ns, last_ts_ns=clouds[n - 1].ts_ns)
        db.add(tr)
        await db.flush()

        created = 0
        for i in range(n):
            cloud, pose = clouds[i], poses[i]
            c, y = map_cuboid_to_frame(center, yaw, pose)
            fr = (await db.execute(select(Frame.frame_id).where(
                Frame.session_id == cloud.session_id, Frame.ts_ns == cloud.ts_ns).limit(1))).scalar()
            db.add(Object3D(cloud_id=cloud.cloud_id, frame_id=fr, track_3d_id=tr.track_3d_id,
                            class_id=class_id, center=c, dims=[float(v) for v in dims], yaw=y,
                            pitch=0.0, roll=0.0, conf=1.0, box_source="lifted", source=source,
                            state="annotate", is_keyframe=(i == 0),
                            attrs={"propagated_from_agg": str(agg_id), "method": "aggregate_propagated",
                                   "is_anchor": i == 0}))
            created += 1
        await db.commit()
    log.info("aggregate.label_propagated", agg=str(agg_id), track=str(tr.track_3d_id), frames=created)
    return {"agg_id": str(agg_id), "track_3d_id": str(tr.track_3d_id), "frames": created,
            "consistent_size": [float(v) for v in dims]}
