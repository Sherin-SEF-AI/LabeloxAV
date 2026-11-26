"""M-4D.1: one-shot label in the aggregated scene propagates to every clip frame, the 4D labor multiplier.

The aggregation stores a per-scan pose that maps each scan's ego points INTO the common map frame. So a
cuboid drawn once in the aggregated (static) scene maps back into every scan's ego frame by that scan's
inverse pose: one label becomes a cuboid on every frame of the clip, threaded onto a single 3D track with
one consistent size across the whole trajectory (the Auto4D size-lock, for free, since the dims come from
the single label). The propagated boxes land in review (state=annotate), never auto-accepted.
"""

from __future__ import annotations

import asyncio
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


# ---- #3 label once in the session scene frame (ego poses), then project into every camera ----

def points_in_box_count(xyz: np.ndarray, center, dims, yaw: float, pad: float = 0.0) -> int:
    """How many points fall inside an oriented cuboid (yaw only), with an optional margin."""
    if len(xyz) == 0:
        return 0
    rel = np.asarray(xyz, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    lx = rel[:, 0] * c + rel[:, 1] * s
    ly = -rel[:, 0] * s + rel[:, 1] * c
    inside = ((np.abs(lx) <= dims[0] / 2 + pad) & (np.abs(ly) <= dims[1] / 2 + pad)
              & (np.abs(rel[:, 2]) <= dims[2] / 2 + pad))
    return int(inside.sum())


def local_ground_z(xyz: np.ndarray, center, dims, yaw: float, ring_m: float = 2.0,
                   min_points: int = 20) -> float | None:
    """The road height under a cuboid in one scan: a low percentile of the points in a ring around its
    footprint (inside the footprint the object's own points would lift it). None when the ring is too empty
    to say, e.g. the box is out of range or behind an occluder in this scan."""
    if len(xyz) == 0:
        return None
    rel = np.asarray(xyz, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    lx = np.abs(rel[:, 0] * c + rel[:, 1] * s)
    ly = np.abs(-rel[:, 0] * s + rel[:, 1] * c)
    hl, hw = dims[0] / 2, dims[1] / 2
    in_outer = (lx <= hl + ring_m) & (ly <= hw + ring_m)
    in_inner = (lx <= hl) & (ly <= hw)
    near_z = np.abs(rel[:, 2]) <= dims[2] / 2 + 1.5
    ring = in_outer & ~in_inner & near_z
    if int(ring.sum()) < min_points:
        return None
    return float(np.percentile(np.asarray(xyz)[ring, 2], 10))


async def propagate_scene_label(session_id, center, dims, yaw: float, class_id: int, *,
                                ground_snap: bool = True, max_range_m: float = 60.0,
                                project: bool = True, job_id=None, user_id=None, progress=None,
                                loader=None) -> dict:
    """One cuboid drawn in a session's scene frame becomes a 3D track on every scan and a 2D box in every
    camera frame at the same instant, all linked: Object3D.track_3d_id, Object.track_id (one 2D track per
    camera), Object3D.object_id (the 2D box in the camera that sees it largest), Track3D.track_id.

    Everything lands in review: a label-once box is right for a static object and wrong for a moving one, and
    only a person looking at the frames can say which this is.
    """
    from sqlalchemy import select

    from db.models import Frame, Object, Object3D, Track, Track3D
    from db.session import get_sessionmaker
    from services.calibration.resolve import resolve_calibration
    from services.lidar.aggregate.scene_frame import session_scene_poses, world_cuboid_to_ego
    from services.lidar.boxes import cuboid_image_box
    from services.lidar.ingest.store import load_cloud

    loader = loader or load_cloud
    dims = [float(v) for v in dims]
    maker = get_sessionmaker()
    async with maker() as db:
        poses, summary = await session_scene_poses(db, session_id)
        if not poses:
            return {"error": summary.get("reason") or summary.get("error") or "no placed scans"}
        frames = (await db.execute(select(Frame.frame_id, Frame.ts_ns, Frame.cam_id, Frame.width, Frame.height)
                                   .where(Frame.session_id == session_id))).all()
    by_ts: dict[int, list] = {}
    for f in frames:
        by_ts.setdefault(int(f.ts_ns), []).append(f)

    def _place(uri: str, c_ego: list[float], y_ego: float) -> tuple[list[float], int, bool]:
        # blocking: object-store read plus numpy, run off the event loop
        xyz = np.asarray(loader(uri).xyz)
        gz = local_ground_z(xyz, c_ego, dims, y_ego) if ground_snap else None
        if gz is not None:
            c_ego = [c_ego[0], c_ego[1], round(gz + dims[2] / 2.0, 4)]
        return c_ego, points_in_box_count(xyz, c_ego, dims, y_ego, pad=0.1), gz is not None

    placed: list[dict] = []
    skipped: dict[str, int] = {}
    for i, sp in enumerate(poses):
        c_ego, y_ego = world_cuboid_to_ego(center, yaw, sp.T_world_ego)
        rng = math.hypot(c_ego[0], c_ego[1])
        if rng > max_range_m:
            skipped["beyond range"] = skipped.get("beyond range", 0) + 1
            continue
        c_ego, n_pts, snapped = await asyncio.to_thread(_place, sp.cloud_uri, c_ego, y_ego)
        placed.append({"sp": sp, "center": c_ego, "yaw": y_ego, "range": round(rng, 2), "n_points": n_pts,
                       "ground_snapped": snapped})
        if progress is not None and i % 10 == 0:
            await progress(0.05 + 0.55 * (i + 1) / len(poses), "placing on scans")
    if not placed:
        return {"error": "the cuboid is beyond range of every scan", "skipped": skipped}

    cal_cache: dict = {}
    boxes_2d = 0
    no_view: dict[str, int] = {}
    async with maker() as db:
        tr3 = Track3D(session_id=session_id, class_id=class_id, first_ts_ns=placed[0]["sp"].ts_ns,
                      last_ts_ns=placed[-1]["sp"].ts_ns,
                      trajectory={"frame": "session_enu", "static": True, "method": "scene_label_once",
                                  "scene_label": {"center": [float(v) for v in center], "dims": dims,
                                                  "yaw": float(yaw)},
                                  "points": [{"ts_ns": p["sp"].ts_ns, "xyz": [float(v) for v in center],
                                              "dims": dims, "yaw": float(yaw)} for p in placed]})
        db.add(tr3)
        await db.flush()
        tracks_2d: dict[str, Track] = {}
        for k, p in enumerate(placed):
            sp = p["sp"]
            fr = by_ts.get(sp.ts_ns, [])
            o3 = Object3D(cloud_id=sp.cloud_id, frame_id=(fr[0].frame_id if fr else None),
                          track_3d_id=tr3.track_3d_id, class_id=class_id, center=p["center"], dims=dims,
                          yaw=p["yaw"], pitch=0.0, roll=0.0, conf=0.9, box_source="native", source="human",
                          state="review", is_keyframe=(k == 0),
                          attrs={"method": "scene_label_once", "n_points": p["n_points"], "range_m": p["range"],
                                 "ground_snapped": p["ground_snapped"]},
                          provenance={"tool": "scene3d_label_once", "job_id": str(job_id) if job_id else None,
                                      "scene_label": {"center": [float(v) for v in center], "dims": dims,
                                                      "yaw": float(yaw)},
                                      "by": str(user_id) if user_id else None})
            db.add(o3)
            await db.flush()
            if not project:
                continue
            best = None
            for f in fr:
                key = (f.cam_id, int(f.width), int(f.height))
                if key not in cal_cache:
                    cal_cache[key] = await resolve_calibration(session_id, f.cam_id, int(f.width), int(f.height))
                box, why = cuboid_image_box(p["center"], dims, p["yaw"], f.cam_id, int(f.width), int(f.height),
                                            calib=cal_cache[key])
                if box is None:
                    no_view[why] = no_view.get(why, 0) + 1
                    continue
                t2 = tracks_2d.get(f.cam_id)
                if t2 is None:
                    t2 = Track(session_id=session_id, class_id=class_id, first_ts_ns=sp.ts_ns, last_ts_ns=sp.ts_ns,
                               tracker_version="scene3d_label_once")
                    db.add(t2)
                    await db.flush()
                    tracks_2d[f.cam_id] = t2
                t2.last_ts_ns = sp.ts_ns
                o2 = Object(frame_id=f.frame_id, track_id=t2.track_id, class_id=class_id, bbox=box, conf=0.9,
                            source="propagated", state="review",
                            provenance={"method": "scene_label_once", "object_3d_id": str(o3.object_3d_id),
                                        "track_3d_id": str(tr3.track_3d_id),
                                        "job_id": str(job_id) if job_id else None,
                                        "cal_source": cal_cache[key].source,
                                        "lidar_points": p["n_points"]})
                db.add(o2)
                await db.flush()
                boxes_2d += 1
                area = (box[2] - box[0]) * (box[3] - box[1])
                if best is None or area > best[0]:
                    best = (area, o2.object_id)
            if best is not None:
                o3.object_id = best[1]
            if progress is not None and k % 20 == 0:
                await progress(0.6 + 0.35 * (k + 1) / len(placed), "projecting into cameras")
        if tracks_2d:
            first_cam = sorted(tracks_2d)[0]
            tr3.track_id = tracks_2d.get("cam_front", tracks_2d[first_cam]).track_id
        await db.commit()
        track_3d_id = tr3.track_3d_id
    res = {"track_3d_id": str(track_3d_id), "scans": len(placed), "boxes_2d": boxes_2d,
           "tracks_2d": {cam: str(t.track_id) for cam, t in tracks_2d.items()},
           "with_points": sum(1 for p in placed if p["n_points"] > 0),
           "ground_snapped": sum(1 for p in placed if p["ground_snapped"]),
           "skipped": skipped, "not_in_view": no_view, "pose_summary": summary}
    log.info("scene_label.propagated", track=str(track_3d_id), scans=len(placed), boxes_2d=boxes_2d)
    return res
