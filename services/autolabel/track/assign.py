"""Re-runnable BoT-SORT tracking pass over a session (M2.0): clears the session's tracks, associates the
existing per-frame fused detections with BoT-SORT + DINOv3 appearance, and writes stable Track rows
(trajectory, id_switch_flags, tracker_version) + points each object at its track. This is the tracking
portion of the scenario miner, exposed standalone so the track-review UI shows real cross-frame tracklets
without a full mine.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import delete, select

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Object, ObjectEmbedding, Track
from db.session import get_sessionmaker
from services.autolabel.track.tracker import track_camera_botsort
from services.intelligence.tracking import Det
from services.intelligence.trajectory import FrameCtx, build_trajectory

log = get_logger("retrack")


async def retrack_session(session_id: UUID) -> dict:
    maker = get_sessionmaker()
    backend = get_settings().intelligence.tracker.backend
    async with maker() as db:
        await db.execute(delete(Track).where(Track.session_id == session_id))  # objects.track_id SET NULL
        await db.commit()

        frows = (await db.execute(
            select(Frame.frame_id, Frame.width, Frame.height, Frame.ego_speed).where(Frame.session_id == session_id))).all()
        frame_ctx = {r.frame_id: FrameCtx(width=r.width, height=r.height, ego_speed=r.ego_speed, lat=None, lon=None)
                     for r in frows}

        orows = (await db.execute(
            select(Object, Frame.cam_id, Frame.ts_ns).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Frame.session_id == session_id, Object.state != "rejected"))).all()
        erows = (await db.execute(
            select(ObjectEmbedding.object_id, ObjectEmbedding.dino_vec)
            .join(Object, Object.object_id == ObjectEmbedding.object_id)
            .join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id))).all()
        emb = {oid: np.asarray(v, np.float32) for oid, v in erows}

        dets_by_cam: dict = {}
        objects_by_id: dict = {}
        for obj, cam_id, ts_ns in orows:
            objects_by_id[obj.object_id] = obj
            b = obj.bbox
            dets_by_cam.setdefault(cam_id, []).append(
                Det(object_id=obj.object_id, frame_id=obj.frame_id, ts_ns=ts_ns, cam_id=cam_id,
                    bbox=(b[0], b[1], b[2], b[3]), class_id=obj.class_id, embedding=emb.get(obj.object_id)))

        assignment: dict = {}
        all_tracks, switches = [], {}
        for dets in dets_by_cam.values():
            if backend == "bot_sort":
                a, tracks, sw = track_camera_botsort(dets)
                switches.update(sw)
            else:
                from services.intelligence.tracking import track_camera
                a, tracks = track_camera(dets)
            assignment.update(a)
            all_tracks.extend(tracks)

        n_switches = 0
        for tr in all_tracks:
            tj = build_trajectory(tr, frame_ctx)
            sw = switches.get(str(tr.track_id))
            n_switches += len(sw) if sw else 0
            db.add(Track(track_id=tr.track_id, session_id=session_id, class_id=tr.class_id,
                         first_ts_ns=tr.first_ts_ns, last_ts_ns=tr.last_ts_ns,
                         trajectory={"points": tj.points, "summary": tj.summary},
                         id_switch_flags={"events": sw} if sw else None,
                         tracker_version=f"{backend}+dinov3"))
        await db.flush()
        for oid, tid in assignment.items():
            objects_by_id[oid].track_id = tid
        await db.commit()

    out = {"session_id": str(session_id), "tracks": len(all_tracks),
           "objects_tracked": len(assignment), "id_switches": n_switches, "tracker": f"{backend}+dinov3"}
    log.info("retrack.done", **out)
    return out
