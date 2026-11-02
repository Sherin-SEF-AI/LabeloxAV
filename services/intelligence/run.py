"""Scenario mining driver: track -> trajectories -> events -> scenario index, for one session.

    python -m services.intelligence.run --session <uuid>
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from uuid import UUID

import click
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKTElement
from sqlalchemy import cast, delete, func, select

from core.config import get_settings
from core.logging import get_logger, setup_logging
from db.models import Frame, Object, Scenario, Track
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.autolabel.track.tracker import track_camera_botsort
from services.intelligence.events import detect_events
from services.intelligence.tracking import Det, TrackResult, track_camera
from services.intelligence.trajectory import FrameCtx, build_trajectory

log = get_logger("mine")


def _tracks_from_rows(tracks: Sequence[Track], dets_by_cam: dict[str, list[Det]], objects_by_id: dict) -> list[TrackResult]:
    """A session's stored tracks in the tracker's own shape, members ordered by time, for mining."""
    members: dict = {}
    cams: dict = {}
    for cam, dets in dets_by_cam.items():
        for d in dets:
            members.setdefault(d.object_id, (cam, d))
    by_track: dict = {}
    for oid, (cam, d) in members.items():
        tid = objects_by_id[oid].track_id
        if tid is not None:
            by_track.setdefault(tid, []).append(d)
            cams[tid] = cam
    out = []
    for t in tracks:
        ms = sorted(by_track.get(t.track_id, []), key=lambda d: d.ts_ns)
        if ms:
            out.append(TrackResult(track_id=t.track_id, cam_id=cams[t.track_id], class_id=t.class_id,
                                   first_ts_ns=ms[0].ts_ns, last_ts_ns=ms[-1].ts_ns, members=ms))
    return out


async def mine_session(session_id: UUID) -> dict:
    onto = get_ontology()
    maker = get_sessionmaker()

    async with maker() as db:
        lat = func.ST_Y(cast(Frame.gnss, Geometry))
        lon = func.ST_X(cast(Frame.gnss, Geometry))
        frows = (
            await db.execute(
                select(Frame.frame_id, Frame.cam_id, Frame.ts_ns, Frame.width, Frame.height,
                       Frame.ego_speed, lat, lon).where(Frame.session_id == session_id)
            )
        ).all()
        frame_ctx: dict = {
            r.frame_id: FrameCtx(width=r.width, height=r.height, ego_speed=r.ego_speed, lat=r[6], lon=r[7])
            for r in frows
        }
        ego_series = sorted(
            {r.ts_ns: r.ego_speed for r in frows if r.ego_speed is not None}.items()
        )

        orows = (
            await db.execute(
                select(Object, Frame.cam_id, Frame.ts_ns)
                .join(Frame, Object.frame_id == Frame.frame_id)
                .where(Frame.session_id == session_id, Object.state != "rejected")
            )
        ).all()

        # DINOv3 object embeddings as the BoT-SORT appearance feature (Phase 1, no extra re-ID model).
        import numpy as np

        from db.models import ObjectEmbedding

        emb_by_id: dict = {}
        erows = (await db.execute(
            select(ObjectEmbedding.object_id, ObjectEmbedding.dino_vec)
            .join(Object, Object.object_id == ObjectEmbedding.object_id)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == session_id))).all()
        for oid, vec in erows:
            emb_by_id[oid] = np.asarray(vec, dtype=np.float32)

        dets_by_cam: dict[str, list[Det]] = {}
        objects_by_id: dict = {}
        for obj, cam_id, ts_ns in orows:
            objects_by_id[obj.object_id] = obj
            b = obj.bbox
            dets_by_cam.setdefault(cam_id, []).append(
                Det(object_id=obj.object_id, frame_id=obj.frame_id, ts_ns=ts_ns, cam_id=cam_id,
                    bbox=(b[0], b[1], b[2], b[3]), class_id=obj.class_id, embedding=emb_by_id.get(obj.object_id))
            )

        backend = get_settings().intelligence.tracker.backend
        all_tracks: list[TrackResult] = []
        assignment: dict = {}
        switches_by_track: dict = {}
        existing = (await db.execute(select(Track).where(Track.session_id == session_id))).scalars().all()
        reused = bool(existing)
        if reused:
            # Mined before: mine from the tracks it already has. Tracking again would leave the old tracks
            # beside new ones, and replacing them would cascade away the track events people wrote on them,
            # so neither is safe here. POST /tracks/retrack replaces a session's tracks, and says so.
            all_tracks = _tracks_from_rows(existing, dets_by_cam, objects_by_id)
        else:
            for cam, dets in dets_by_cam.items():
                if backend == "bot_sort":
                    a, tracks, sw = track_camera_botsort(dets)
                    switches_by_track.update(sw)
                else:
                    a, tracks = track_camera(dets)
                assignment.update(a)
                all_tracks.extend(tracks)

        # Persist new tracks, then point objects at them. Reused tracks are already stored.
        trajs: dict[str, object] = {}
        track_class: dict[str, int] = {}
        for tr in all_tracks:
            tj = build_trajectory(tr, frame_ctx)
            trajs[str(tr.track_id)] = tj
            track_class[str(tr.track_id)] = tr.class_id
            if reused:
                continue
            sw = switches_by_track.get(str(tr.track_id))
            db.add(Track(track_id=tr.track_id, session_id=session_id, class_id=tr.class_id,
                         first_ts_ns=tr.first_ts_ns, last_ts_ns=tr.last_ts_ns,
                         trajectory={"points": tj.points, "summary": tj.summary},
                         id_switch_flags={"events": sw} if sw else None,
                         tracker_version=f"{backend}+dinov3"))
        await db.flush()
        for object_id, track_id in assignment.items():
            objects_by_id[object_id].track_id = track_id
        if not reused:
            await db.flush()
            from services.temporal.tube import score_session_tracks

            await score_session_tracks(db, session_id)

        # Scenarios are derived from the tracks and nothing refers to one, so a re-mine replaces them rather
        # than adding a second copy of every event.
        await db.execute(delete(Scenario).where(Scenario.session_id == session_id))

        scenarios = detect_events(all_tracks, trajs, frame_ctx, ego_series, onto)

        by_type: dict[str, int] = {}
        for sc in scenarios:
            actor_classes = [onto.by_id(track_class[a]).name for a in sc.actors if a in track_class]
            meta = dict(sc.meta)
            if actor_classes:
                meta["actor_classes"] = actor_classes
            geo = WKTElement(f"POINT({sc.lon} {sc.lat})", srid=4326) if sc.lat is not None and sc.lon is not None else None
            db.add(Scenario(session_id=session_id, type=sc.type, t_in_ns=sc.t_in_ns, t_out_ns=sc.t_out_ns,
                            actors=sc.actors, criticality=sc.criticality, geo=geo, tags=sc.tags, meta=meta))
            by_type[sc.type] = by_type.get(sc.type, 0) + 1

        await db.commit()

    summary = {
        "session_id": str(session_id),
        "tracks": len(all_tracks),
        "tracks_reused": reused,
        "objects_tracked": len(assignment),
        "scenarios": len(scenarios),
        "by_type": by_type,
    }
    log.info("mine.done", **summary)
    return summary


@click.command()
@click.option("--session", "session_id", required=True, type=str)
def main(session_id: str) -> None:
    setup_logging(get_settings().log_level)
    summary = asyncio.run(mine_session(UUID(session_id)))
    click.echo(summary)


if __name__ == "__main__":
    main()
