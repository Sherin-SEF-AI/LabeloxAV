"""Milestone C: cross-modal QA at the seams, the audit instrument for the 3D and 4D program. The 2D-to-3D
reprojection consistency engine already exists (services/lidar/quality3d.check_2d3d_consistency, persisted as
QualityFlag3D); this adds the timestamp-seam check (a non-camera event whose instant has no aligned camera
frame is a seam defect) and assembles both into one worst-first QA queue wired into the review flow. Most
multimodal quality failures happen at these seams, so this is the permanent regression instrument that the
calibration milestone is measured against.
"""

from __future__ import annotations

from core.logging import get_logger
from services.intelligence.timeline import nearest_index

log = get_logger("consistency_qa")


def timestamp_seam_flags(event_rows: list[dict], frame_ts: list[int], max_skew_ns: int) -> list[dict]:
    """Flag each non-camera event whose nearest camera frame is beyond max_skew_ns: a misaligned timestamp
    (or no frame at all) means the event has no visual anchor at its instant. event_rows carry
    {event_id, kind, modality, t_start_ns}."""
    flags = []
    for e in event_rows:
        fi = nearest_index(frame_ts, e["t_start_ns"])
        skew = abs(frame_ts[fi] - e["t_start_ns"]) if fi is not None else None
        if skew is None or skew > max_skew_ns:
            flags.append({"event_id": e["event_id"], "kind": e["kind"], "modality": e["modality"],
                          "skew_ns": skew, "reason": "no_visual_anchor" if skew is None else "timestamp_seam"})
    return sorted(flags, key=lambda f: (f["skew_ns"] is not None, -(f["skew_ns"] or 0)))


async def consistency_qa_queue(session_id, max_skew_ns: int = 200_000_000) -> dict:
    """The worst-first cross-modal QA queue: the persisted 2D-3D reprojection-inconsistency flags ranked by
    severity, plus the timestamp-seam flags for events with no aligned frame. The single queue a reviewer
    works and a gate reads alongside the gold eval."""
    from sqlalchemy import select

    from db.models import Frame, PointCloud, QualityFlag3D, TimelineEvent
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        box_flags = (await db.execute(
            select(QualityFlag3D).join(PointCloud, QualityFlag3D.cloud_id == PointCloud.cloud_id)
            .where(PointCloud.session_id == session_id, QualityFlag3D.kind == "box_2d3d_inconsistent")
            .order_by(QualityFlag3D.score.desc()))).scalars().all()
        frame_ts = [int(t) for t in (await db.execute(
            select(Frame.ts_ns).where(Frame.session_id == session_id).order_by(Frame.ts_ns))).scalars().all()]
        events = (await db.execute(select(TimelineEvent).where(
            TimelineEvent.session_id == session_id, TimelineEvent.modality != "scene"))).scalars().all()

    event_rows = [{"event_id": str(e.event_id), "kind": e.kind, "modality": e.modality,
                   "t_start_ns": e.t_start_ns} for e in events if e.modality != "geo"]
    seams = timestamp_seam_flags(event_rows, frame_ts, max_skew_ns)
    box = [{"object_3d_id": str(f.object_3d_id) if f.object_3d_id else None, "score": f.score,
            "detail": f.detail} for f in box_flags]
    log.info("consistency_qa.queue", session=str(session_id), box=len(box), seams=len(seams))
    return {"session_id": str(session_id), "box_2d3d_worst_first": box, "timestamp_seams": seams,
            "total": len(box) + len(seams)}
