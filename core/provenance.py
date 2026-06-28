"""Provenance builders and the single-walk chain (Principle 09).

From any object you can traverse: object -> track -> frame -> session -> raw clip URI ->
sensor serial + calibration hash -> model versions -> reviewer (if touched) -> dataset commit.
This walk is the audit and debug spine and must never break.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.schemas import PathProposal, Provenance


def build_provenance(
    proposals: list[PathProposal],
    raw_conf: dict[str, float],
    agreement: bool,
    mask_box_disagree: bool,
    ontology_version: str,
    calibrated_from: float | None = None,
    notes: list[str] | None = None,
) -> Provenance:
    return Provenance(
        proposals=proposals,
        agreement=agreement,
        mask_box_disagree=mask_box_disagree,
        raw_conf=raw_conf,
        calibrated_from=calibrated_from,
        ontology_version=ontology_version,
        notes=notes or [],
    )


async def walk_provenance(session: AsyncSession, object_id: UUID) -> dict:
    """Return the complete lineage chain for one object. Raises if the chain is broken."""
    from db.models import DatasetCommit, Frame, Object, Review, Track
    from db.models import Session as DbSession

    obj = await session.get(Object, object_id)
    if obj is None:
        raise ValueError(f"object {object_id} not found")

    frame = await session.get(Frame, obj.frame_id)
    if frame is None:
        raise ValueError(f"provenance broken: frame {obj.frame_id} missing for object {object_id}")

    drive = await session.get(DbSession, frame.session_id)
    if drive is None:
        raise ValueError(f"provenance broken: session {frame.session_id} missing")

    track = await session.get(Track, obj.track_id) if obj.track_id else None

    reviews = (
        (await session.execute(select(Review).where(Review.object_id == object_id).order_by(Review.ts_ns)))
        .scalars()
        .all()
    )

    commit = await session.get(DatasetCommit, drive.commit_id) if drive.commit_id else None

    model_versions = []
    if isinstance(obj.provenance, dict):
        for p in obj.provenance.get("proposals", []):
            mv = p.get("model_version")
            if mv and mv not in model_versions:
                model_versions.append(mv)

    return {
        "object_id": str(obj.object_id),
        "class_id": obj.class_id,
        "state": obj.state,
        "source": obj.source,
        "conf": obj.conf,
        "track_id": str(track.track_id) if track else None,
        "frame": {
            "frame_id": str(frame.frame_id),
            "ts_ns": frame.ts_ns,
            "cam_id": frame.cam_id,
            "img_uri": frame.img_uri,
        },
        "session": {
            "session_id": str(drive.session_id),
            "vehicle_id": drive.vehicle_id,
            "raw_uri": drive.raw_uri,
            "mcap_uri": drive.mcap_uri,
            "sensors": drive.sensors,
            "ontology_version": drive.ontology_version,
        },
        "model_versions": model_versions,
        "reviews": [
            {"reviewer": r.reviewer, "action": r.action, "ts_ns": r.ts_ns} for r in reviews
        ],
        "dataset_commit": commit.commit_id if commit else None,
    }
