"""Cross-plane release lineage (M6). One spine makes the query the whole product exists for trivial: given a
release (or any object set), resolve back to the exact sessions the samples came from and each session's
SANYX health and CALYX calibration state. This answers "which deployed model was trained on samples from
sessions CALYX later flagged for drift" with a single walk, because every plane keys off the same session id."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CalibrationValidation, Frame, Object, SessionHealth
from db.models import Session as DbSession


async def resolve_lineage(db: AsyncSession, object_ids: list[str]) -> dict:
    """Resolve an object set to its sessions and each session's health + calibration state."""
    if not object_ids:
        return {"n_objects": 0, "sessions": []}
    frame_ids = set((await db.execute(
        select(Object.frame_id).where(Object.object_id.in_(object_ids)))).scalars())
    if not frame_ids:
        return {"n_objects": len(object_ids), "sessions": []}
    session_ids = set((await db.execute(
        select(Frame.session_id).where(Frame.frame_id.in_(frame_ids)))).scalars())

    out = []
    for sid in session_ids:
        sess = await db.get(DbSession, sid)
        health = (await db.execute(
            select(SessionHealth).where(SessionHealth.session_id == sid)
            .order_by(SessionHealth.created_at.desc()).limit(1))).scalar_one_or_none()
        calib = (await db.execute(
            select(CalibrationValidation.severity).where(CalibrationValidation.session_id == sid,
                                                         CalibrationValidation.severity.isnot(None))
            .order_by(CalibrationValidation.created_at.desc()).limit(1))).scalar_one_or_none()
        out.append({
            "session_id": str(sid),
            "vehicle_id": sess.vehicle_id if sess else None,
            "city": sess.city if sess else None,
            "sanyx_score": health.score if health else None,
            "sanyx_decision": (health.decision or health.verdict) if health else None,
            "calyx_severity": calib,
        })
    flagged = [s for s in out if s["sanyx_decision"] in ("quarantine", "fail") or s["calyx_severity"] == "block"]
    return {"n_objects": len(object_ids), "n_sessions": len(out), "sessions": out,
            "flagged_sessions": [s["session_id"] for s in flagged]}


async def gold_lineage(db: AsyncSession, gold_id: str) -> dict:
    """Resolve a sealed gold set's lineage (gold_set stores the frozen object_ids)."""
    from db.models import GoldSet
    gs = await db.get(GoldSet, gold_id)
    if gs is None:
        return {"error": "gold set not found"}
    return {"gold_id": gold_id, **await resolve_lineage(db, [str(o) for o in (gs.object_ids or [])])}
