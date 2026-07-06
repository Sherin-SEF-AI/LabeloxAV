"""ORACLYX persistence and distillation export. record_consensus() writes the consensus verdict onto the
pseudo_label side table and sets the object state (auto_accept on consensus, review on disagreement), so the
easy majority is auto-truthed and only disagreements reach the human queue. export_distillation() gathers the
consensus pseudo-labels plus their soft targets into a distillation set tied to a Release, for training the
real-time edge model FORGYX will later compile."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Object, PseudoLabel
from db.session import get_sessionmaker
from services.oraclyx.consensus import vote

log = get_logger("oraclyx.run")


async def record_consensus(object_id: UUID, fusion: dict, auto_paths: list[dict],
                           fusion_run_id: str | None = None) -> dict:
    """Vote fusion against the auto-label paths for one object, persist the verdict, and route the object."""
    r = vote(fusion, auto_paths)
    async with get_sessionmaker()() as db:
        await db.merge(PseudoLabel(object_id=object_id, consensus=r["consensus"],
                                   consensus_score=r["consensus_score"], voters=r["voters"],
                                   fusion_run_id=fusion_run_id))
        obj = await db.get(Object, object_id)
        if obj is not None and obj.source != "human":
            obj.state = "auto_accept" if r["consensus"] else "review"
            obj.source = "fused"
        await db.commit()
    log.info("oraclyx.consensus", object=str(object_id), consensus=r["consensus"],
             score=r["consensus_score"], routed="auto_accept" if r["consensus"] else "review")
    return r


async def consensus_summary(db: AsyncSession) -> dict:
    """The consensus/disagreement board: how many pseudo-labels were auto-accepted vs routed to humans."""
    total = (await db.execute(select(func.count()).select_from(PseudoLabel))).scalar() or 0
    auto = (await db.execute(
        select(func.count()).select_from(PseudoLabel).where(PseudoLabel.consensus.is_(True)))).scalar() or 0
    return {"total": total, "auto_accepted": auto, "routed_to_human": total - auto,
            "auto_accept_rate": round(auto / total, 4) if total else 0.0}


async def export_distillation(db: AsyncSession, min_score: float = 0.5, limit: int = 100000) -> dict:
    """Build the distillation manifest: every consensus pseudo-label with its object geometry and a soft target
    (the consensus score), the training signal for the edge model. Returned as a manifest with lineage to the
    exact object ids; sealing it into a Release is the export step FORGYX consumes."""
    rows = (await db.execute(
        select(PseudoLabel.object_id, PseudoLabel.consensus_score, Object.class_id, Object.bbox, Object.frame_id)
        .join(Object, Object.object_id == PseudoLabel.object_id)
        .where(PseudoLabel.consensus.is_(True), PseudoLabel.consensus_score >= min_score)
        .limit(limit))).all()
    manifest = [{"object_id": str(oid), "frame_id": str(fid), "class_id": cid, "bbox": bbox,
                 "soft_target": float(score)} for oid, score, cid, bbox, fid in rows]
    return {"n": len(manifest), "min_score": min_score, "manifest": manifest}
