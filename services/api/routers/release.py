"""Release endpoints: the immutable dataset-commit registry and the cross-plane lineage walk (a release back
to its sessions and their SANYX health and CALYX calibration state). Content-hashing and export live in the
existing services/export; this adds the reproducible content fingerprint and the lineage query. Mounted under
/api."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DatasetCommit
from services.api.deps import db_session
from services.release.lineage import gold_lineage

router = APIRouter()


@router.get("/release/{commit_id}/verify")
async def verify(commit_id: str, db: AsyncSession = Depends(db_session)):
    """Enforce release immutability: recompute the content fingerprint from the live objects the commit's
    slice selects and compare it to the one sealed at export time. A mismatch means an annotation was mutated
    after the release was cut (the commit_id, hashing only object ids, would not have caught it)."""
    from services.export.dataset import SliceSpec, fetch_records, seal_content_fingerprint

    c = await db.get(DatasetCommit, commit_id)
    if c is None:
        return {"error": "commit not found", "commit_id": commit_id}
    if not c.content_fingerprint:
        return {"commit_id": commit_id, "fingerprinted": False,
                "detail": "sealed before content fingerprinting (migration 0061); cannot verify"}
    records = await fetch_records(SliceSpec(**c.slice_spec))
    recomputed = seal_content_fingerprint(SliceSpec(**c.slice_spec), records, c.ontology_version)
    immutable = recomputed == c.content_fingerprint
    return {"commit_id": commit_id, "fingerprinted": True, "immutable": immutable,
            "sealed_fingerprint": c.content_fingerprint, "recomputed_fingerprint": recomputed,
            "object_count_now": len(records), "object_count_sealed": c.object_count}


@router.get("/release/registry")
async def registry(limit: int = 100, db: AsyncSession = Depends(db_session)):
    """The Release registry: content-addressed dataset commits with their parent lineage, newest first."""
    rows = (await db.execute(
        select(DatasetCommit).order_by(DatasetCommit.created_at.desc()).limit(min(max(limit, 1), 500)))).scalars().all()
    return {"commits": [
        {"commit_id": c.commit_id, "parent_id": c.parent_id, "object_count": c.object_count,
         "object_3d_count": c.object_3d_count, "ontology_version": c.ontology_version,
         "created_at": c.created_at.isoformat() if c.created_at else None} for c in rows]}


@router.get("/release/gold/{gold_id}/lineage")
async def lineage(gold_id: str, db: AsyncSession = Depends(db_session)):
    """Resolve a sealed gold set back to its sessions and each session's health and calibration state, so a
    single query answers whether a release drew from sessions a plane later flagged."""
    return await gold_lineage(db, gold_id)
