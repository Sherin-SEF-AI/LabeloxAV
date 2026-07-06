"""SIEVYX curation endpoints: the ranked label-queue priority (model uncertainty fused with embedding-space
rarity) and the queue-composition dashboard (what the label budget is being spent on). Natural-language
scenario search, near-duplicate suppression, and rarity live in the existing /search, /curation, and
/activelearn routers; the platform registry groups them under SIEVYX and this router adds the unified
priority + composition surface. Mounted under /api."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.activelearn.selector import score_candidates
from services.api.deps import db_session
from services.sievyx.composition import compose
from services.sievyx.discovery import discover_rare_clusters
from services.sievyx.maneuver import recognize
from services.sievyx.odd import coverage_gaps

router = APIRouter()


class ManeuverIn(BaseModel):
    trajectory: list[dict]   # [{t, x, y}, ...] in a BEV/ego frame, metres


@router.post("/sievyx/maneuver")
async def maneuver(payload: ManeuverIn):
    """Recognize the maneuver of a track's trajectory (cut-in, unprotected turn, U-turn, lane change, jaywalk)."""
    return recognize(payload.trajectory)


class OddIn(BaseModel):
    fleet_counts: dict[str, int]
    target_odd: dict[str, float]
    min_gap: float = 0.02


@router.post("/sievyx/odd/gaps")
async def odd_gaps(payload: OddIn):
    """ODD coverage-gap report: scenario cells underrepresented against the target operational design domain."""
    return coverage_gaps(payload.fleet_counts, payload.target_odd, payload.min_gap)


@router.get("/sievyx/discover")
async def discover(limit: int = 400, min_size: int = 2, db: AsyncSession = Depends(db_session)):
    """Unsupervised long-tail discovery: cluster a sample of object embeddings and surface the rarest groups
    for naming, growing the scenario ontology from data."""
    from db.models import ObjectEmbedding
    rows = (await db.execute(
        select(ObjectEmbedding.object_id, ObjectEmbedding.dino_vec).limit(min(max(limit, 8), 2000)))).all()
    if len(rows) < min_size:
        return {"n": len(rows), "clusters": []}
    ids = [str(r[0]) for r in rows]
    vecs = [json.loads(r[1]) if isinstance(r[1], str) else list(r[1]) for r in rows]
    clusters = discover_rare_clusters(vecs, ids, min_size=min_size)
    return {"n": len(rows), "n_clusters": len(clusters),
            "clusters": [{"size": c["size"], "rarity": c["rarity"], "method": c["method"],
                          "rep_ids": c["member_ids"][:6]} for c in clusters[:20]]}


@router.get("/sievyx/priority")
async def priority(session_id: str | None = None, limit: int = 100, db: AsyncSession = Depends(db_session)):
    """The label queue ranked by the combined priority score (uncertainty + diversity + rarity + error +
    recall value). This is the order a human should label in; SIEVYX feeds it to the LabeloxAV workspace."""
    items = await score_candidates(db, session_id)
    return {"pool": len(items), "items": items[: min(max(limit, 1), 1000)]}


@router.get("/sievyx/composition")
async def composition(session_id: str | None = None, top_n: int = 500, db: AsyncSession = Depends(db_session)):
    """The queue-composition dashboard: the class mix, rarity-band split, and mean per-signal contribution of
    the top priority items, so a human can see whether the budget is buying rare tail data or common data."""
    items = await score_candidates(db, session_id)
    return compose(items, top_n=min(max(top_n, 1), 2000))
