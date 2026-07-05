"""SIEVYX curation endpoints: the ranked label-queue priority (model uncertainty fused with embedding-space
rarity) and the queue-composition dashboard (what the label budget is being spent on). Natural-language
scenario search, near-duplicate suppression, and rarity live in the existing /search, /curation, and
/activelearn routers; the platform registry groups them under SIEVYX and this router adds the unified
priority + composition surface. Mounted under /api."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from services.activelearn.selector import score_candidates
from services.api.deps import db_session
from services.sievyx.composition import compose

router = APIRouter()


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
