"""Budgeted batch selection (M4.0): given a human-hour budget, pick the N highest-value items, suppressing
near-duplicates by DINOv3 cosine so the batch is diverse rather than redundant. Persists an al_selection."""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import AlSelection, ObjectEmbedding
from services.activelearn.selector import score_candidates

log = get_logger("al_budget")


async def select_batch(db: AsyncSession, budget_hours: float, session_id: str | None = None,
                       dedup_cos: float = 0.92, persist: bool = True) -> dict:
    cfg = get_settings().phase4.activelearn
    n_target = max(1, int(round(budget_hours * 3600.0 / cfg.sec_per_item)))
    scored = await score_candidates(db, session_id)
    if not scored:
        return {"batch_id": None, "n_target": n_target, "n_selected": 0, "items": [], "reason": "no candidates"}

    oids = [s["object_id"] for s in scored]
    emb = {str(o): np.asarray(v, np.float32) for o, v in (await db.execute(
        select(ObjectEmbedding.object_id, ObjectEmbedding.dino_vec).where(ObjectEmbedding.object_id.in_(oids)))).all()}

    picked: list[dict] = []
    picked_vecs: list[np.ndarray] = []
    for s in scored:
        if len(picked) >= n_target:
            break
        v = emb.get(s["object_id"])
        if v is not None:
            vn = v / (np.linalg.norm(v) + 1e-9)
            if picked_vecs and max(float(vn @ pv) for pv in picked_vecs) > dedup_cos:
                continue  # near-duplicate of an already-picked item, skip for diversity
        picked.append(s)
        if v is not None:
            picked_vecs.append(v / (np.linalg.norm(v) + 1e-9))

    vals = [p["value"] for p in picked]
    expected = {"total_value": round(float(sum(vals)), 4), "mean_value": round(float(np.mean(vals)), 4) if vals else 0.0,
                "n_rare": sum(1 for p in picked if p["scores"]["rarity"] > 0.5),
                "n_uncertain": sum(1 for p in picked if p["scores"]["uncertainty"] > 0.5),
                "suppressed_duplicates": len(scored) - len(picked) if len(picked) < n_target else None}
    strategy = {"w_uncertainty": cfg.w_uncertainty, "w_diversity": cfg.w_diversity,
                "w_rarity": cfg.w_rarity, "w_error_prone": cfg.w_error_prone, "dedup_cos": dedup_cos}

    batch_id = None
    if persist:
        batch = AlSelection(strategy=strategy, item_ids=[p["object_id"] for p in picked],
                            budget_hours=budget_hours, expected_value=expected, status="open")
        db.add(batch)
        await db.flush()
        batch_id = str(batch.batch_id)
        await db.commit()

    log.info("al.batch", batch_id=batch_id, n=len(picked), target=n_target)
    return {"batch_id": batch_id, "n_target": n_target, "n_selected": len(picked),
            "budget_hours": budget_hours, "strategy": strategy, "expected_value": expected, "items": picked}
