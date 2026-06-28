"""Intelligent frame extraction (M1.6): replace fixed-rate sampling with content and novelty-aware
selection. Over a session's frames (ordered by time), keep a frame when the scene changes (DINOv3
cosine to the previous kept frame drops), when it carries rare events (rare-class objects), or when
enough has passed; drop near-static redundant stretches. Honors a frame budget and the existing quality
gate. Re-runnable over existing sessions: writes frame.selected + frame.novelty_score, never deletes raw.

Composes with dedup (M1.1): it only re-selects frames that are not dedup-redundant, so the two layers
stack (exact near-dups removed by dedup, near-static non-novel stretches thinned here).
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import or_, select, update

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, FrameEmbedding, Object
from db.session import get_sessionmaker

log = get_logger("extract_smart")


async def smart_select_session(session_id: UUID) -> dict:
    cfg = get_settings().intel.extract
    maker = get_sessionmaker()

    async with maker() as db:
        # reset extraction marks while preserving dedup: re-select every non-dup-redundant frame.
        await db.execute(update(Frame).where(
            Frame.session_id == session_id,
            or_(Frame.dup_group_id.is_(None), Frame.is_dup_canonical.is_(True))).values(selected=True))
        await db.commit()

        rows = (await db.execute(
            select(Frame.frame_id, FrameEmbedding.dino_vec)
            .join(FrameEmbedding, FrameEmbedding.frame_id == Frame.frame_id)
            .where(Frame.session_id == session_id, Frame.selected.is_(True), FrameEmbedding.dino_vec.isnot(None))
            .order_by(Frame.ts_ns))).all()
        # object count + rare flag per frame
        from services.autolabel.gate import is_rare
        from services.autolabel.ontology import get_ontology
        onto = get_ontology()
        ocount: dict = {}
        rare: dict = {}
        obj_rows = (await db.execute(
            select(Object.frame_id, Object.class_id).join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == session_id, Object.state != "rejected"))).all()
        for fid, cid in obj_rows:
            ocount[fid] = ocount.get(fid, 0) + 1
            if is_rare(cid, onto):
                rare[fid] = True

    if len(rows) < 2:
        return {"session_id": str(session_id), "frames": len(rows), "kept": len(rows), "dropped": 0}

    frames = [{"id": r[0], "vec": np.asarray(r[1], dtype=np.float32)} for r in rows]
    n = len(frames)

    # greedy: keep frame 0, then keep on scene-change / rare / event-dense, drop near-static
    kept_idx = [0]
    novelty = {frames[0]["id"]: 1.0}
    last = frames[0]["vec"]
    for i in range(1, n):
        fid = frames[i]["id"]
        cos = float(frames[i]["vec"] @ last)
        nov = round(1.0 - cos, 4)
        novelty[fid] = nov
        is_rare_f = fid in rare
        dense = ocount.get(fid, 0) >= 8
        gap = i - kept_idx[-1]
        keep = is_rare_f or dense or cos < cfg.scene_change_cos or gap >= max(cfg.min_gap_frames * 4, 8)
        if cos > cfg.diversity_cos and not is_rare_f and gap < cfg.min_gap_frames:
            keep = False
        if keep:
            kept_idx.append(i)
            last = frames[i]["vec"]

    # budget: cap kept fraction, keeping the most novel (rare frames always survive)
    budget = max(1, int(cfg.target_budget_frac * n))
    if len(kept_idx) > budget:
        ranked = sorted(kept_idx, key=lambda i: (frames[i]["id"] in rare, novelty[frames[i]["id"]]), reverse=True)
        kept_idx = set(ranked[:budget])
    else:
        kept_idx = set(kept_idx)

    async with maker() as db:
        for i, f in enumerate(frames):
            await db.execute(update(Frame).where(Frame.frame_id == f["id"])
                             .values(selected=i in kept_idx, novelty_score=novelty[f["id"]]))
        await db.commit()

    out = {"session_id": str(session_id), "frames": n, "kept": len(kept_idx), "dropped": n - len(kept_idx),
           "kept_frac": round(len(kept_idx) / n, 3)}
    log.info("extract_smart.done", **out)
    return out
