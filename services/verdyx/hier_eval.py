"""Scoring a run at every level of the class tree, and reporting the gap between them.

Flat AP is the number the promotion gate compares, and it charges the same for every misclassification. A
scooter called a motorcycle and a scooter called a truck both score zero, which makes AP unable to
distinguish two models that differ mainly in which mistakes they make. That is what a champion and a
challenger usually differ in.

This scores the same prediction plane at leaf, l1, l0 and root, and the gap between the levels is what the
report leads with, because it says what to work on:

    a large leaf-to-l1 gap    the detector finds objects and names them wrong; label class boundaries
    a small one               it is not finding them; more class labels will not help

It reads the prediction plane, not services/training/eval.py. That module runs a separate Ultralytics AP
which never calls core/accel/ap.py, so a hierarchical number derived from it would not be commensurable
with the leaf AP the gate already uses. Taking both from the same path makes the leaf level identical by
construction rather than by coincidence, which is the only way the gap means anything.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.hier_ap import hierarchical_ap
from core.logging import get_logger
from db.models import GoldSet, InferenceRun, Object, Prediction

log = get_logger("hier_eval")


async def evaluate_hierarchical(db: AsyncSession, *, run_id: str, gold_id: str | None = None,
                                score_thr: float = 0.0) -> dict[str, Any]:
    """AP at every level of the pack's class tree, for one run against one sealed gold set."""
    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"measured": False, "reason": "inference run not found", "run_id": run_id}
    if (run.params or {}).get("reconstructed"):
        # No real confidence means no PR curve at any level, for the same reason the gate refuses it.
        return {"measured": False, "run_id": run_id,
                "reason": "this run is reconstructed and has no real confidence distribution"}

    gid = gold_id or run.gold_id
    if not gid:
        return {"measured": False, "run_id": run_id,
                "reason": "the run scored no gold set, so there is no ground truth to score against"}
    gold = await db.get(GoldSet, gid)
    if gold is None or not (gold.object_ids or []):
        return {"measured": False, "run_id": run_id, "gold_id": gid,
                "reason": "gold set not found or empty"}

    from packs.registry import default_pack_id, get_pack
    from services.autolabel.ontology import get_ontology

    tree = get_pack(default_pack_id()).class_tree
    if tree is None:
        return {"measured": False, "run_id": run_id,
                "reason": "this pack defines no class tree, so there are no levels to score at"}
    onto = get_ontology()
    levels = tree.levels_for(onto)

    gold_ids = {UUID(str(o)) for o in gold.object_ids}
    gt = (await db.execute(
        select(Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.object_id.in_(gold_ids)))).all()
    if not gt:
        return {"measured": False, "run_id": run_id, "gold_id": gid,
                "reason": "the sealed gold objects are not in the corpus"}

    frames = {r[0] for r in gt}
    preds = (await db.execute(
        select(Prediction.frame_id, Prediction.class_id, Prediction.bbox, Prediction.conf)
        .where(Prediction.run_id == run.run_id, Prediction.frame_id.in_(list(frames)),
               Prediction.conf >= score_thr))).all()

    pb = np.asarray([list(p[2]) for p in preds], dtype=float).reshape(-1, 4)
    ps = np.asarray([float(p[3] or 0.0) for p in preds], dtype=float)
    pc = np.asarray([int(p[1]) for p in preds], dtype=np.int64)
    gb = np.asarray([list(g[2]) for g in gt], dtype=float).reshape(-1, 4)
    gc = np.asarray([int(g[1]) for g in gt], dtype=np.int64)

    res = hierarchical_ap(pb, ps, pc, gb, gc, levels=levels)
    out: dict[str, Any] = {
        "measured": True, "run_id": run_id, "gold_id": gid, "model_version": run.model_version,
        "n_predictions": res["n_predictions"], "n_gt": res["n_gt"],
        "levels": {name: {"ap50": lv.ap50, "ap": lv.ap, "n_classes": lv.n_classes,
                          "measured": lv.measured, "reason": lv.reason,
                          "per_class": lv.per_class if name != "leaf" else {}}
                   for name, lv in res["levels"].items()},
        "gap": res["gap"],
        "leaf_per_class": res["levels"]["leaf"].per_class if "leaf" in res["levels"] else {},
    }
    leaf = res["levels"].get("leaf")
    l1 = res["levels"].get("l1")
    if leaf is not None and l1 is not None and leaf.ap50 is not None and l1.ap50 is not None:
        out["naming_loss"] = round(l1.ap50 - leaf.ap50, 6)
        out["reading"] = (
            "most of the loss is naming: the detector finds objects and calls them the wrong thing, so "
            "labelling class boundaries is what buys AP"
            if (l1.ap50 - leaf.ap50) > 0.05 else
            "most of the loss is finding: the detector is not detecting the objects at all, so more class "
            "labels will not help")
    log.info("hier_eval.scored", run=run_id, gold=gid,
             leaf=leaf.ap50 if leaf else None, l1=l1.ap50 if l1 else None,
             gap=out.get("naming_loss"))
    return out


__all__ = ["evaluate_hierarchical"]
