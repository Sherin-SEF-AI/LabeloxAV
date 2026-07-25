"""Evaluation drill-down: turn a confusion cell into the actual crops behind it.

A confusion matrix tells you that pedestrians are being called poles; it does not show you the pedestrians.
This scores one label source against a sealed gold set and records every individual outcome as an `eval_patch`
row, so clicking a cell opens the real objects through the existing GET /api/objects/{id}/crop.

Matching is deliberately CLASS-AGNOSTIC, unlike core/accel/matching.py:match_detections, which requires a
shared class and therefore can only ever produce tp/fp/fn. To populate an off-diagonal confusion cell you have
to let a prediction match a gold box of a DIFFERENT class and then compare the two classes; that pairing is
exactly what "pedestrian predicted where gold says pole" means. The spatial work still reuses the accelerated
core/accel/boxes.py:box_iou_matrix.

Outcomes recorded, matching how a confusion matrix is read:
    tp  matched, same class                  -> diagonal cell
    fp  matched, different class             -> off-diagonal cell (gt_class_id, pred_class_id both set)
    fp  unmatched prediction                 -> predicted-where-nothing-is (gt_class_id null)
    fn  unmatched gold object                -> missed (pred_class_id null)
"""

from __future__ import annotations

import uuid as uuidlib
from uuid import UUID

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.boxes import box_iou_matrix
from core.logging import get_logger
from db.models import EvalPatch, Frame, GoldSet, Object

log = get_logger("analytics_evaluation")


def _greedy_match(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                  gt_boxes: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-agnostic greedy IoU match, predictions in descending score order.
    Returns (match_gt (P,) index or -1, matched_iou (P,), gt_matched (G,) bool)."""
    P, G = pred_boxes.shape[0], gt_boxes.shape[0]
    match_gt = np.full(P, -1, dtype=np.int64)
    matched_iou = np.zeros(P, dtype=float)
    gt_matched = np.zeros(G, dtype=bool)
    if P == 0 or G == 0:
        return match_gt, matched_iou, gt_matched

    iou = box_iou_matrix(pred_boxes, gt_boxes)          # (P, G), accelerated
    for i in np.argsort(-pred_scores, kind="stable"):
        row = iou[i].copy()
        row[gt_matched] = 0.0                            # a gold box can only be claimed once
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            match_gt[i] = j
            matched_iou[i] = float(row[j])
            gt_matched[j] = True
    return match_gt, matched_iou, gt_matched


async def evaluate_gold_patches(db: AsyncSession, gold_id: str, *,
                                pred_sources: list[str] | None = None,
                                iou_thr: float = 0.5, model_version: str | None = None) -> dict:
    """Score the machine labels on a gold set's frames against that gold set, writing one eval_patch per
    outcome. `pred_sources` selects which label source counts as the prediction (default: the machine ones).

    Returns the run summary plus the confusion cells, so the caller can render the matrix and then drill in.
    """
    gold = await db.get(GoldSet, gold_id)
    if gold is None:
        return {"error": "gold set not found", "gold_id": gold_id}
    gold_ids = {UUID(str(o)) for o in (gold.object_ids or [])}
    if not gold_ids:
        return {"error": "gold set has no objects", "gold_id": gold_id}

    sources = pred_sources or ["fused", "auto_accept", "interpolated", "propagated", "relabel"]

    # gold objects, grouped by frame
    gold_rows = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.object_id.in_(gold_ids)))).all()
    by_frame_gt: dict[uuidlib.UUID, list] = {}
    for oid, fid, cid, bbox in gold_rows:
        by_frame_gt.setdefault(fid, []).append((oid, cid, list(bbox)))
    if not by_frame_gt:
        return {"error": "gold objects not found in the corpus", "gold_id": gold_id}

    # candidate predictions on those same frames, excluding the gold objects themselves
    pred_rows = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox, Object.conf)
        .where(Object.frame_id.in_(list(by_frame_gt.keys())),
               Object.source.in_(sources),
               Object.object_id.notin_(gold_ids)))).all()
    by_frame_pred: dict[uuidlib.UUID, list] = {}
    for oid, fid, cid, bbox, conf in pred_rows:
        by_frame_pred.setdefault(fid, []).append((oid, cid, list(bbox), float(conf or 0.0)))

    eval_id = uuidlib.uuid4()
    patches: list[EvalPatch] = []
    n_tp = n_fp = n_fn = 0

    for fid, gts in by_frame_gt.items():
        preds = by_frame_pred.get(fid, [])
        gt_boxes = np.asarray([g[2] for g in gts], dtype=float).reshape(-1, 4)
        p_boxes = np.asarray([p[2] for p in preds], dtype=float).reshape(-1, 4)
        p_scores = np.asarray([p[3] for p in preds], dtype=float).reshape(-1)

        match_gt, matched_iou, gt_matched = _greedy_match(p_boxes, p_scores, gt_boxes, iou_thr)

        for i, (poid, pcid, _pb, pconf) in enumerate(preds):
            j = int(match_gt[i])
            if j < 0:
                n_fp += 1
                patches.append(EvalPatch(eval_id=eval_id, gold_id=gold_id, model_version=model_version,
                                         object_id=poid, frame_id=fid, outcome="fp",
                                         gt_class_id=None, pred_class_id=pcid,
                                         iou=None, conf=pconf))
                continue
            g_oid, g_cid, _gb = gts[j]
            same = int(g_cid) == int(pcid)
            if same:
                n_tp += 1
            else:
                n_fp += 1
            patches.append(EvalPatch(eval_id=eval_id, gold_id=gold_id, model_version=model_version,
                                     object_id=poid, frame_id=fid, outcome="tp" if same else "fp",
                                     gt_class_id=g_cid, pred_class_id=pcid,
                                     iou=round(float(matched_iou[i]), 4), conf=pconf))

        for j, (g_oid, g_cid, _gb) in enumerate(gts):
            if not gt_matched[j]:
                n_fn += 1
                patches.append(EvalPatch(eval_id=eval_id, gold_id=gold_id, model_version=model_version,
                                         object_id=g_oid, frame_id=fid, outcome="fn",
                                         gt_class_id=g_cid, pred_class_id=None, iou=None, conf=None))

    db.add_all(patches)
    await db.commit()

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    log.info("evaluation.gold_patches", eval_id=str(eval_id), gold_id=gold_id,
             tp=n_tp, fp=n_fp, fn=n_fn, frames=len(by_frame_gt))
    return {"eval_id": str(eval_id), "gold_id": gold_id, "frames": len(by_frame_gt),
            "tp": n_tp, "fp": n_fp, "fn": n_fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "patches": len(patches)}


async def confusion_cells(db: AsyncSession, eval_id: str, limit: int = 300) -> dict:
    """The confusion matrix for one evaluation, as (gt_class, pred_class, count) cells with names attached."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    rows = (await db.execute(
        select(EvalPatch.gt_class_id, EvalPatch.pred_class_id, EvalPatch.outcome, func.count())
        .where(EvalPatch.eval_id == UUID(eval_id))
        .group_by(EvalPatch.gt_class_id, EvalPatch.pred_class_id, EvalPatch.outcome)
        .order_by(func.count().desc()).limit(limit))).all()

    def _name(cid):
        if cid is None:
            return None
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    cells = [{"gt_class_id": g, "pred_class_id": p, "gt_class": _name(g), "pred_class": _name(p),
              "outcome": o, "count": int(n)} for g, p, o, n in rows]
    return {"eval_id": eval_id, "cells": cells}


async def cell_patches(db: AsyncSession, eval_id: str, *, gt_class_id: int | None = None,
                       pred_class_id: int | None = None, outcome: str | None = None,
                       limit: int = 120) -> dict:
    """The individual objects behind one confusion cell, for the patch grid. Each item carries the object_id
    the UI renders through /api/objects/{id}/crop."""
    stmt = select(EvalPatch).where(EvalPatch.eval_id == UUID(eval_id))
    if gt_class_id is not None:
        stmt = stmt.where(EvalPatch.gt_class_id == gt_class_id)
    if pred_class_id is not None:
        stmt = stmt.where(EvalPatch.pred_class_id == pred_class_id)
    if outcome:
        stmt = stmt.where(EvalPatch.outcome == outcome)
    rows = (await db.execute(stmt.order_by(EvalPatch.conf.desc().nullslast()).limit(limit))).scalars().all()
    return {"eval_id": eval_id, "count": len(rows), "patches": [
        {"patch_id": str(r.patch_id), "object_id": str(r.object_id) if r.object_id else None,
         "frame_id": str(r.frame_id) if r.frame_id else None, "outcome": r.outcome,
         "gt_class_id": r.gt_class_id, "pred_class_id": r.pred_class_id,
         "iou": r.iou, "conf": r.conf,
         "crop_url": f"/api/objects/{r.object_id}/crop" if r.object_id else None} for r in rows]}


async def list_evaluations(db: AsyncSession, limit: int = 50) -> list[dict]:
    """Recent evaluation runs with their outcome mix."""
    rows = (await db.execute(
        select(EvalPatch.eval_id, EvalPatch.gold_id, EvalPatch.outcome, func.count(),
               func.min(EvalPatch.created_at))
        .group_by(EvalPatch.eval_id, EvalPatch.gold_id, EvalPatch.outcome)
        .order_by(func.min(EvalPatch.created_at).desc()).limit(limit * 3))).all()
    runs: dict[str, dict] = {}
    for eid, gid, outcome, n, created in rows:
        r = runs.setdefault(str(eid), {"eval_id": str(eid), "gold_id": gid, "tp": 0, "fp": 0, "fn": 0,
                                       "created_at": created.isoformat() if created else None})
        r[outcome] = int(n)
    return list(runs.values())[:limit]


async def delete_evaluation(db: AsyncSession, eval_id: str) -> dict:
    n = (await db.execute(delete(EvalPatch).where(EvalPatch.eval_id == UUID(eval_id)))).rowcount
    await db.commit()
    return {"deleted": True, "eval_id": eval_id, "patches": int(n or 0)}
