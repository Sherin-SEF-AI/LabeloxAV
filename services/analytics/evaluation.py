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

from core.accel.ap import iou_thresholds_50_95, mean_ap
from core.accel.boxes import box_iou_matrix
from core.accel.matching import match_detections
from core.logging import get_logger
from db.models import (
    EvalPatch,
    GoldSet,
    InferenceRun,
    ModelRegistry,
    Object,
    Prediction,
    Review,
)

log = get_logger("analytics_evaluation")

# The label sources that are a machine detection (as opposed to a human-drawn box). Used only by the
# provenance report to tell a confirmed detection from one drawn from scratch.
_MACHINE_SOURCES = ("fused", "auto_accept", "interpolated", "propagated", "relabel", "recall", "vlm_review")


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


async def evaluate_gold_patches(db: AsyncSession, gold_id: str, *, run_id: str,
                                iou_thr: float = 0.5, score_thr: float = 0.0,
                                model_vocabulary: frozenset[str] | None = None) -> dict:
    """Score one inference run against a sealed gold set. Ground truth is the sealed GoldSet.object_ids;
    predictions are the immutable Prediction rows of `run_id` on those frames, and NOTHING from Object. The old
    harness drew predictions from live corpus rows, but human review mutates those rows in place (the confirmed
    detection becomes source="human" and leaves the prediction population), so it scored only the residue.

    Two matches run per frame, for two different questions. The class-AGNOSTIC greedy match populates the
    confusion patches: an off-diagonal cell only exists when a prediction of one class lands on a gold box of
    another, so that cross-class pairing must be allowed. The class-AWARE match (core/accel/matching) is what AP
    is defined on: a true positive requires the same class. They are computed separately and never conflated.
    """
    from services.autolabel.ontology import get_ontology

    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"error": "inference run not found", "run_id": run_id}
    reg = await db.get(ModelRegistry, run.model_version)
    if reg is None:
        # A run whose model is not registered cannot be attributed to a real model; refuse rather than score
        # anonymously (the misattribution the caller-supplied model_version used to cause).
        return {"error": f"model_version '{run.model_version}' not in registry", "run_id": run_id}
    model_version = run.model_version
    reconstructed = bool((run.params or {}).get("reconstructed"))

    gold = await db.get(GoldSet, gold_id)
    if gold is None:
        return {"error": "gold set not found", "gold_id": gold_id}
    gold_ids = {UUID(str(o)) for o in (gold.object_ids or [])}
    if not gold_ids:
        return {"error": "gold set has no objects", "gold_id": gold_id}

    onto = get_ontology()

    gold_rows = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.object_id.in_(gold_ids)))).all()

    # Score the same population the val pass scores, or the two harnesses are not measuring the same thing.
    #
    # `services/training/gold.py:_materialize_aligned` drops every gold object whose ontology class the model
    # was never taught, which is right: you cannot hold a detector responsible for a class it has no output
    # for. This harness scored the sealed set unfiltered, so those objects arrived here as false negatives
    # that were absent over there. That is the whole of the `harness_divergent` flag, and it fires for every
    # model because every model has some gap against a 192-class ontology. Measured on the DashLab detector:
    # 10 objects on 5 frames aligned, 47 on 7 frames here, and the 37 extra were exactly its out-of-vocabulary
    # classes (sedan, rider, pedestrian, cattle, traffic_sign, cycle), taking ap50 from 0.537 to 0.133.
    #
    # Filtering the gold side is enough to align both populations: `by_frame_gt` gates which frames'
    # predictions are fetched at all, so a frame left with no in-vocabulary gold contributes no phantom false
    # positives either.
    n_declared = len(gold_rows)
    if model_vocabulary is not None:
        gold_rows = [r for r in gold_rows if onto.by_id(int(r[2])).name in model_vocabulary]

    by_frame_gt: dict[uuidlib.UUID, list] = {}
    for oid, fid, cid, bbox in gold_rows:
        by_frame_gt.setdefault(fid, []).append((oid, int(cid), list(bbox)))
    if not by_frame_gt:
        if model_vocabulary is not None and n_declared:
            # Refusing beats returning zeros: a model that shares no class with the gold set has not scored
            # badly, it has not been measured, and a 0.0 here would read as the former.
            return {"error": "no gold object is in this model's vocabulary, so it cannot be scored",
                    "gold_id": gold_id, "run_id": run_id, "model_version": model_version,
                    "gold_resolvable": n_declared, "gold_scored": 0}
        return {"error": "gold objects not found in the corpus", "gold_id": gold_id}

    pred_rows = (await db.execute(
        select(Prediction.prediction_id, Prediction.frame_id, Prediction.class_id, Prediction.bbox, Prediction.conf)
        .where(Prediction.run_id == run.run_id,
               Prediction.frame_id.in_(list(by_frame_gt.keys())),
               Prediction.conf >= score_thr))).all()
    by_frame_pred: dict[uuidlib.UUID, list] = {}
    for pid, fid, cid, bbox, conf in pred_rows:
        by_frame_pred.setdefault(fid, []).append((pid, int(cid), list(bbox), float(conf)))

    eval_id = uuidlib.uuid4()
    patches: list[EvalPatch] = []
    n_tp = n_fp = n_fn = 0

    thresholds = list(iou_thresholds_50_95())            # 0.5 .. 0.95 (includes 0.5)
    per_thr: dict[float, dict[int, tuple[list, list]]] = {t: {} for t in thresholds}
    gt_counts: dict[int, int] = {}
    matched_gt_50: dict[int, int] = {}                   # class_id -> matched gold count at IoU 0.5 (recall)

    for fid, gts in by_frame_gt.items():
        preds = by_frame_pred.get(fid, [])
        gt_boxes = np.asarray([g[2] for g in gts], dtype=float).reshape(-1, 4)
        gt_classes = np.asarray([g[1] for g in gts], dtype=np.int64)
        p_boxes = np.asarray([p[2] for p in preds], dtype=float).reshape(-1, 4)
        p_scores = np.asarray([p[3] for p in preds], dtype=float).reshape(-1)
        p_classes = np.asarray([p[1] for p in preds], dtype=np.int64)

        for _, gcid, _ in gts:
            gt_counts[gcid] = gt_counts.get(gcid, 0) + 1

        # Confusion patches: class-agnostic greedy match (cross-class pairing allowed).
        match_gt, matched_iou, gt_matched = _greedy_match(p_boxes, p_scores, gt_boxes, iou_thr)
        for i, (pid, pcid, _pb, pconf) in enumerate(preds):
            j = int(match_gt[i])
            if j < 0:
                n_fp += 1
                patches.append(EvalPatch(eval_id=eval_id, gold_id=gold_id, model_version=model_version,
                                         run_id=run.run_id, prediction_id=pid, frame_id=fid, outcome="fp",
                                         gt_class_id=None, pred_class_id=pcid, iou=None, conf=pconf))
                continue
            _goid, gcid, _gb = gts[j]
            same = gcid == pcid
            n_tp += 1 if same else 0
            n_fp += 0 if same else 1
            patches.append(EvalPatch(eval_id=eval_id, gold_id=gold_id, model_version=model_version,
                                     run_id=run.run_id, prediction_id=pid, frame_id=fid,
                                     outcome="tp" if same else "fp",
                                     gt_class_id=gcid, pred_class_id=pcid,
                                     iou=round(float(matched_iou[i]), 4), conf=pconf))
        for j, (goid, gcid, _gb) in enumerate(gts):
            if not gt_matched[j]:
                n_fn += 1
                patches.append(EvalPatch(eval_id=eval_id, gold_id=gold_id, model_version=model_version,
                                         run_id=run.run_id, object_id=goid, frame_id=fid, outcome="fn",
                                         gt_class_id=gcid, pred_class_id=None, iou=None, conf=None))

        # AP: class-aware match at each IoU threshold (a TP requires the same class).
        if not reconstructed and len(preds):
            for t in thresholds:
                m = match_detections(p_boxes, p_scores, p_classes, gt_boxes, gt_classes, iou_thr=t)
                acc = per_thr[t]
                for k, (_pid, pcid, _b, pconf) in enumerate(preds):
                    s, tps = acc.setdefault(pcid, ([], []))
                    s.append(pconf)
                    tps.append(bool(m["tp"][k]))
                if t == 0.5:
                    for jj, gcid in enumerate(gt_classes.tolist()):
                        if bool(m["gt_matched"][jj]):
                            matched_gt_50[gcid] = matched_gt_50.get(gcid, 0) + 1

    db.add_all(patches)
    await db.commit()

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0

    def _name(cid: int) -> str:
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    per_class_recall = {_name(c): round(matched_gt_50.get(c, 0) / n, 4) for c, n in gt_counts.items() if n}

    result = {
        "eval_id": str(eval_id), "run_id": run_id, "gold_id": gold_id, "model_version": model_version,
        "score_thr": score_thr, "frames": len(by_frame_gt),
        # What was actually scored, not what was sealed. A number over 10 of 400 objects and a number over
        # 400 of 400 are both reported as "ap50" and only these fields tell them apart.
        "gold_declared": len(gold_ids), "gold_resolvable": n_declared, "gold_scored": len(gold_rows),
        "vocabulary_filtered": model_vocabulary is not None,
        "tp": n_tp, "fp": n_fp, "fn": n_fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "per_class_recall": per_class_recall, "patches": len(patches),
    }
    if reconstructed:
        # No raw confidence, so no PR curve and no AP. Return fixed-threshold precision/recall with a caveat so
        # no downstream reads an AP that was never computable.
        result["ap50"] = None
        result["ap50_95"] = None
        result["caveat"] = "reconstructed run: conf unavailable, no PR curve or AP computable"
    else:
        ap50, per_class_ap = mean_ap(per_thr[0.5], gt_counts)
        ap_by_thr = [mean_ap(per_thr[t], gt_counts)[0] for t in thresholds]
        result["ap50"] = round(ap50, 4)
        result["ap50_95"] = round(float(np.mean(ap_by_thr)) if ap_by_thr else 0.0, 4)
        result["per_class_ap50"] = {_name(c): round(a, 4) for c, a in per_class_ap.items()}

    log.info("evaluation.gold_patches", eval_id=str(eval_id), run_id=run_id, gold_id=gold_id,
             model_version=model_version, tp=n_tp, fp=n_fp, fn=n_fn,
             ap50=result.get("ap50"), frames=len(by_frame_gt))
    return result


async def gold_provenance_report(db: AsyncSession, gold_id: str) -> dict:
    """Split the gold set by how each object came to be: a machine detection a human later confirmed
    (recoverable from the first Review.before.source, now that review stamps it), versus one a human drew from
    scratch with no prior detection. Only the drawn-from-scratch population (plus classes never detected at all)
    is a genuine model miss; conflating them with confirmed detections was half the original confusion.
    """
    gold = await db.get(GoldSet, gold_id)
    if gold is None:
        return {"error": "gold set not found", "gold_id": gold_id}
    gold_ids = {UUID(str(o)) for o in (gold.object_ids or [])}
    if not gold_ids:
        return {"error": "gold set has no objects", "gold_id": gold_id}

    # The EARLIEST review per object captured its origin; later reviews are re-edits of an already-human row.
    rows = (await db.execute(
        select(Review.object_id, Review.before, Review.ts_ns)
        .where(Review.object_id.in_(gold_ids)).order_by(Review.ts_ns.asc()))).all()
    first_before: dict = {}
    for oid, before, _ts in rows:
        first_before.setdefault(oid, before or {})

    confirmed_machine = drawn_from_scratch = unknown = 0
    for oid in gold_ids:
        b = first_before.get(oid)
        if b is None:
            drawn_from_scratch += 1          # created directly as human, no prior detection
            continue
        src = b.get("source")
        if src in _MACHINE_SOURCES:
            confirmed_machine += 1
        elif "source" not in b:
            unknown += 1                     # reviewed before source was captured (pre-fix); unrecoverable
        else:
            drawn_from_scratch += 1          # source was human/other at first review

    return {
        "gold_id": gold_id, "n_objects": len(gold_ids),
        "confirmed_machine": confirmed_machine, "drawn_from_scratch": drawn_from_scratch, "unknown": unknown,
        "note": ("only drawn_from_scratch (and classes never detected) are genuine model misses; "
                 "confirmed_machine objects were correct detections a human accepted"),
    }


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

    def _crop_url(r: EvalPatch) -> str | None:
        # A tp/fp patch is a prediction (crop from the prediction bbox); an fn is the missed gold object.
        if r.prediction_id:
            return f"/api/predictions/{r.prediction_id}/crop"
        if r.object_id:
            return f"/api/objects/{r.object_id}/crop"
        return None

    return {"eval_id": eval_id, "count": len(rows), "patches": [
        {"patch_id": str(r.patch_id),
         "prediction_id": str(r.prediction_id) if r.prediction_id else None,
         "object_id": str(r.object_id) if r.object_id else None,
         "frame_id": str(r.frame_id) if r.frame_id else None, "outcome": r.outcome,
         "gt_class_id": r.gt_class_id, "pred_class_id": r.pred_class_id,
         "iou": r.iou, "conf": r.conf, "crop_url": _crop_url(r)} for r in rows]}


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
