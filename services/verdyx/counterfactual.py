"""Recall before and after one controlled change, which is the only causal thing this engine can measure.

Every other number here is observational. A slice metric says the model is worse on dark frames; it cannot
say whether that is the darkness or the fact that dark frames in this corpus are also mostly highways at
speed. A counterfactual holds the scene fixed and changes one thing, so the difference is attributable.

The design is one sentence: score a run's frames, perturb the same frames, score again with the same
labels, and report the drop per class with its interval.

**The labels never move.** A rider behind an added occlusion is still a rider at the same box. That is what
makes the comparison a comparison, and it is why perturbed frames are never given objects of their own.

**A drop is reported with an interval or not at all.** A recall of 3 of 4 falling to 2 of 4 is not a 25%
regression, it is four objects. `MIN_SUPPORT` is the floor below which a class is reported unmeasured with
its support, because a gate that refused a promotion on four objects would be refusing on noise.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("counterfactual")

# Gold objects of a class before its recall drop is a measurement rather than an anecdote.
MIN_SUPPORT = 30
# Matching rule, the same as every other recall number here so the two are comparable.
IOU_THR = 0.5
# The perturbations the gate actually refuses on. Named rather than "all of them": these two are the
# conditions this corpus is full of and the ones a safety class fails on first.
GATE_PERTURBATIONS = ("occlude", "dusk")
# How far a safety class's recall may fall under a gate perturbation before the promotion is blocked.
MAX_COUNTERFACTUAL_DROP = 0.10


def match_recall(gold_boxes: list, pred_boxes: list, iou_thr: float = IOU_THR) -> tuple[int, int]:
    """How many gold boxes a prediction set covers, and how many there were.

    Greedy at the given IoU, one prediction per gold box, which is the same rule the aggregate recall
    uses. A different matching rule here would make the before and after numbers incomparable with
    everything else in the system while looking like the same statistic.
    """
    from core.accel.boxes import box_iou_matrix

    n = len(gold_boxes)
    if n == 0 or not pred_boxes:
        return 0, n
    m = box_iou_matrix(gold_boxes, pred_boxes)
    used: set[int] = set()
    hit = 0
    for i in range(n):
        best, best_iou = -1, iou_thr
        for j in range(len(pred_boxes)):
            if j in used:
                continue
            v = float(m[i][j])
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            used.add(best)
            hit += 1
    return hit, n


def drop_with_interval(base_hit: int, base_n: int, pert_hit: int, pert_n: int) -> dict:
    """The recall drop and whether the two intervals are far enough apart to call it one.

    Both recalls carry a Wilson interval and the drop is reported as significant only when the perturbed
    upper bound sits below the baseline lower bound. Two point estimates differing is not a finding on a
    sample this size, and a gate that treated it as one would refuse promotions on sampling noise.
    """
    from services.labelops.sampling import wilson_interval

    if base_n == 0:
        return {"measured": False, "reason": "no gold objects of this class to measure against"}
    base = wilson_interval(base_hit, base_n)
    pert = wilson_interval(pert_hit, pert_n or base_n)
    drop = base["p"] - pert["p"]
    return {"measured": True, "support": base_n,
            "recall_before": round(base["p"], 4), "recall_after": round(pert["p"], 4),
            "drop": round(drop, 4),
            "before_interval": [round(base["lo"], 4), round(base["hi"], 4)],
            "after_interval": [round(pert["lo"], 4), round(pert["hi"], 4)],
            # The separation test, stated rather than left for the reader to do with four numbers.
            "separated": bool(pert["hi"] < base["lo"]),
            "significant_drop": bool(drop > 0 and pert["hi"] < base["lo"])}


def summarise(per_class: dict, *, max_drop: float = MAX_COUNTERFACTUAL_DROP,
              safety_classes: set[str] | None = None) -> dict:
    """Which classes fell far enough, and with enough evidence, to be worth blocking a promotion on."""
    blocking, unmeasured = [], []
    for name, row in sorted(per_class.items()):
        if not row.get("measured"):
            unmeasured.append({"class_name": name, "reason": row.get("reason")})
            continue
        if safety_classes is not None and name not in safety_classes:
            continue
        if row["support"] < MIN_SUPPORT:
            unmeasured.append({"class_name": name,
                               "reason": (f"only {row['support']} gold objects; {MIN_SUPPORT} is the "
                                          f"floor for a recall drop to be a measurement")})
            continue
        if row["drop"] > max_drop and row["significant_drop"]:
            blocking.append({"class_name": name, **row})
    return {"blocking": blocking, "unmeasured": unmeasured,
            "measured": [n for n, r in per_class.items()
                         if r.get("measured") and r.get("support", 0) >= MIN_SUPPORT]}


async def counterfactual_eval(db: AsyncSession, *, run_id: str, perturbations: list[str] | None = None,
                              gold_id: str | None = None, score_thr: float = 0.5,
                              max_frames: int = 200, seed: int = 7) -> dict:
    """Score a model on a gold set's frames, then on perturbed copies of the same frames.

    Nothing is written. The perturbed images exist for the length of one forward pass, because persisting
    a perturbed copy of every gold frame for every perturbation would multiply the corpus by the size of
    the perturbation set to answer a question that is the same next week.
    """
    import cv2
    from sqlalchemy import select

    from core.config import get_settings
    from core.storage import get_object_store
    from db.models import Frame, GoldSet, InferenceRun, ModelRegistry, Object
    from services.autolabel.ontology import get_ontology
    from services.verdyx.perturb import DEFAULT_STRENGTHS, apply

    names = list(perturbations or GATE_PERTURBATIONS)
    unknown = [n for n in names if n not in DEFAULT_STRENGTHS]
    if unknown:
        return {"measured": False, "reason": f"unknown perturbations: {unknown}"}

    run = await db.get(InferenceRun, uuid.UUID(str(run_id)))
    if run is None:
        return {"measured": False, "reason": "inference run not found"}
    gid = gold_id or run.gold_id
    if not gid:
        return {"measured": False, "reason": "the run scored no gold set, so there is nothing to hold fixed"}
    gold = await db.get(GoldSet, gid)
    if gold is None or not (gold.object_ids or []):
        return {"measured": False, "reason": "gold set not found or empty", "gold_id": gid}

    reg = await db.get(ModelRegistry, run.model_version)
    if reg is None or not reg.weights_uri:
        return {"measured": False, "reason": "the run's model has no downloadable weights to re-score with",
                "model_version": run.model_version}

    onto = get_ontology()
    rows = (await db.execute(
        select(Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.object_id.in_(gold.object_ids)))).all()
    by_frame: dict = {}
    for fid, cid, bbox in rows:
        by_frame.setdefault(fid, []).append((int(cid), [float(v) for v in bbox]))
    frame_ids = sorted(by_frame, key=str)[:max_frames]
    if not frame_ids:
        return {"measured": False, "reason": "the gold set has no frames", "gold_id": gid}

    frames = {f.frame_id: f for f in (await db.execute(
        select(Frame).where(Frame.frame_id.in_(frame_ids)))).scalars().all()}

    from services.verdyx.inference_run import _infer, _load_model

    settings = get_settings()
    scratch = settings.scratch_path() / "counterfactual"
    scratch.mkdir(parents=True, exist_ok=True)
    local = str(scratch / f"{run.model_version}.pt")
    model, names_list = _load_model(reg.weights_uri, local)
    from services.training.gold import align_model_to_ontology

    idx_to_onto = align_model_to_ontology(names_list)
    imgsz = int(getattr(settings.training, "eval_imgsz", 960))
    store = get_object_store()

    def _score(images: list[np.ndarray]) -> list[list]:
        return _infer(model, images, imgsz, score_thr, settings.gpu.device)

    def _preds_for(dets) -> dict[int, list]:
        out: dict[int, list] = {}
        for cls_idx, conf, box, _top in dets:
            cid = idx_to_onto[cls_idx] if cls_idx < len(idx_to_onto) else None
            if cid is None or conf < score_thr:
                continue
            out.setdefault(int(cid), []).append(box)
        return out

    results: dict[str, dict] = {}
    base_counts: dict[str, list[int]] = {}
    pert_counts: dict[str, dict[str, list[int]]] = {n: {} for n in names}
    scored = 0
    for fid in frame_ids:
        fr = frames.get(fid)
        if fr is None:
            continue
        try:
            img = cv2.imdecode(np.frombuffer(store.get_bytes(fr.img_uri), np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001 - a missing blob skips one frame, never the evaluation
            img = None
        if img is None:
            continue
        scored += 1
        truth: dict[int, list] = {}
        for cid, box in by_frame[fid]:
            truth.setdefault(cid, []).append(box)

        base = _preds_for(_score([img])[0])
        for cid, boxes in truth.items():
            name = onto.by_id(cid).name
            hit, n = match_recall(boxes, base.get(cid, []))
            acc = base_counts.setdefault(name, [0, 0])
            acc[0] += hit
            acc[1] += n

        for pname in names:
            # Occlusion needs a box; the largest gold box on the frame is the one worth occluding, since
            # occluding a distant speck measures nothing.
            box = max((b for bs in truth.values() for b in bs),
                      key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)
            try:
                pimg = apply(pname, img, box=box, seed=seed)
            except ValueError:
                continue
            pred = _preds_for(_score([pimg])[0])
            for cid, boxes in truth.items():
                name = onto.by_id(cid).name
                hit, n = match_recall(boxes, pred.get(cid, []))
                acc = pert_counts[pname].setdefault(name, [0, 0])
                acc[0] += hit
                acc[1] += n

    for pname in names:
        per_class = {}
        for name, (bh, bn) in base_counts.items():
            ph, pn = pert_counts[pname].get(name, [0, bn])
            per_class[name] = drop_with_interval(bh, bn, ph, pn)
        results[pname] = {"per_class": per_class, **summarise(per_class)}

    log.info("counterfactual.evaluated", run_id=str(run_id), frames=scored,
             perturbations=len(names), classes=len(base_counts))
    return {"measured": scored > 0, "run_id": str(run_id), "gold_id": gid,
            "model_version": run.model_version, "frames": scored, "score_thr": score_thr,
            "perturbations": results,
            "reason": None if scored else "no gold frame could be read, so nothing was scored"}
