"""Average precision that charges for how wrong a confusion is, not merely that one happened.

Flat AP treats every misclassification as a total miss. A scooter called a motorcycle and a scooter called
a truck are both scored zero, and they are not the same event: one is a naming difference between two
things that behave identically on a road, and the other is a planner braking for the wrong reason. A
metric that cannot tell them apart cannot be used to choose between two models that differ mainly in which
mistakes they make, which is what a champion and a challenger usually differ in.

Hierarchical AP scores the same detections at every level of the class tree. At the leaf it is ordinary
AP. One level up, a scooter called a motorcycle is CORRECT, because both are two-wheelers and the question
at that level is whether the detector found a two-wheeler. At the root, every detection of any object is
correct if it landed on an object.

The gap between levels is the informative part, and it is what the report leads with:

    leaf AP high, l1 AP barely higher   the model's errors are mostly localisation and missed objects
    leaf AP low, l1 AP much higher      the model finds things and names them wrong
    both low                            it is not finding them

Those three want completely different work, and flat AP reports the same number for all three.

AP AT A COARSER LEVEL IS NOT GUARANTEED TO BE HIGHER, AND THAT IS NOT A BUG. AP is macro-averaged over
the classes present, so each level averages over a different vocabulary. On the champion's gold run the
levels come out leaf 0.072, l1 0.143, l0 0.088, root 0.190: l0 sits below l1 because averaging over three
labels weights a single weak label far more heavily than averaging over twelve does. The comparison that
means something is each level against the LEAF, which is what `gap` reports; comparing two coarse levels
to each other compares two different averages and is not a statement about the model.

WHY NOT JUST WEIGHT THE CONFUSION MATRIX. A weighted matrix tells you the cost of the mistakes among
detections that matched something. AP is defined over the whole precision-recall curve, including the
detections that matched nothing, so a model can improve its confusion cost while getting worse at
detecting. Recomputing AP per level keeps the two coupled.

NumPy is the oracle; this reuses core/accel/ap.py's AP so the leaf level is identical to the number the
engine already reports, by construction rather than by coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from core.accel.ap import average_precision, iou_thresholds_50_95
from core.accel.matching import match_detections


@dataclass(frozen=True)
class LevelAP:
    """AP at one level of the class tree, with the vocabulary it collapsed to.

    `n_classes` is how many distinct labels exist at this level: the root has one, and if a level has as
    many labels as the leaf it is not a level, it is the leaf under another name and the report says so.
    """

    level: str
    ap50: float | None
    ap: float | None
    per_class: dict[str, float]
    n_classes: int
    n_gt: int
    measured: bool
    reason: str | None = None


def _collapse(labels: npt.NDArray[np.int64], mapping: dict[int, str]) -> tuple[npt.NDArray[np.int64],
                                                                              dict[str, int]]:
    """Map leaf class ids to a level's labels, as dense integer ids plus the name table.

    A leaf with no entry at this level keeps its own identity rather than being pooled into a catch-all.
    Pooling it would make the level's AP depend on which classes the tree happened to forget, and an
    incomplete tree would look like a better model.
    """
    names: dict[str, int] = {}
    out = np.empty(labels.size, dtype=np.int64)
    for i, leaf in enumerate(labels.tolist()):
        name = mapping.get(int(leaf), f"__leaf_{leaf}")
        if name not in names:
            names[name] = len(names)
        out[i] = names[name]
    return out, names


def hierarchical_ap(pred_boxes: npt.ArrayLike, pred_scores: npt.ArrayLike, pred_classes: npt.ArrayLike,
                    gt_boxes: npt.ArrayLike, gt_classes: npt.ArrayLike, *,
                    levels: dict[str, dict[int, str]],
                    iou_thresholds: tuple[float, ...] | None = None) -> dict[str, Any]:
    """AP at each named level of a class tree, over one frame's or one corpus's detections.

    `levels` maps a level name to {leaf class id: label at that level}. The caller supplies the tree; this
    module does not know what an l1 is, which is what keeps it usable by a pack that groups differently.
    """
    pb = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    ps = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    pc = np.asarray(pred_classes, dtype=np.int64).reshape(-1)
    gb = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gc = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
    if pb.shape[0] != ps.size or ps.size != pc.size:
        raise ValueError(f"{pb.shape[0]} boxes, {ps.size} scores, {pc.size} classes")
    if gb.shape[0] != gc.size:
        raise ValueError(f"{gb.shape[0]} gt boxes, {gc.size} gt classes")

    thresholds = tuple(iou_thresholds) if iou_thresholds is not None else tuple(iou_thresholds_50_95())
    out: dict[str, LevelAP] = {}

    for level_name, mapping in levels.items():
        if gc.size == 0:
            out[level_name] = LevelAP(level_name, None, None, {}, 0, 0, False,
                                      "no ground truth at this level")
            continue
        p_lab, p_names = _collapse(pc, mapping)
        g_lab, g_names = _collapse(gc, mapping)
        # One shared vocabulary, or a prediction and a gold box carrying the same label would be given
        # different ids and never match.
        shared: dict[str, int] = {}
        for n in list(p_names) + list(g_names):
            shared.setdefault(n, len(shared))
        p_lab = np.array([shared[n] for n in _inv(p_names, p_lab)], dtype=np.int64)
        g_lab = np.array([shared[n] for n in _inv(g_names, g_lab)], dtype=np.int64)

        gt_counts: dict[int, int] = {}
        for c in g_lab.tolist():
            gt_counts[c] = gt_counts.get(c, 0) + 1

        per_thr: dict[float, dict[int, tuple[list[float], list[bool]]]] = {t: {} for t in thresholds}
        for t in thresholds:
            m = match_detections(pb, ps, p_lab, gb, g_lab, iou_thr=t)
            acc = per_thr[t]
            for k in range(p_lab.size):
                s, tps = acc.setdefault(int(p_lab[k]), ([], []))
                s.append(float(ps[k]))
                tps.append(bool(m["tp"][k]))

        ap_by_thr = []
        per_class_50: dict[str, float] = {}
        inv_shared = {v: k for k, v in shared.items()}
        for t in thresholds:
            aps = []
            for cid, n_gt in gt_counts.items():
                scores, tps = per_thr[t].get(cid, ([], []))
                a = average_precision(np.array(scores), np.array(tps, dtype=bool), n_gt)
                if a is None:
                    continue
                aps.append(a)
                if abs(t - 0.5) < 1e-9:
                    per_class_50[inv_shared[cid]] = round(float(a), 6)
            ap_by_thr.append(float(np.mean(aps)) if aps else None)

        ap50 = ap_by_thr[0] if ap_by_thr and abs(thresholds[0] - 0.5) < 1e-9 else None
        valid = [a for a in ap_by_thr if a is not None]
        out[level_name] = LevelAP(
            level=level_name,
            ap50=round(ap50, 6) if ap50 is not None else None,
            ap=round(float(np.mean(valid)), 6) if valid else None,
            per_class=per_class_50, n_classes=len(shared), n_gt=int(gc.size),
            measured=bool(valid),
            reason=None if valid else "no class at this level had enough to compute AP")

    return {"levels": {k: v for k, v in out.items()},
            "gap": _gaps(out),
            "n_predictions": int(ps.size), "n_gt": int(gc.size)}


def _inv(names: dict[str, int], ids: npt.NDArray[np.int64]) -> list[str]:
    back = {v: k for k, v in names.items()}
    return [back[int(i)] for i in ids.tolist()]


def _gaps(levels: dict[str, LevelAP]) -> dict[str, float | None]:
    """AP at each level minus AP at the finest one: how much of the loss is naming rather than finding.

    This is the number the report leads with, because it is the one that says what to work on. A large gap
    means the detector finds objects and calls them the wrong thing; a small one means it is not finding
    them, and no amount of class-boundary labelling will help.
    """
    finest = min(levels.values(), key=lambda v: -v.n_classes, default=None)
    if finest is None or finest.ap50 is None:
        return {}
    return {k: (round(v.ap50 - finest.ap50, 6) if v.ap50 is not None else None)
            for k, v in levels.items()}


__all__ = ["LevelAP", "hierarchical_ap"]
