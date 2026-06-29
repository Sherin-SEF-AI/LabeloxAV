"""Eval harness + regression gate (Principle: no pipeline ships if gold metrics drop).

evaluate() runs a model on a dataset's val split and returns mAP plus per-class AP. regression_gate()
compares a candidate against a baseline and decides promote/reject: overall mAP must not drop and no
class may regress past a tolerance.

Honest note: until a human-verified gold set exists, val labels are the (noisy) auto-labels, so the
numbers measure agreement-with-corpus, not absolute accuracy. The harness is unchanged once gold
arrives: point the val split at human-reviewed frames (the builder already routes them there).
"""

from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger

log = get_logger("eval")


def evaluate(weights: str, data_yaml: str, split: str = "val", imgsz: int = 960) -> dict:
    from ultralytics import YOLO

    device = get_settings().gpu.device
    model = YOLO(weights)
    res = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, verbose=False, plots=False)

    per_class: dict[str, float] = {}
    per_class_pr: dict[str, dict] = {}
    try:
        names = res.names if hasattr(res, "names") else model.names
        ap50 = res.box.ap50  # per-evaluated-class AP@50, aligned to ap_class_index by position
        p = res.box.p        # per-class precision, same alignment
        r = res.box.r        # per-class recall, same alignment
        for i, ci in enumerate(res.box.ap_class_index):
            name = names[int(ci)]
            per_class[name] = round(float(ap50[i]), 4) if i < len(ap50) else 0.0
            per_class_pr[name] = {
                "precision": round(float(p[i]), 4) if i < len(p) else 0.0,
                "recall": round(float(r[i]), 4) if i < len(r) else 0.0,
                "ap50": per_class[name],
            }
    except Exception as exc:  # noqa: BLE001
        log.warning("eval.per_class_failed", error=str(exc))

    out = {
        "map50": round(float(res.box.map50), 4),
        "map": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "per_class": per_class,
        "per_class_pr": per_class_pr,
        # Flat class_name -> recall, so the champion recall gate (fail-closed on safety-class recall)
        # can read it directly from gold_metrics.
        "per_class_recall": {name: pr["recall"] for name, pr in per_class_pr.items()},
    }
    # Segmentation models also report mask metrics; absent for detect models.
    if getattr(res, "seg", None) is not None:
        try:
            out["mask_map50"] = round(float(res.seg.map50), 4)
            out["mask_map"] = round(float(res.seg.map), 4)
        except Exception:  # noqa: BLE001
            pass
    return out


def safe_miou_report(weights: str, data_yaml: str, split: str = "val", imgsz: int = 960) -> dict:
    """Run a val pass and score the class confusion matrix with the ontology-derived Safe-mIoU, so
    unsafe confusions (pedestrian->pole) are penalized far harder than benign ones (sedan->hatchback)."""
    from ultralytics import YOLO

    from services.autolabel.ontology import get_ontology
    from services.training.safe_miou import BACKGROUND_ID, safe_miou

    settings = get_settings()
    onto = get_ontology()
    model = YOLO(weights)
    # plots=True is required: ultralytics only accumulates the confusion matrix when plotting is on,
    # otherwise res.confusion_matrix.matrix is all zeros and Safe-mIoU reads as undefined.
    res = model.val(data=data_yaml, split=split, imgsz=imgsz, device=settings.gpu.device, verbose=False, plots=True)

    names = dict(res.names if hasattr(res, "names") else model.names)
    nc = len(names)
    # map each model class index to an ontology id (gold split names are ontology names)
    class_ids: list[int] = []
    for i in range(nc):
        try:
            class_ids.append(onto.by_name(names[i]).id)
        except Exception:  # noqa: BLE001
            class_ids.append(-1)

    try:
        matrix = res.confusion_matrix.matrix  # (nc+1, nc+1), last index is background
        # Keep the background row/col so a missed VRU (true class predicted nothing) and a false positive
        # (background predicted as a class) are scored, not discarded. Dropping them is exactly why a weak
        # detector that misses everything read as undefined; the spec wants "pedestrian versus background".
        k = matrix.shape[0]
        ids_bg = (class_ids + [BACKGROUND_ID])[:k]
        score = safe_miou(matrix, ids_bg, onto, settings.m9.safety_weight)
        offdiag = float(matrix.sum() - sum(matrix[i][i] for i in range(k)))
    except Exception as exc:  # noqa: BLE001
        log.warning("safe_miou.failed", error=str(exc))
        return {"safe_miou": None, "n_classes": nc, "offdiag_mass": None}

    return {"safe_miou": round(score, 4) if score is not None else None, "n_classes": nc,
            "offdiag_mass": offdiag, "safety_weight": settings.m9.safety_weight}


def regression_gate(
    candidate: dict, baseline: dict, min_map_delta: float = 0.0, max_class_drop: float = 0.15,
    min_safe_miou: float | None = None,
) -> dict:
    reasons: list[str] = []
    delta = round(candidate["map50"] - baseline["map50"], 4)
    if delta < min_map_delta:
        reasons.append(f"map50 delta {delta} < required {min_map_delta}")

    # Optional Safe-mIoU floor (off by default: gating on a noisy first gold set would block
    # legitimate promotions). When set, the candidate must not confuse VRUs unsafely below the floor.
    if min_safe_miou is not None:
        sm = candidate.get("safe_miou")
        if sm is not None and sm < min_safe_miou:
            reasons.append(f"safe_miou {sm} < required {min_safe_miou}")

    # Only classes present in BOTH vocabularies are comparable. A class the candidate's vocabulary
    # does not contain (e.g. a COCO baseline's 'bicycle' vs the ontology's 'cycle', or 'person' vs
    # 'pedestrian') is a naming difference, not a regression, so it is skipped.
    regressed = []
    cand_classes = candidate.get("per_class", {})
    for cls, base_ap in baseline.get("per_class", {}).items():
        if cls not in cand_classes:
            continue
        cand_ap = cand_classes[cls]
        if base_ap - cand_ap > max_class_drop:
            regressed.append({"class": cls, "from": base_ap, "to": cand_ap})
    if regressed:
        reasons.append(f"{len(regressed)} classes regressed past {max_class_drop}")

    promote = len(reasons) == 0
    return {
        "promote": promote,
        "map50_delta": delta,
        "regressed_classes": regressed,
        "reasons": reasons or ["passes: mAP improved, no class regressed past tolerance"],
    }
