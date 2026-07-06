"""LabeloxAV label quality layer (M13). Per-annotation quality scoring, inter-annotator agreement, and
gold-set audits, so a reviewer sees which labels to trust and seeded bad annotations are caught. Reuses the
existing object.quality_score notion and the Review trail; pure over annotation dicts, so the scoring,
agreement, and audit are testable."""

from __future__ import annotations

from services.oraclyx.consensus import iou


def annotation_flags(obj: dict, img_w: float = 1920, img_h: float = 1080) -> list[str]:
    """Structural problems with one annotation's geometry."""
    flags = []
    box = obj.get("bbox")
    if box:
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 4 or h < 4:
            flags.append("tiny_box")
        if box[0] < -2 or box[1] < -2 or box[2] > img_w + 2 or box[3] > img_h + 2:
            flags.append("off_screen")
        if w <= 0 or h <= 0:
            flags.append("degenerate_box")
    if obj.get("class_id") is None:
        flags.append("no_class")
    return flags


def annotation_quality(obj: dict, agreement: float | None = None) -> dict:
    """Quality score in [0,1] from confidence, structural flags, and (when available) inter-annotator
    agreement. Human-source labels start from a high floor; flags subtract."""
    flags = annotation_flags(obj)
    base = 0.9 if obj.get("source") == "human" else float(obj.get("conf", 0.5))
    penalty = 0.25 * len(flags)
    agree_term = 1.0 if agreement is None else float(agreement)
    quality = max(0.0, min(1.0, (base - penalty) * (0.6 + 0.4 * agree_term)))
    return {"quality": round(quality, 4), "flags": flags, "agreement": agreement}


def inter_annotator_agreement(annotations: list[dict], iou_thr: float = 0.5) -> float:
    """Agreement across annotators for the same object: the fraction of annotator pairs that agree on class and
    overlap past iou_thr. 1.0 when everyone agrees, lower as they diverge."""
    n = len(annotations)
    if n < 2:
        return 1.0
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            a, b = annotations[i], annotations[j]
            if a.get("class_id") == b.get("class_id") and iou(a.get("bbox", []), b.get("bbox", [])) >= iou_thr:
                agree += 1
    return round(agree / total, 4) if total else 1.0


def gold_audit(predicted: list[dict], gold: list[dict], iou_thr: float = 0.5) -> dict:
    """Audit annotations against a gold set: greedily match each predicted box to a gold box of the same class;
    a matched prediction passes, an unmatched one (wrong class or misplaced) fails. Catches seeded bad
    annotations. Returns per-annotation verdicts and the summary."""
    used = set()
    verdicts = []
    for p in predicted:
        best_j, best_iou = None, 0.0
        for j, g in enumerate(gold):
            if j in used or g.get("class_id") != p.get("class_id"):
                continue
            ov = iou(p.get("bbox", []), g.get("bbox", []))
            if ov > best_iou:
                best_j, best_iou = j, ov
        passed = best_j is not None and best_iou >= iou_thr
        if passed:
            used.add(best_j)
        verdicts.append({"object_id": p.get("object_id"), "verdict": "pass" if passed else "fail",
                         "iou": round(best_iou, 3)})
    n_fail = sum(1 for v in verdicts if v["verdict"] == "fail")
    return {"n": len(predicted), "n_pass": len(predicted) - n_fail, "n_fail": n_fail,
            "missed_gold": len(gold) - len(used), "verdicts": verdicts}
