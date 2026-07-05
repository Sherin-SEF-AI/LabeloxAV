"""ORACLYX offline-fusion consensus (M5). ORACLYX is a label-free labeling path: it fuses LiDAR, multi-view,
and heavy offline models into pseudo ground truth. The consensus gate is where its value concentrates. Where
ORACLYX fusion and the three LabeloxAV auto-label paths (YOLO26, SAM 3.1, Qwen3-VL) agree within tolerance,
the pseudo-label is accepted automatically; where they disagree, the sample is routed to the human queue,
which is exactly where human attention has the highest marginal value.

vote() is a pure function over per-path detections so the agreement rule is unit-testable; run.py persists the
verdict onto the pseudo_label side table and sets the object state (auto_accept on consensus, review on
disagreement)."""

from __future__ import annotations


def iou(a: list[float], b: list[float]) -> float:
    """Axis-aligned box IoU for [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def vote(fusion: dict, auto_paths: list[dict], iou_tol: float = 0.5, min_agree: int = 3) -> dict:
    """Decide consensus between the ORACLYX fusion detection and the auto-label paths for one object.

    ``fusion`` and each of ``auto_paths`` is {"path", "class_id", "bbox", "conf"}. A path agrees when it shares
    the fusion class and overlaps its box past iou_tol. Consensus holds when at least ``min_agree`` of the
    (fusion + paths) voters agree. Returns the consensus flag, a 0..1 score (agreeing fraction weighted by
    confidence), and the per-voter verdicts."""
    voters: dict[str, dict] = {}
    fclass, fbox = fusion.get("class_id"), fusion.get("bbox")
    agree_count = 1  # fusion agrees with itself
    conf_sum = float(fusion.get("conf", 1.0))
    conf_agree = conf_sum
    voters["fusion"] = {"agree": True, "conf": round(float(fusion.get("conf", 1.0)), 4)}
    for p in auto_paths:
        same_class = p.get("class_id") == fclass
        overlap = iou(fbox, p["bbox"]) if (fbox and p.get("bbox")) else 0.0
        agree = bool(same_class and overlap >= iou_tol)
        c = float(p.get("conf", 0.0))
        voters[p["path"]] = {"agree": agree, "conf": round(c, 4), "iou": round(overlap, 3)}
        conf_sum += c
        if agree:
            agree_count += 1
            conf_agree += c
    consensus = agree_count >= min_agree
    score = round(conf_agree / conf_sum, 4) if conf_sum > 0 else 0.0
    return {"consensus": consensus, "consensus_score": score, "agree_count": agree_count,
            "n_voters": 1 + len(auto_paths), "voters": voters}
