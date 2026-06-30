"""Milestone I: inter-annotator agreement. When two annotators label the same frame independently, this
scores how much they agree: boxes are matched by IoU (greedy, highest overlap first), detection agreement is
the matched fraction of the union, and on the matched pairs we report class agreement, mean IoU, and Cohen's
kappa (agreement corrected for chance). Pure math over two label sets, so it scores without infra; the
double-pass collection that produces two independent label sets is the data seam, not faked here.
"""

from __future__ import annotations

from collections import Counter

from core.logging import get_logger

log = get_logger("iaa")


def iou(a: list, b: list) -> float:
    """IoU of two boxes [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(boxes_a: list, boxes_b: list, iou_thresh: float = 0.5) -> list[tuple]:
    """Greedy one-to-one matching by descending IoU above the threshold. Returns [(i, j, iou)]."""
    cand = sorted((iou(a, b), i, j) for i, a in enumerate(boxes_a) for j, b in enumerate(boxes_b)
                  if iou(a, b) >= iou_thresh)
    cand.reverse()
    used_a: set = set()
    used_b: set = set()
    matches = []
    for v, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, v))
    return matches


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa for two label sequences. Perfect agreement on a single shared category returns 1.0 (the
    chance-corrected form is 0/0 there, resolved to perfect by convention)."""
    n = len(labels_a)
    if n == 0:
        return 1.0
    po = sum(1 for x, y in zip(labels_a, labels_b, strict=False) if x == y) / n
    cats = set(labels_a) | set(labels_b)
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def iaa_score(set_a: list, set_b: list, iou_thresh: float = 0.5) -> dict:
    """Score agreement between two label sets [{bbox, class_name}]. detection_agreement is matched / union;
    class_agreement, mean_iou, and cohen_kappa are over the matched pairs."""
    matches = match_boxes([s["bbox"] for s in set_a], [s["bbox"] for s in set_b], iou_thresh)
    n_match, n_a, n_b = len(matches), len(set_a), len(set_b)
    union = n_a + n_b - n_match
    la = [set_a[i]["class_name"] for i, _, _ in matches]
    lb = [set_b[j]["class_name"] for _, j, _ in matches]
    return {
        "detection_agreement": round(n_match / union, 4) if union else 1.0,
        "class_agreement": round(sum(1 for x, y in zip(la, lb, strict=False) if x == y) / n_match, 4) if n_match else 0.0,
        "mean_iou": round(sum(v for _, _, v in matches) / n_match, 4) if n_match else 0.0,
        "cohen_kappa": round(cohen_kappa(la, lb), 4) if n_match else 0.0,
        "n_matched": n_match, "n_a": n_a, "n_b": n_b,
    }
