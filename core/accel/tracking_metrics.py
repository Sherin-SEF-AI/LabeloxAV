"""Multi-object tracking metrics: MOTA, IDF1, and the HOTA components.

`db/models.py::ModelRegistry.gold_metrics` documents itself as carrying "MOTA, IDF1" and nothing in the tree
ever produced either. `services/verdyx/track_metrics.py` has three useful per-track statistics (time to
detection, fragmentation, id switches) and zero callers. So the system tracks objects across frames, stores
`Track` and `Track3D`, and had no way to say whether the tracking was any good.

The three metrics are kept separate on purpose because they disagree by design:

- MOTA counts detection mistakes (misses, false positives) and identity switches in one number, so it is
  dominated by detection quality and can go negative.
- IDF1 measures how long the right identity is held, so a tracker that detects everything but swaps ids
  scores well on MOTA and badly here.
- HOTA balances detection and association explicitly, which is why it became the MOT benchmark default.

Implemented over an explicit per-frame association so the intermediate counts (matches, switches) stay
inspectable rather than collapsing into a single opaque score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.accel.boxes import box_iou_matrix


@dataclass(frozen=True)
class Detection:
    """One box in one frame, carrying the identity assigned to it."""

    frame: int
    track_id: str
    bbox: tuple[float, float, float, float]


def _greedy_associate(pred: list[Detection], gt: list[Detection], iou_thr: float) -> list[tuple[int, int]]:
    """Associate predictions to ground truth within a single frame by descending IoU."""
    if not pred or not gt:
        return []
    ious = box_iou_matrix([list(p.bbox) for p in pred], [list(g.bbox) for g in gt])
    pairs: list[tuple[int, int]] = []
    used_p: set[int] = set()
    used_g: set[int] = set()
    # Descending IoU over all candidate pairs, which is the standard MOT association when no cost solver is
    # used and is stable enough for evaluation (unlike a first-match scan, which depends on input order).
    order = np.dstack(np.unravel_index(np.argsort(ious, axis=None)[::-1], ious.shape))[0]
    for pi, gi in order:
        pi, gi = int(pi), int(gi)
        if ious[pi, gi] < iou_thr:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi))
    return pairs


def evaluate_tracking(pred: list[Detection], gt: list[Detection], iou_thr: float = 0.5) -> dict:
    """MOTA, IDF1, and HOTA components over a sequence.

    pred and gt are flat detection lists; frames are grouped internally so a caller can hand over a whole
    session without pre-bucketing it.
    """
    frames = sorted({d.frame for d in pred} | {d.frame for d in gt})
    if not frames:
        return {"measured": False, "reason": "no detections"}

    pred_by_frame: dict[int, list[Detection]] = {}
    gt_by_frame: dict[int, list[Detection]] = {}
    for d in pred:
        pred_by_frame.setdefault(d.frame, []).append(d)
    for d in gt:
        gt_by_frame.setdefault(d.frame, []).append(d)

    total_gt = len(gt)
    misses = fps = switches = matches = 0
    # gt track id -> the prediction track id it was last matched to, for switch counting
    last_match: dict[str, str] = {}
    # (gt_id, pred_id) -> how many frames that pairing held, for IDF1
    pair_counts: dict[tuple[str, str], int] = {}
    pred_counts: dict[str, int] = {}
    gt_counts: dict[str, int] = {}

    for f in frames:
        p = pred_by_frame.get(f, [])
        g = gt_by_frame.get(f, [])
        for d in p:
            pred_counts[d.track_id] = pred_counts.get(d.track_id, 0) + 1
        for d in g:
            gt_counts[d.track_id] = gt_counts.get(d.track_id, 0) + 1

        pairs = _greedy_associate(p, g, iou_thr)
        matches += len(pairs)
        misses += len(g) - len(pairs)
        fps += len(p) - len(pairs)

        for pi, gi in pairs:
            gid, pid = g[gi].track_id, p[pi].track_id
            pair_counts[(gid, pid)] = pair_counts.get((gid, pid), 0) + 1
            if gid in last_match and last_match[gid] != pid:
                switches += 1
            last_match[gid] = pid

    # MOTA can be negative: a tracker emitting more false positives than there are objects is worse than
    # emitting nothing, and clamping that to zero would hide it.
    mota = 1.0 - (misses + fps + switches) / total_gt if total_gt else None

    idf1 = _idf1(pair_counts, gt_counts, pred_counts)

    # HOTA components at this threshold: detection accuracy and association accuracy, and their geometric
    # mean. The full HOTA integrates over thresholds; this is the single-threshold form.
    det_a = matches / (matches + misses + fps) if (matches + misses + fps) else 0.0
    ass_a = _association_accuracy(pair_counts, gt_counts, pred_counts, matches)
    hota = float(np.sqrt(det_a * ass_a))

    return {
        "measured": True,
        "mota": round(mota, 4) if mota is not None else None,
        "idf1": round(idf1, 4),
        "hota": round(hota, 4),
        "det_a": round(det_a, 4),
        "ass_a": round(ass_a, 4),
        "matches": matches, "misses": misses, "false_positives": fps, "id_switches": switches,
        "gt_detections": total_gt, "frames": len(frames), "iou_thr": iou_thr,
    }


def _idf1(pair_counts: dict[tuple[str, str], int], gt_counts: dict[str, int],
          pred_counts: dict[str, int]) -> float:
    """Identity F1 over a one-to-one identity assignment.

    Each ground-truth track is assigned the predicted track it overlapped most, greedily and without reuse.
    Allowing reuse would let one predicted track claim credit for several ground-truth identities, which is
    precisely the failure IDF1 exists to punish.
    """
    if not pair_counts:
        return 0.0
    idtp = 0
    used_pred: set[str] = set()
    used_gt: set[str] = set()
    for (gid, pid), n in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        if gid in used_gt or pid in used_pred:
            continue
        used_gt.add(gid)
        used_pred.add(pid)
        idtp += n
    total_gt_dets = sum(gt_counts.values())
    total_pred_dets = sum(pred_counts.values())
    denom = total_gt_dets + total_pred_dets
    return (2 * idtp / denom) if denom else 0.0


def _association_accuracy(pair_counts: dict[tuple[str, str], int], gt_counts: dict[str, int],
                          pred_counts: dict[str, int], matches: int) -> float:
    """HOTA's AssA: the average, over matched detections, of how well their two tracks correspond."""
    if not matches:
        return 0.0
    total = 0.0
    for (gid, pid), tpa in pair_counts.items():
        fna = gt_counts.get(gid, 0) - tpa
        fpa = pred_counts.get(pid, 0) - tpa
        denom = tpa + fna + fpa
        if denom:
            total += tpa * (tpa / denom)
    return total / matches
