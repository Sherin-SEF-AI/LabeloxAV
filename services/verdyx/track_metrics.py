"""VERDYX track-level safety metrics (M15). A per-frame mAP hides the failures that matter for driving: an
object that is detected but late, a track that shatters into fragments, an identity that swaps between two
objects. This scores a matched prediction track against a ground-truth track on the metrics a safety case
actually needs: time-to-detection, track fragmentation, and identity switches. Pure over per-frame association
records, so each metric is testable without a corpus."""

from __future__ import annotations

from services.oraclyx.consensus import iou


def match_frame(pred_boxes: list[dict], gt_box: list[float], iou_thr: float = 0.5) -> dict | None:
    """Best prediction for a GT box in one frame, or None if nothing overlaps past iou_thr."""
    best, best_iou = None, iou_thr
    for p in pred_boxes:
        ov = iou(p.get("bbox", []), gt_box)
        if ov >= best_iou:
            best, best_iou = p, ov
    return best


def time_to_detection(gt_track: list[dict], pred_by_frame: dict[int, list[dict]], fps: float = 10.0,
                      iou_thr: float = 0.5) -> dict:
    """Frames (and seconds) from a GT object first appearing to its first correct detection. A large
    time-to-detection is a latent safety failure a frame-averaged recall would smear away. Returns
    detected=False when the object is never detected."""
    frames = sorted(gt_track, key=lambda g: g["frame"])
    if not frames:
        return {"detected": False, "frames_to_detect": None, "seconds_to_detect": None}
    first = frames[0]["frame"]
    for g in frames:
        m = match_frame(pred_by_frame.get(g["frame"], []), g["bbox"], iou_thr)
        if m is not None:
            n = g["frame"] - first
            return {"detected": True, "frames_to_detect": n,
                    "seconds_to_detect": round(n / fps, 3) if fps else None}
    return {"detected": False, "frames_to_detect": None, "seconds_to_detect": None}


def track_fragmentation(gt_track: list[dict], pred_by_frame: dict[int, list[dict]],
                        iou_thr: float = 0.5) -> dict:
    """How badly a single GT track is split. Fragments = the number of contiguous detected runs; a clean track
    is 1, a flickering one is many. Also returns tracking coverage (fraction of GT frames matched)."""
    frames = sorted(gt_track, key=lambda g: g["frame"])
    matched = [match_frame(pred_by_frame.get(g["frame"], []), g["bbox"], iou_thr) is not None for g in frames]
    fragments = 0
    prev = False
    for m in matched:
        if m and not prev:
            fragments += 1
        prev = m
    covered = sum(matched)
    return {"fragments": fragments, "coverage": round(covered / len(frames), 4) if frames else 0.0,
            "gt_frames": len(frames), "matched_frames": covered}


def id_switches(gt_track: list[dict], pred_by_frame: dict[int, list[dict]], iou_thr: float = 0.5) -> int:
    """Identity switches along one GT track: the number of times the matched prediction's track_id changes
    between consecutive detected frames. Each switch is a place a downstream planner would lose the object."""
    frames = sorted(gt_track, key=lambda g: g["frame"])
    last_id = None
    switches = 0
    for g in frames:
        m = match_frame(pred_by_frame.get(g["frame"], []), g["bbox"], iou_thr)
        if m is None:
            continue
        pid = m.get("track_id")
        if last_id is not None and pid is not None and pid != last_id:
            switches += 1
        if pid is not None:
            last_id = pid
    return switches
