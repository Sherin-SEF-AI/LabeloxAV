"""Milestone G: 4D semantic segmentation consistency. Per-frame semantic labels already exist (2D
FrameSegmentation, 3D PointSegmentation, projected per-point seg), but each frame is labeled independently,
so a region's class can flicker frame to frame (a wall briefly read as a building, a momentary
misclassification on one frame). The 4D piece is temporal consistency: a sliding-window majority filter that
corrects an isolated flicker using its neighbours while a sustained change survives (a run longer than the
window keeps its label), plus a consistency metric that measures the flicker. Pure over a label sequence, so
it is tested without infra.
"""

from __future__ import annotations

from collections import Counter

from core.logging import get_logger

log = get_logger("seg4d")


def temporal_majority_filter(labels: list, window: int = 2) -> list:
    """Smooth a per-frame label sequence by majority vote over [i-window, i+window]. An isolated flicker (a
    run shorter than the window) is voted out; a sustained transition survives. Ties keep the original label,
    so the filter never flips a frame on an arbitrary tie."""
    n = len(labels)
    out = []
    for i in range(n):
        win = labels[max(0, i - window):min(n, i + window + 1)]
        counts = Counter(win)
        top, top_count = counts.most_common(1)[0]
        out.append(labels[i] if counts[labels[i]] == top_count else top)
    return out


def label_consistency(labels: list) -> dict:
    """The dominant label, the fraction of frames holding it (temporal consistency), and the number of
    frame-to-frame label changes (transitions, a flicker proxy)."""
    if not labels:
        return {"dominant": None, "consistency": 0.0, "transitions": 0}
    dominant, count = Counter(labels).most_common(1)[0]
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    return {"dominant": dominant, "consistency": round(count / len(labels), 4), "transitions": transitions}


async def track_class_consistency(track_id, window: int = 2) -> dict:
    """Measure 4D semantic consistency of a track's per-frame class and how many isolated flickers the
    temporal majority filter would correct. Proposes corrections, does not auto-apply them."""
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object.class_id).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Object.track_id == track_id).order_by(Frame.ts_ns))).scalars().all()
    onto = get_ontology()
    names = [onto.by_id(int(c)).name for c in rows]
    smoothed = temporal_majority_filter(names, window)
    corrections = sum(1 for a, b in zip(names, smoothed, strict=False) if a != b)
    cons = label_consistency(names)
    log.info("seg4d.track", track=str(track_id), frames=len(names), consistency=cons["consistency"],
             corrections=corrections)
    return {"track_id": str(track_id), "n_frames": len(names), **cons,
            "flicker_corrections": corrections,
            "smoothed_dominant": label_consistency(smoothed)["dominant"]}
