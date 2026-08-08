"""ORACLYX 4D pseudo-GT tracks (M14). Lift per-frame consensus boxes into track-level pseudo-ground-truth with
one stable identity across time, interpolation over short observation gaps, and healing across occlusion. Where
LABELOX M13 propagate4d serves a human keyframe, this serves the auto-truth path: it stitches ORACLYX
consensus observations from many frames into a single 4D track the distillation set can trust. Pure over
observation dicts, so identity stability and gap healing are testable without a corpus."""

from __future__ import annotations

from uuid import UUID, uuid4


def _interp(a: list[float], b: list[float], t: float) -> list[float]:
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def stitch_track(observations: dict[int, list[float]], n_frames: int, max_heal_gap: int = 8,
                 track_id: UUID | None = None) -> dict:
    """Stitch sparse per-frame consensus boxes into one 4D track.

    observations maps a frame index to a consensus box. Frames between two observations no more than
    ``max_heal_gap`` apart are healed by interpolation (occlusion healing); wider gaps are left as holes rather
    than fabricated. Returns one track_id, a box per covered frame with a source in {observed, interpolated},
    and the count of frames healed. Every box carries the same identity, the invariant the distillation path
    relies on."""
    tid = track_id or uuid4()
    if not observations:
        return {"track_id": str(tid), "n_frames": n_frames, "boxes": [], "identities": 0, "healed_gaps": 0,
                "coverage": 0.0}

    anchors = sorted(observations)
    boxes: dict[int, dict] = {i: {"frame": i, "bbox": list(observations[i]), "source": "observed"}
                              for i in anchors}
    healed = 0
    for lo, hi in zip(anchors, anchors[1:], strict=False):
        span = hi - lo
        if span <= 1 or span > max_heal_gap:
            continue
        for i in range(lo + 1, hi):
            t = (i - lo) / span
            boxes[i] = {"frame": i, "bbox": _interp(observations[lo], observations[hi], t),
                        "source": "interpolated"}
            healed += 1

    ordered = [boxes[i] for i in sorted(boxes)]
    covered = len(ordered)
    return {"track_id": str(tid), "n_frames": n_frames, "boxes": ordered,
            "identities": 1, "healed_gaps": healed,
            "coverage": round(covered / n_frames, 4) if n_frames else 0.0}
