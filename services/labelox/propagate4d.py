"""LabeloxAV 4D label propagation (M13). Propagate an annotation across a clip with a single, stable track
identity, pre-filled from ORACLYX consensus where available. A constant-velocity box model carries the box
forward and backward from the keyframe; gaps (an occluded or missing frame) are healed by interpolating
between the surrounding known boxes. Every propagated box carries the same track_id, so a correction on the
keyframe propagates with consistent identity. Reuses the interpolation idea from services/temporal; pure over
boxes, so identity stability and gap healing are testable."""

from __future__ import annotations

from uuid import UUID, uuid4


def _interp_box(a: list[float], b: list[float], t: float) -> list[float]:
    return [a[i] + (b[i] - a[i]) * t for i in range(4)]


def propagate_track(keyframe_box: list[float], n_frames: int, velocity: list[float] | None = None,
                    known: dict[int, list[float]] | None = None, track_id: UUID | None = None) -> dict:
    """Propagate a box across n_frames from a keyframe at index 0.

    velocity is the per-frame box delta (constant-velocity model); known maps a frame index to an observed box
    (e.g. an ORACLYX consensus box) which anchors the propagation and heals gaps by interpolation. Returns one
    track_id and a box per frame, with a per-box source in {keyframe, observed, propagated, interpolated}."""
    tid = track_id or uuid4()
    known = dict(known or {})
    known.setdefault(0, list(keyframe_box))
    v = velocity or [0.0, 0.0, 0.0, 0.0]

    boxes: dict[int, dict] = {}
    for i in known:
        boxes[i] = {"frame": i, "bbox": list(known[i]),
                    "source": "keyframe" if i == 0 else "observed"}

    anchors = sorted(known)
    for i in range(n_frames):
        if i in boxes:
            continue
        lo = max([a for a in anchors if a < i], default=None)
        hi = min([a for a in anchors if a > i], default=None)
        if lo is not None and hi is not None:
            # heal the gap by interpolating between the two surrounding anchors
            t = (i - lo) / (hi - lo)
            boxes[i] = {"frame": i, "bbox": _interp_box(known[lo], known[hi], t), "source": "interpolated"}
        else:
            # extrapolate with the constant-velocity model from the nearest anchor
            base = lo if lo is not None else hi
            step = i - base
            boxes[i] = {"frame": i, "bbox": [known[base][k] + v[k] * step for k in range(4)],
                        "source": "propagated"}

    ordered = [boxes[i] for i in range(n_frames)]
    return {"track_id": str(tid), "n_frames": n_frames,
            "boxes": ordered,
            "identities": len({tid}),   # always one identity: the invariant M13 guarantees
            "healed_gaps": sum(1 for b in ordered if b["source"] == "interpolated")}
