"""Keyframe interpolation along a track (M2.5). Between human keyframes, fill the frames with linear or
cubic boxes (and SAM 3.1 mask propagation on the pod), marking them source=interpolated + interp_source so
provenance shows they are machine-filled. Builds on the box-interpolation geometry from propagate.py and
the BoT-SORT tracks from M2.0.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import delete, or_, select

from core.logging import get_logger
from db.models import Frame, Object, Track
from db.session import get_sessionmaker

log = get_logger("interpolate")


async def _keyframes(db, track_id: UUID):
    """Anchor objects on a track: human-verified or explicitly-marked keyframes, ordered by time."""
    rows = (await db.execute(
        select(Object, Frame.ts_ns).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.track_id == track_id, or_(Object.is_keyframe.is_(True), Object.source == "human"))
        .order_by(Frame.ts_ns))).all()
    return rows


def _to_cxcywh(box: np.ndarray) -> np.ndarray:
    """Corner boxes (N,4 as x1,y1,x2,y2) -> center+size (N,4 as cx,cy,w,h). We interpolate motion and scale
    separately: a car approaching the camera grows in a way that oscillates badly if you spline the corners
    independently, but is smooth and monotone in width/height."""
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=1)


def build_box_interpolator(kf_ts: list[int], kf_box: np.ndarray, method: str):
    """Return (box_at, src). `box_at(ts)` gives an [x1,y1,x2,y2] box for any ts inside the keyframe span.

    method='cubic' uses a shape-preserving monotone spline (PCHIP) on center and size. Unlike an ordinary
    cubic it does not overshoot between anchors (no Runge wobble, no box that briefly balloons or inverts),
    while still curving through the acceleration a straight line would miss. Falls back to linear with <3
    keyframes or if SciPy is unavailable.
    """
    ts = np.asarray(kf_ts, dtype=float)
    cc = _to_cxcywh(np.asarray(kf_box, dtype=float))
    # collapse duplicate timestamps (two keyframes on the same frame) so the spline sees a strictly increasing grid
    uniq_ts, idx = np.unique(ts, return_index=True)
    ts, cc = uniq_ts, cc[idx]

    fns = None
    src = "linear"
    if method in ("cubic", "pchip", "spline") and len(ts) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator

            fns = [PchipInterpolator(ts, cc[:, i], extrapolate=True) for i in range(4)]
            src = "cubic"
        except Exception:  # noqa: BLE001 - SciPy missing/edge case: degrade to linear rather than fail the fill
            fns = None

    def box_at(t: float) -> list[float]:
        if fns is not None:
            cx, cy, w, h = (float(fns[i](t)) for i in range(4))
        else:
            cx, cy, w, h = (float(np.interp(t, ts, cc[:, i])) for i in range(4))
        w, h = max(1.0, w), max(1.0, h)   # a spline must never emit a zero-area or inverted box
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    return box_at, src


def _interp_conf(ts: float, kf_ts: list[int]) -> float:
    """Confidence that decays with temporal distance from the nearest keyframe: an anchor-adjacent fill is
    trustworthy, a fill in the middle of a long gap is not, so it earns a lower conf and lands in review."""
    left = max((k for k in kf_ts if k <= ts), default=kf_ts[0])
    right = min((k for k in kf_ts if k >= ts), default=kf_ts[-1])
    span = right - left
    if span <= 0:
        return 0.55
    t = (ts - left) / span                       # 0 at left anchor, 1 at right anchor
    closeness = 1.0 - abs(2 * t - 1)             # 0 at an anchor, 1 at the gap midpoint
    return round(0.55 - 0.25 * closeness, 3)     # 0.55 next to an anchor, 0.30 mid-gap


async def interpolate_track_keyframed(track_id: UUID, method: str = "linear", lo_ts: int | None = None, hi_ts: int | None = None) -> dict:
    """Fill frames between keyframes with interpolated boxes. If lo_ts/hi_ts are given, only that segment
    is (re)interpolated (edit-propagation); otherwise the whole track between first and last keyframe."""
    maker = get_sessionmaker()
    async with maker() as db:
        tr = await db.get(Track, track_id)
        if tr is None:
            return {"created": 0, "reason": "track not found"}
        anchors = await _keyframes(db, track_id)
        if len(anchors) < 2:
            return {"created": 0, "reason": "need at least 2 keyframes (mark human-verified frames)"}

        kf_ts = [ts for _, ts in anchors]
        kf_box = np.asarray([list(o.bbox) for o, _ in anchors], dtype=float)
        class_id = anchors[0][0].class_id
        a, b = (lo_ts if lo_ts is not None else kf_ts[0]), (hi_ts if hi_ts is not None else kf_ts[-1])

        frames = (await db.execute(
            select(Frame.frame_id, Frame.ts_ns)
            .where(Frame.session_id == tr.session_id, Frame.ts_ns > a, Frame.ts_ns < b)
            .order_by(Frame.ts_ns))).all()
        # clear existing machine-filled boxes on this track in the segment (idempotent re-interpolation)
        seg_fids = [fid for fid, _ in frames]
        if seg_fids:
            await db.execute(delete(Object).where(
                Object.track_id == track_id, Object.source == "interpolated", Object.frame_id.in_(seg_fids)))

        box_at, src = build_box_interpolator(kf_ts, kf_box, method)

        kf_set = set(kf_ts)
        created = 0
        for fid, ts in frames:
            if ts in kf_set:
                continue
            conf = _interp_conf(float(ts), kf_ts)
            db.add(Object(frame_id=fid, track_id=track_id, class_id=class_id, bbox=box_at(float(ts)),
                          conf=conf, source="interpolated", state="annotate", interp_source=src,
                          provenance={"method": "interpolate", "interp_source": src, "conf_by_gap": conf}))
            created += 1
        await db.commit()

    out = {"track_id": str(track_id), "created": created, "method": src, "keyframes": len(kf_ts)}
    log.info("interpolate.done", **out)
    return out
