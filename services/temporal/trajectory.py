"""M-4D.2: trajectory smoothing. A tracked box jitters frame to frame (detector noise, propagation drift),
producing a ragged motion path and velocity discontinuities. smooth_path low-pass-filters the centroid path
(Savitzky-Golay where scipy is present, a centered moving average otherwise), preserving the endpoints so
the track is not pulled off its true start and end. smooth_track applies it to a 2D track, shifting each box
to its smoothed centre without changing the box size. The interactive drag-the-path editor is a frontend
follow-on; this is the algorithmic core it calls.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

log = get_logger("trajectory")


def smooth_path(points: list, window: int = 5) -> list[list[float]]:
    """Smooth a sequence of N-D points, reducing jitter while keeping the overall path and fixing the endpoints.
    Paths shorter than 3 points are returned unchanged."""
    a = np.asarray(points, dtype=float)
    if len(a) < 3:
        return [[round(float(v), 4) for v in p] for p in a]
    w = min(window if window % 2 else window + 1, len(a) if len(a) % 2 else len(a) - 1)
    w = max(3, w)
    try:
        from scipy.signal import savgol_filter
        sm = savgol_filter(a, w, min(2, w - 1), axis=0)
    except Exception:  # noqa: BLE001  scipy missing -> centered moving average
        sm = a.copy()
        half = w // 2
        for i in range(len(a)):
            sm[i] = a[max(0, i - half): min(len(a), i + half + 1)].mean(axis=0)
    sm[0], sm[-1] = a[0], a[-1]                 # anchor the endpoints to the true track start/end
    return [[round(float(v), 4) for v in p] for p in sm]


async def smooth_track(track_id, window: int = 5) -> dict:
    """Smooth a 2D track's motion path: shift every box to its smoothed centroid, keeping each box's size.
    Returns the number of boxes moved and the total pixel displacement."""
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        from db.models import Frame
        objs = (await db.execute(
            select(Object).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Object.track_id == track_id).order_by(Frame.ts_ns))).scalars().all()
        if len(objs) < 3:
            return {"track_id": str(track_id), "smoothed": 0, "reason": "track too short to smooth"}

        centers = [[(o.bbox[0] + o.bbox[2]) / 2.0, (o.bbox[1] + o.bbox[3]) / 2.0] for o in objs]
        sm = smooth_path(centers, window)
        moved = 0
        total_disp = 0.0
        for o, (cx0, cy0), (cx1, cy1) in zip(objs, centers, sm, strict=False):
            dx, dy = cx1 - cx0, cy1 - cy0
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue
            o.bbox = [o.bbox[0] + dx, o.bbox[1] + dy, o.bbox[2] + dx, o.bbox[3] + dy]
            o.version += 1
            moved += 1
            total_disp += (dx * dx + dy * dy) ** 0.5
        await db.commit()
    log.info("trajectory.smoothed", track=str(track_id), moved=moved)
    return {"track_id": str(track_id), "smoothed": moved, "total_displacement_px": round(total_disp, 1)}
