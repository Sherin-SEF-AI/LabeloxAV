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


# ---- #8 paths and intent: the kinematic baseline a future-path label starts from --------------------------------
#
# There is no learned motion forecaster in this repo, so the future-path editor is seeded by the simplest model
# that is still honest about turning: constant turn rate and velocity (CTRV). It is labelled a baseline wherever it
# is shown. Its job is to give the annotator a plausible curve to drag, not to predict; `baseline_error` measures
# how far it is from what the road user actually did wherever the track continues past the anchor.


def estimate_ctrv_state(times_s: list[float], xy: list[list[float]], window: int = 5) -> dict | None:
    """Speed, heading and yaw rate at the last sample of a (t, x, y) history, after light smoothing.

    Heading comes from the displacement over the last `window` samples rather than the last step, because one
    step of a monocular track is mostly detector jitter. Yaw rate is the change between the heading of the older
    half of that window and the newer half. None when fewer than two distinct samples exist.
    """
    t = np.asarray(times_s, dtype=float)
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(t) < 2 or len(p) != len(t):
        return None
    if len(p) >= 3:
        p = np.asarray(smooth_path(p.tolist(), window), dtype=float)
    k = min(len(p) - 1, max(1, window))
    dt = t[-1] - t[-1 - k]
    if dt <= 1e-6:
        return None
    d = p[-1] - p[-1 - k]
    speed = float(np.hypot(*d) / dt)
    heading = float(np.arctan2(d[1], d[0]))
    yaw_rate = 0.0
    if k >= 2 and speed > 0.3:
        mid = len(p) - 1 - k // 2
        d1, d2 = p[mid] - p[-1 - k], p[-1] - p[mid]
        t1, t2 = t[mid] - t[-1 - k], t[-1] - t[mid]
        if np.hypot(*d1) > 1e-3 and np.hypot(*d2) > 1e-3 and t1 > 0 and t2 > 0:
            h1, h2 = np.arctan2(d1[1], d1[0]), np.arctan2(d2[1], d2[0])
            dh = (h2 - h1 + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = float(dh / ((t1 + t2) / 2.0))
    return {"x": float(p[-1, 0]), "y": float(p[-1, 1]), "speed_mps": speed, "heading_rad": heading,
            "yaw_rate_rps": yaw_rate}


def ctrv_forecast(state: dict, horizon_s: float = 3.0, dt_s: float = 0.25,
                  max_yaw_rate: float = 0.8) -> list[list[float]]:
    """Roll a CTRV state forward: [[t, x, y], ...] from t=0 (the anchor) to `horizon_s`.

    The yaw rate is clipped to what a road vehicle can sustain, because a noisy estimate of 3 rad/s turns a
    straight track into a circle and the annotator would have to undo the whole curve.
    """
    x, y = float(state["x"]), float(state["y"])
    v, th = float(state["speed_mps"]), float(state["heading_rad"])
    w = float(np.clip(state.get("yaw_rate_rps", 0.0), -max_yaw_rate, max_yaw_rate))
    out = [[0.0, round(x, 4), round(y, 4)]]
    n = max(1, int(round(horizon_s / dt_s)))
    for i in range(1, n + 1):
        t = i * dt_s
        if abs(w) < 1e-4:
            px, py = x + v * t * np.cos(th), y + v * t * np.sin(th)
        else:
            px = x + v / w * (np.sin(th + w * t) - np.sin(th))
            py = y + v / w * (np.cos(th) - np.cos(th + w * t))
        out.append([round(t, 4), round(float(px), 4), round(float(py), 4)])
    return out


def baseline_error(forecast: list[list[float]], observed: list[list[float]]) -> dict | None:
    """Average and final displacement (ADE / FDE, metres) of a forecast against what was actually observed.

    Both are [[t, x, y], ...] on the same time origin. The forecast is interpolated at each observed time inside
    its horizon; observations past the horizon are ignored. None when nothing overlaps.
    """
    f = np.asarray(forecast, dtype=float).reshape(-1, 3)
    o = np.asarray(observed, dtype=float).reshape(-1, 3)
    o = o[(o[:, 0] > 1e-6) & (o[:, 0] <= f[-1, 0] + 1e-6)]
    if len(o) == 0 or len(f) < 2:
        return None
    fx = np.interp(o[:, 0], f[:, 0], f[:, 1])
    fy = np.interp(o[:, 0], f[:, 0], f[:, 2])
    d = np.hypot(fx - o[:, 1], fy - o[:, 2])
    return {"ade_m": round(float(d.mean()), 3), "fde_m": round(float(d[-1]), 3), "n": int(len(d)),
            "horizon_s": round(float(o[-1, 0]), 3)}
