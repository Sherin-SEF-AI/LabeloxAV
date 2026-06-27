"""Video label propagation: label once, carry the box across the clip. Two complementary modes that
need no extra model weights:

- propagate_forward: Lucas-Kanade optical flow tracks feature points inside the box from frame to frame;
  a similarity transform (translation + scale + rotation) is fit and applied to the box corners. Handles
  approaching/receding objects. Drifts on long runs, so it seeds a track of state="annotate" boxes a
  human confirms.
- interpolate_track: linear bbox interpolation between a track's confirmed keyframes, filling the gaps.
  Anchored at both ends so it never drifts - the CVAT keyframe-interpolation accelerator.

    python -m services.intelligence.propagate --object <uuid> --frames 12
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import click
import cv2
import numpy as np
from sqlalchemy import select

from core.logging import get_logger, setup_logging
from core.config import get_settings
from core.storage import get_object_store
from db.models import Frame, Object, Track
from db.session import get_sessionmaker

log = get_logger("propagate")


def _gray(store, uri: str) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _clip_box(b, w, h):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))))
    y1, y2 = sorted((max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))))
    return [x1, y1, x2, y2]


async def propagate_forward(object_id: UUID, n_frames: int = 12) -> dict:
    maker = get_sessionmaker()
    store = get_object_store()
    async with maker() as db:
        obj = await db.get(Object, object_id)
        if obj is None:
            return {"created": 0, "reason": "object not found"}
        frame = await db.get(Frame, obj.frame_id)
        nexts = (
            await db.execute(
                select(Frame)
                .where(Frame.session_id == frame.session_id, Frame.ts_ns > frame.ts_ns)
                .order_by(Frame.ts_ns.asc())
                .limit(n_frames)
            )
        ).scalars().all()
        if not nexts:
            return {"created": 0, "reason": "no subsequent frames"}

        # Ensure a track to thread the propagated boxes onto.
        track_id = obj.track_id
        if track_id is None:
            tr = Track(session_id=frame.session_id, class_id=obj.class_id,
                       first_ts_ns=frame.ts_ns, last_ts_ns=frame.ts_ns)
            db.add(tr)
            await db.flush()
            track_id = tr.track_id
            obj.track_id = track_id

        prev_gray = _gray(store, frame.img_uri)
        box = list(obj.bbox)
        created: list[str] = []
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        for nf in nexts:
            cur_gray = _gray(store, nf.img_uri)
            if prev_gray is None or cur_gray is None:
                break
            h, w = cur_gray.shape
            x1, y1, x2, y2 = (int(v) for v in box)
            if x2 - x1 < 4 or y2 - y1 < 4:
                break
            roi = prev_gray[y1:y2, x1:x2]
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.01, minDistance=4)
            if pts is None or len(pts) < 4:
                break
            pts = pts.reshape(-1, 2) + np.float32([x1, y1])
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **lk)
            st = st.reshape(-1).astype(bool)
            if st.sum() < 4:
                break
            p0, p1 = pts[st], nxt.reshape(-1, 2)[st]
            M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
            if M is None:
                break
            corners = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
            moved = cv2.transform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
            box = _clip_box([moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()], w, h)
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                break
            o = Object(frame_id=nf.frame_id, track_id=track_id, class_id=obj.class_id,
                       bbox=[float(v) for v in box], conf=0.5, source="propagated", state="annotate",
                       provenance={"propagated_from": str(object_id), "method": "optical-flow"})
            db.add(o)
            await db.flush()
            created.append(str(o.object_id))
            prev_gray = cur_gray

        if created:
            tr = await db.get(Track, track_id)
            tr.last_ts_ns = nexts[len(created) - 1].ts_ns
        await db.commit()
    log.info("propagate.done", created=len(created), track_id=str(track_id))
    return {"created": len(created), "track_id": str(track_id), "object_ids": created}


async def interpolate_track(track_id: UUID) -> dict:
    """Fill gaps between a track's keyframes with linearly-interpolated boxes (never drifts)."""
    maker = get_sessionmaker()
    async with maker() as db:
        tr = await db.get(Track, track_id)
        if tr is None:
            return {"created": 0, "reason": "track not found"}
        items = (
            await db.execute(
                select(Object, Frame.ts_ns, Frame.frame_id)
                .join(Frame, Frame.frame_id == Object.frame_id)
                .where(Object.track_id == track_id)
                .order_by(Frame.ts_ns.asc())
            )
        ).all()
        keys = [(ts, o.bbox) for o, ts, _ in items]
        have_ts = {ts for ts, _ in keys}
        # candidate frames between first and last key that have no box on this track
        gap_frames = (
            await db.execute(
                select(Frame)
                .where(Frame.session_id == tr.session_id,
                       Frame.ts_ns > keys[0][0], Frame.ts_ns < keys[-1][0])
                .order_by(Frame.ts_ns.asc())
            )
        ).scalars().all() if len(keys) >= 2 else []

        created = 0
        for f in gap_frames:
            if f.ts_ns in have_ts:
                continue
            # bracketing keyframes
            lo = max((k for k in keys if k[0] <= f.ts_ns), key=lambda k: k[0])
            hi = min((k for k in keys if k[0] >= f.ts_ns), key=lambda k: k[0])
            if hi[0] == lo[0]:
                continue
            a = (f.ts_ns - lo[0]) / (hi[0] - lo[0])
            box = [lo[1][i] + a * (hi[1][i] - lo[1][i]) for i in range(4)]
            db.add(Object(frame_id=f.frame_id, track_id=track_id, class_id=tr.class_id,
                          bbox=[float(v) for v in box], conf=0.5, source="interp", state="annotate",
                          provenance={"method": "interpolate"}))
            created += 1
        await db.commit()
    log.info("interpolate.done", created=created, track_id=str(track_id))
    return {"created": created, "track_id": str(track_id)}


@click.command()
@click.option("--object", "object_id", default=None)
@click.option("--track", "track_id", default=None)
@click.option("--frames", type=int, default=12)
def main(object_id, track_id, frames) -> None:
    setup_logging(get_settings().log_level)
    if track_id:
        click.echo(asyncio.run(interpolate_track(UUID(track_id))))
    elif object_id:
        click.echo(asyncio.run(propagate_forward(UUID(object_id), frames)))
    else:
        raise SystemExit("pass --object <uuid> [--frames N] or --track <uuid>")


if __name__ == "__main__":
    main()
