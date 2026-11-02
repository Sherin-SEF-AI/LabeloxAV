"""How well does propagation follow an object? Measured against tracks people have already accepted.

Propagation (the editor's "propagate" action) carries one box across the next frames so an annotator corrects
boxes instead of drawing them. There are two ways to do it here: optical flow alone (services/intelligence/
propagate.py), and flow with each box tightened by a SAM box prompt (services/temporal/sam_propagate.py).
Neither had been compared with what a person would have drawn.

The ground truth is a track whose boxes were accepted on consecutive frames. From its first box, each method
propagates over the following frames without writing anything, and each propagated box is compared with the
accepted box on that frame. Reported per step: mean IoU and the share of boxes a reviewer could keep as they
are (IoU >= 0.7) or with a nudge (IoU >= 0.5), plus how often tracking gave up.
"""

from __future__ import annotations

import random
from collections import defaultdict

import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from db.models import Frame, Object
from services.intelligence.propagate import _gray
from services.temporal.sam_propagate import _sam_refine, flow_track

KEEP_IOU, NUDGE_IOU = 0.7, 0.5


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


async def accepted_runs(db: AsyncSession, *, steps: int, n_tracks: int, seed: int = 7,
                        min_side_px: float = 24.0, track_ids: list | None = None) -> list[list[tuple]]:
    """Runs of `steps + 1` accepted boxes on consecutive frames of one track: [(frame, bbox), ...]."""
    q = select(Object.track_id).where(Object.track_id.isnot(None), Object.state == "accepted")
    if track_ids is not None:
        q = q.where(Object.track_id.in_(track_ids))
    long_tracks = (await db.execute(q.group_by(Object.track_id).having(func.count() > steps))).scalars().all()
    rng = random.Random(seed)
    rng.shuffle(long_tracks)
    runs: list[list[tuple]] = []
    for tid in long_tracks:
        if len(runs) >= n_tracks:
            break
        rows = (await db.execute(
            select(Frame, Object.bbox).join(Object, Object.frame_id == Frame.frame_id)
            .where(Object.track_id == tid, Object.state == "accepted").order_by(Frame.ts_ns))).all()
        if not rows:
            continue
        session_frames = (await db.execute(
            select(Frame.frame_id).where(Frame.session_id == rows[0][0].session_id,
                                         Frame.cam_id == rows[0][0].cam_id).order_by(Frame.ts_ns))).scalars().all()
        position = {fid: i for i, fid in enumerate(session_frames)}
        # The first stretch where the accepted boxes sit on adjacent frames, which is what propagation sees.
        for start in range(len(rows) - steps):
            chunk = rows[start:start + steps + 1]
            idx = [position.get(f.frame_id) for f, _ in chunk]
            if None in idx or any(b - a != 1 for a, b in zip(idx, idx[1:], strict=False)):
                continue
            b0 = chunk[0][1]
            if min(b0[2] - b0[0], b0[3] - b0[1]) < min_side_px:
                continue
            runs.append([(f, list(b)) for f, b in chunk])
            break
    return runs


async def evaluate_propagation(db: AsyncSession, *, steps: int = 12, n_tracks: int = 30,
                               refine: bool = True, seed: int = 7, track_ids: list | None = None) -> dict:
    """Flow alone against flow + SAM, step by step, on accepted tracks. Reads only."""
    store = get_object_store()
    runs = await accepted_runs(db, steps=steps, n_tracks=n_tracks, seed=seed, track_ids=track_ids)
    methods = ["flow", "flow_sam"] if refine else ["flow"]
    per_step: dict[str, dict[int, list[float]]] = {m: defaultdict(list) for m in methods}
    lost: dict[str, dict[int, int]] = {m: defaultdict(int) for m in methods}
    for run in runs:
        grays = [_gray(store, f.img_uri) for f, _ in run]
        flow_boxes = flow_track(grays, run[0][1])
        for step in range(1, steps + 1):
            truth = run[step][1]
            fb = flow_boxes[step - 1] if step - 1 < len(flow_boxes) else None
            if fb is None:
                for m in methods:
                    lost[m][step] += 1
                    per_step[m][step].append(0.0)
                continue
            per_step["flow"][step].append(_iou(fb, truth))
            if refine:
                img = cv2.imdecode(np.frombuffer(store.get_bytes(run[step][0].img_uri), np.uint8), cv2.IMREAD_COLOR)
                ref = _sam_refine(img, fb) if img is not None else None
                per_step["flow_sam"][step].append(_iou(ref[0] if ref else fb, truth))

    def summarise(m: str) -> list[dict]:
        out = []
        for step in sorted(per_step[m]):
            v = np.asarray(per_step[m][step])
            out.append({"step": step, "n": int(v.size), "mean_iou": round(float(v.mean()), 4),
                        "keep": round(float((v >= KEEP_IOU).mean()), 4),
                        "nudge": round(float((v >= NUDGE_IOU).mean()), 4), "lost": lost[m][step]})
        return out

    return {"tracks": len(runs), "steps": steps, "keep_iou": KEEP_IOU, "nudge_iou": NUDGE_IOU,
            "methods": {m: summarise(m) for m in methods}}


__all__ = ["accepted_runs", "evaluate_propagation"]
