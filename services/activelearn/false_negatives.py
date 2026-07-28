"""Frame-level candidates: the frames the detector missed entirely.

Active learning has always drawn its candidates from `Object` rows, which means it can only ever propose
reviewing something the detector already found. A frame where the detector saw nothing produces no rows,
scores nothing, and is unreachable by selection. That is precisely the blind spot: the model cannot be
corrected on what it never proposed, so its recall failures are invisible to the loop that exists to fix
them.

Four signals, each attacking a different way a miss becomes invisible. They are combined rather than used
alone because any one of them alone has an obvious failure mode:

- **Sparsity against neighbours.** A frame whose visual neighbours are densely annotated and which has
  almost nothing itself is either genuinely empty or a miss, and the neighbours say which. On its own this
  flags every real empty road.
- **Low-confidence residue.** A frame whose only detections sit just under the accept threshold is a frame
  the model nearly saw something in. On its own this misses the frames where it saw nothing at all.
- **Temporal discontinuity.** A tracked object present before and after a frame and absent in it is a miss
  with a timestamp. This is the strongest single signal and it only fires where tracks exist.
- **Embedding novelty.** A frame far from everything already labelled is unlikely to be well served by the
  current model whatever it reported. This is the only signal that works on a frame with no detections and
  no neighbours, which is the hardest case.

The output is a review queue of frames, not of objects: the reviewer's job on these is to draw what is
missing, which is a different task from correcting a box and needs a different entry point.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("fn_mining")


@dataclass
class FrameCandidate:
    frame_id: str
    session_id: str
    score: float
    reasons: list[str]
    n_objects: int
    max_conf: float | None
    signals: dict


async def _frame_stats(db: AsyncSession, session_id: str | None, limit: int) -> list[dict]:
    """Per frame: how many objects it holds and the best confidence among them."""
    from db.models import Frame, Object

    stmt = (select(Frame.frame_id, Frame.session_id, Frame.ts_ns,
                   func.count(Object.object_id), func.max(Object.conf))
            .select_from(Frame)
            .outerjoin(Object, (Object.frame_id == Frame.frame_id) & (Object.state != "rejected"))
            .group_by(Frame.frame_id, Frame.session_id, Frame.ts_ns)
            .order_by(Frame.ts_ns)
            .limit(min(max(limit, 1), 20000)))
    if session_id:
        stmt = stmt.where(Frame.session_id == uuid.UUID(session_id))
    rows = (await db.execute(stmt)).all()
    return [{"frame_id": str(f), "session_id": str(s), "ts_ns": int(ts or 0),
             "n_objects": int(n or 0), "max_conf": (float(c) if c is not None else None)}
            for f, s, ts, n, c in rows]


def _sparsity_signal(frames: list[dict], window: int = 8) -> dict[str, float]:
    """How empty a frame is relative to its temporal neighbours in the same session.

    Temporal neighbours rather than embedding neighbours because consecutive dashcam frames genuinely depict
    the same scene: a frame with twelve objects either side and none of its own is not an empty road.
    """
    by_session: dict[str, list[dict]] = {}
    for f in frames:
        by_session.setdefault(f["session_id"], []).append(f)

    out: dict[str, float] = {}
    for items in by_session.values():
        items.sort(key=lambda f: f["ts_ns"])
        counts = [f["n_objects"] for f in items]
        for i, f in enumerate(items):
            lo, hi = max(0, i - window), min(len(items), i + window + 1)
            neighbours = counts[lo:i] + counts[i + 1:hi]
            if not neighbours:
                out[f["frame_id"]] = 0.0
                continue
            expected = float(np.median(neighbours))
            if expected < 1.0:
                # Nothing around it either: this is a genuinely quiet stretch, not evidence of a miss.
                out[f["frame_id"]] = 0.0
                continue
            deficit = max(0.0, expected - f["n_objects"]) / expected
            out[f["frame_id"]] = min(1.0, deficit)
    return out


def _residue_signal(frames: list[dict], accept_threshold: float = 0.5) -> dict[str, float]:
    """A frame whose best detection sits just below the accept line nearly produced something.

    Peaks just under the threshold and falls away on both sides: a frame at 0.49 is far more interesting
    than one at 0.05, which the model is confidently uninterested in, or one at 0.9, which it already got.
    """
    out: dict[str, float] = {}
    for f in frames:
        c = f["max_conf"]
        if c is None:
            # No detections at all. Handled by sparsity and novelty; scoring it here would double-count.
            out[f["frame_id"]] = 0.0
            continue
        if c >= accept_threshold:
            out[f["frame_id"]] = 0.0
            continue
        out[f["frame_id"]] = max(0.0, 1.0 - (accept_threshold - c) / max(accept_threshold, 1e-6))
    return out


async def _discontinuity_signal(db: AsyncSession, frames: list[dict]) -> dict[str, float]:
    """A tracked object present before and after a frame, and absent in it, is a miss with a timestamp.

    The strongest signal available, because it needs no threshold and no model: continuity of a physical
    object is the ground truth, and a gap in it is a detection failure rather than an object teleporting.
    """
    from db.models import Object

    out: dict[str, float] = {f["frame_id"]: 0.0 for f in frames}
    by_session: dict[str, list[dict]] = {}
    for f in frames:
        by_session.setdefault(f["session_id"], []).append(f)

    for items in by_session.values():
        items.sort(key=lambda f: f["ts_ns"])
        index_of = {f["frame_id"]: i for i, f in enumerate(items)}
        rows = (await db.execute(
            select(Object.track_id, Object.frame_id)
            .where(Object.track_id.isnot(None), Object.state != "rejected",
                   Object.frame_id.in_([uuid.UUID(f["frame_id"]) for f in items])))).all()

        per_track: dict[str, list[int]] = {}
        for track_id, frame_id in rows:
            i = index_of.get(str(frame_id))
            if i is not None:
                per_track.setdefault(str(track_id), []).append(i)

        for indices in per_track.values():
            if len(indices) < 2:
                continue
            indices.sort()
            for a, b in zip(indices, indices[1:], strict=False):
                gap = b - a - 1
                # Only short gaps. A track absent for fifty frames left the scene; a track absent for two
                # was missed, and treating the first as a miss would flood the queue with departures.
                if 0 < gap <= 5:
                    for j in range(a + 1, b):
                        fid = items[j]["frame_id"]
                        out[fid] = min(1.0, out.get(fid, 0.0) + 1.0 / gap)
    return out


async def _novelty_signal(db: AsyncSession, frames: list[dict]) -> dict[str, float]:
    """Distance from the labelled distribution, using the frame embeddings the corpus already carries.

    The only one of the four that works on a frame with no detections, no neighbours and no tracks, which
    is the hardest case and the one a purely object-derived queue can never reach.
    """
    from db.models import FrameEmbedding

    out: dict[str, float] = {f["frame_id"]: 0.0 for f in frames}
    ids = [uuid.UUID(f["frame_id"]) for f in frames]
    if not ids:
        return out
    rows = (await db.execute(
        select(FrameEmbedding.frame_id, FrameEmbedding.dino_vec)
        .where(FrameEmbedding.frame_id.in_(ids)))).all()
    vecs = {str(fid): np.asarray(v, dtype=np.float32) for fid, v in rows if v is not None}
    if len(vecs) < 8:
        # Too few embeddings to define a distribution. Returning zeros is honest; a distance computed
        # against five points would be noise dressed as a signal.
        return out

    labelled = [vecs[f["frame_id"]] for f in frames
                if f["n_objects"] > 0 and f["frame_id"] in vecs]
    if len(labelled) < 4:
        return out
    matrix = np.stack(labelled)
    matrix /= (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)

    for fid, v in vecs.items():
        u = v / (np.linalg.norm(v) + 1e-9)
        # Distance to the nearest labelled frame: a frame close to something already annotated is well
        # covered whatever its own count says.
        nearest = float(np.max(matrix @ u))
        out[fid] = float(min(1.0, max(0.0, (1.0 - nearest) / 0.5)))
    return out


async def mine_false_negatives(db: AsyncSession, *, session_id: str | None = None,
                               limit: int = 4000, top_k: int = 200,
                               accept_threshold: float = 0.5,
                               weights: dict | None = None) -> dict:
    """Rank frames by how likely the detector missed something in them.

    Returns frames, not objects. The reviewer's task here is to draw what is absent, which is a different
    action from correcting a box and belongs in a different queue.
    """
    w = {"sparsity": 0.30, "residue": 0.20, "discontinuity": 0.35, "novelty": 0.15, **(weights or {})}

    frames = await _frame_stats(db, session_id, limit)
    if not frames:
        return {"candidates": [], "considered": 0, "detail": "no frames match"}

    sparsity = _sparsity_signal(frames)
    residue = _residue_signal(frames, accept_threshold)
    discontinuity = await _discontinuity_signal(db, frames)
    novelty = await _novelty_signal(db, frames)

    out: list[FrameCandidate] = []
    for f in frames:
        fid = f["frame_id"]
        signals = {"sparsity": sparsity.get(fid, 0.0), "residue": residue.get(fid, 0.0),
                   "discontinuity": discontinuity.get(fid, 0.0), "novelty": novelty.get(fid, 0.0)}
        score = sum(w[k] * v for k, v in signals.items())
        if score <= 0:
            continue
        reasons = [k for k, v in signals.items() if v > 0.25]
        out.append(FrameCandidate(
            frame_id=fid, session_id=f["session_id"], score=round(float(score), 4),
            reasons=reasons or ["weak signal"], n_objects=f["n_objects"],
            max_conf=f["max_conf"], signals={k: round(float(v), 4) for k, v in signals.items()}))

    out.sort(key=lambda c: c.score, reverse=True)
    kept = out[:max(1, top_k)]
    log.info("fn_mining.ranked", considered=len(frames), scored=len(out), kept=len(kept))
    return {
        "considered": len(frames),
        "scored": len(out),
        # Stated rather than implied by the length: a caller that asked for 200 and got 200 cannot otherwise
        # tell whether the queue was exhausted or truncated.
        "truncated": max(0, len(out) - len(kept)),
        "weights": w,
        "candidates": [c.__dict__ for c in kept],
    }
