"""Score a real tracker against a sealed gold set.

The tracking metrics have existed and been tested for a while, and could never be run on anything. MOTA,
IDF1 and HOTA associate predicted identities with true ones, and the gold sealer kept boxes without keeping
identities, so there was ground truth for detection and none for association. The sealer now records
`track_ids` and a `tracks_sealed` flag; this is the consumer.

The refusal is the important part. A gold set where only some objects carry a track produces a number that
looks like a tracking score and is really a measurement of the gap in the labels: every object without an
identity reads as a fresh track in each frame, inflating the identity-switch count without the tracker doing
anything wrong. Scoring that would be worse than not scoring, so a partially-identified set is refused by
name and the caller is told how many objects are missing an identity.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.tracking_metrics import Detection, evaluate_tracking
from core.logging import get_logger
from db.models import GoldSet

log = get_logger("track_eval")


class TrackGroundTruthUnavailable(RuntimeError):
    """The gold set cannot support a tracking evaluation, and why."""


async def gold_track_detections(db: AsyncSession, gold_id: str) -> tuple[list[Detection], dict]:
    """Ground-truth tracks from a sealed set, ordered into frame indices.

    Frames are indexed by timestamp rather than by their uuid, because MOTA is defined over an ordered
    sequence: identity switches and fragmentations are counted between consecutive frames, and an arbitrary
    ordering would invent switches that never happened.
    """
    from db.models import Frame, Object

    gold = await db.get(GoldSet, gold_id)
    if gold is None:
        raise TrackGroundTruthUnavailable(f"gold set {gold_id!r} not found")
    if not gold.tracks_sealed:
        n_missing = sum(1 for t in (gold.track_ids or []) if not t)
        raise TrackGroundTruthUnavailable(
            f"gold set {gold_id!r} was not sealed with track identities "
            f"({n_missing} of {gold.n_objects} objects carry no track). A tracking metric associates "
            "predicted identities with true ones, and objects without one read as a new track in every "
            "frame, so the score would measure the gap in the labels rather than the tracker. Re-seal a "
            "set whose objects all carry a track.")

    ids = [uuid.UUID(o) for o in (gold.object_ids or [])]
    if not ids:
        raise TrackGroundTruthUnavailable(f"gold set {gold_id!r} is empty")

    rows = (await db.execute(
        select(Object, Frame.ts_ns, Frame.frame_id)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Object.object_id.in_(ids)))).all()

    order = sorted({int(ts) for _o, ts, _f in rows})
    index_of = {ts: i for i, ts in enumerate(order)}

    dets: list[Detection] = []
    without_track = 0
    for obj, ts, _fid in rows:
        if not obj.track_id:
            without_track += 1
            continue
        dets.append(Detection(frame=index_of[int(ts)], track_id=str(obj.track_id),
                              bbox=tuple(float(v) for v in obj.bbox)))
    if without_track:
        # Belt and braces: the flag said the set was complete, and the rows disagree. Trusting the flag here
        # would silently produce the exact number the flag exists to prevent.
        raise TrackGroundTruthUnavailable(
            f"{without_track} sealed objects have lost their track id since sealing; re-seal the set")

    return dets, {"gold_id": gold_id, "frames": len(order), "objects": len(dets),
                  "tracks": len({d.track_id for d in dets})}


async def predicted_track_detections(db: AsyncSession, run_id: str,
                                     gold_id: str) -> list[Detection]:
    """A tracker's own output over the same frames, from the immutable prediction plane.

    Restricted to the gold set's frames on purpose. Scoring a tracker over frames the gold set does not
    cover would count every correct prediction there as a false positive.
    """
    from db.models import Frame, Object, Prediction

    gold = await db.get(GoldSet, gold_id)
    if gold is None:
        raise TrackGroundTruthUnavailable(f"gold set {gold_id!r} not found")
    ids = [uuid.UUID(o) for o in (gold.object_ids or [])]
    frame_rows = (await db.execute(
        select(Object.frame_id).where(Object.object_id.in_(ids)))).scalars().all()
    frame_ids = sorted({f for f in frame_rows})
    if not frame_ids:
        return []

    ts_rows = (await db.execute(
        select(Frame.frame_id, Frame.ts_ns).where(Frame.frame_id.in_(frame_ids)))).all()
    index_of = {fid: i for i, (fid, _ts) in enumerate(sorted(ts_rows, key=lambda r: int(r[1])))}

    preds = (await db.execute(
        select(Prediction).where(Prediction.run_id == uuid.UUID(run_id),
                                 Prediction.frame_id.in_(frame_ids)))).scalars().all()
    out: list[Detection] = []
    for pred in preds:
        if not pred.track_id:
            continue
        out.append(Detection(frame=index_of[pred.frame_id], track_id=str(pred.track_id),
                             bbox=tuple(float(v) for v in pred.bbox)))
    return out


async def score_tracker(db: AsyncSession, *, gold_id: str, run_id: str,
                        iou_thr: float = 0.5) -> dict:
    """MOTA, IDF1 and HOTA for one inference run against one sealed gold set."""
    gt, meta = await gold_track_detections(db, gold_id)
    pred = await predicted_track_detections(db, run_id, gold_id)
    if not pred:
        # An honest zero would be indistinguishable from a tracker that produced nothing, which is a
        # different situation with a different fix.
        raise TrackGroundTruthUnavailable(
            f"inference run {run_id!r} carries no track identities over these frames; the run was a "
            "detector, not a tracker, so there is nothing to associate")

    metrics = evaluate_tracking(pred, gt, iou_thr=iou_thr)
    log.info("track_eval.scored", gold=gold_id, run=run_id,
             mota=metrics.get("mota"), idf1=metrics.get("idf1"))
    return {**meta, "run_id": run_id, "iou_thr": iou_thr,
            "predicted_detections": len(pred), "metrics": metrics}
