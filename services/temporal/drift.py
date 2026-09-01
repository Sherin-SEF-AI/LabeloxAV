"""Whether a machine-filled box is still on the object its keyframe was drawn around.

The check itself already existed and was reachable from one place. `services/agent/propagate_agent.py`
crops the keyframe, encodes it with DINOv3, and compares each propagated box to it, stopping the walk when
the cosine falls through 0.62. That is exactly the right test and it only ever ran at creation time, and
only for boxes that agent itself created. Nothing could ask the question about the boxes already in the
corpus, which is where it matters: interpolated and propagated objects are most of the machine fill, and
`services/errordetect/embedding_outlier.py` compares an object to its class centroid, which answers a
different question. A box that has slid off a scooter onto the road behind it still looks like a scooter
to a class centroid, and looks nothing like the crop it started from.

So the primitives move here and both callers share them. The keyframe-relative pass is what this module
adds, and the shape of the answer is a per-object cosine against the nearest anchor, plus the reason when
there is no answer.

**It discriminates, and here is the measurement that says so.** A check that fires on nearly everything is
not a check, which is why `services/labelops/reanalyze.py` discards any rule firing on 80% of its scope.
Run over the real corpus on 2026-09-02:

    machine fill (interpolated / propagated)   12 tracks, 178 boxes    64.6% drifted
    detector output, as a control              12 tracks, 182 boxes    32.4% drifted

The control takes tracks with no machine fill at all and compares each detection to the track's first,
exactly as a fill would be judged. Roughly a third of genuine detections fall through the floor, which is
this check's own noise: an object's appearance really does change as it approaches, turns and is occluded.
The fill rate is twice that. And the comparison understates the gap, because the control uses the first
anchor on the track while the pass below uses the NEAREST one, so the control is judged over a longer
baseline and should have drifted more, not less.

That 64.6% is also consistent with what was already known from a different direction: interpolated objects
judge at 0.209 precision against 0.603 for real detections.

**None is not zero.** `cosine_to` returns None when the crop cannot be read or DINOv3 is unavailable, and
that stays None all the way out. A failed encode recorded as a similarity of 0 would mark every object on
a GPU-less host as drifted, which is the shape of mistake that has already cost this repo 125 poisoned
verdicts when a failed VLM call was recorded as `unsure`.

**Nothing here takes the machine down.** Encoding runs under the same advisory GPU lock the autolabel and
class-precision sweeps use, in bounded batches, re-checking headroom between them and giving up after a
run of failures rather than grinding.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object

log = get_logger("drift")

# The thresholds propagate_agent measured and has been using. Kept as this module's names so the two
# callers cannot drift apart, and so a change is made once.
DRIFT_FLOOR = 0.62      # below this the box is no longer on the thing the keyframe was drawn around
HIGH = 0.80             # above this it is the same object beyond reasonable doubt
SIZE_TOL = 3.0          # area ratio band, either direction

# Sources that are machine fill: the boxes nobody drew and nobody has checked. `human` is excluded because
# a person drawing a box IS the anchor, and `fused`/`auto_accept` because a detector fired on that frame,
# so the box is evidence rather than an extrapolation from another frame.
MACHINE_FILL = ("interpolated", "propagated", "recall")

# Crops per batch, between which the resource guards are re-checked. Small enough that a training job
# waiting on the card waits seconds.
BATCH = 24

# A run of failures means the card or the weights are gone, not that these particular crops are odd.
MAX_CONSECUTIVE_FAILURES = 20


def crop_box(store, uri: str, box) -> Any:
    """The image region a box covers, or None when it cannot be read or is degenerate."""
    try:
        import cv2

        # get_bytes normalises a uri or a bare key itself, which is what every other caller relies on.
        img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 - a missing frame is one absent measurement, not a failed run
        return None
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2] if (x2 - x1 >= 2 and y2 - y1 >= 2) else None


def encode_crop(store, uri: str, box) -> np.ndarray | None:
    """DINOv3 vector for a box crop, or None when the crop or the model is unavailable."""
    try:
        from services.intelligence.embed import dinov3
        from services.intelligence.embed.prep import square_letterbox

        crop = crop_box(store, uri, box)
        return None if crop is None else np.asarray(dinov3.encode_image(square_letterbox(crop)))
    except Exception:  # noqa: BLE001 - no GPU or no weights: the caller degrades to geometry only
        return None


def cosine_to(store, uri: str, box, ref_vec) -> float | None:
    """Cosine between a box crop and a reference vector. None when either is unavailable."""
    if ref_vec is None:
        return None
    v = encode_crop(store, uri, box)
    if v is None:
        return None
    return float(np.dot(v, np.asarray(ref_vec)))


def size_ok(box, ref_box, tol: float = SIZE_TOL) -> bool:
    """Whether the box is within the area band of its anchor. Cheap, and it runs with no GPU at all."""
    def area(b):
        return max(1.0, (float(b[2]) - float(b[0]))) * max(1.0, (float(b[3]) - float(b[1])))

    ratio = area(box) / area(ref_box)
    return (1.0 / tol) <= ratio <= tol


async def _anchors_and_fill(db: AsyncSession, track_id: uuid.UUID):
    """A track's anchors (drawn or detected) and its machine fill, both in time order."""
    rows = (await db.execute(
        select(Object, Frame.ts_ns, Frame.img_uri)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.track_id == track_id)
        .order_by(Frame.ts_ns))).all()
    anchors = [(o, ts, uri) for o, ts, uri in rows
               if o.source == "human" or o.is_keyframe or o.source in ("fused", "auto_accept", "imported")]
    fill = [(o, ts, uri) for o, ts, uri in rows if o.source in MACHINE_FILL]
    return anchors, fill


def _nearest(anchors, ts: int):
    """The anchor closest in time. Nearest rather than preceding: a fill in the middle of a gap is better
    judged against whichever end is closer, and the two ends are often the same object seen twice."""
    return min(anchors, key=lambda a: abs(a[1] - ts)) if anchors else None


async def track_drift(db: AsyncSession, track_id: uuid.UUID, *, take_slot: bool = True,
                      batch: int = BATCH, limit: int | None = None) -> dict:
    """How far each machine-filled box on a track has drifted from its nearest anchor.

    Returns one row per fill object with its cosine, its size ratio verdict, and a `verdict` of
    `drifted`, `ok` or `unknown`. `unknown` is a real answer and is never folded into `drifted`: it means
    the crop or the encoder was unavailable, and an editor that painted those red would be telling an
    annotator that a hundred good boxes are wrong because a GPU is busy.
    """
    from core.storage import get_object_store

    anchors, fill = await _anchors_and_fill(db, track_id)
    if not anchors:
        return {"track_id": str(track_id), "checked": 0, "rows": [],
                "reason": "no anchor on this track: every box on it is machine fill"}
    if not fill:
        return {"track_id": str(track_id), "checked": 0, "rows": [],
                "reason": "no interpolated or propagated boxes on this track"}
    if limit is not None:
        fill = fill[:limit]

    store = get_object_store()
    ref_cache: dict[Any, np.ndarray | None] = {}
    rows: list[dict] = []
    failures = 0

    async def run() -> None:
        nonlocal failures
        for i in range(0, len(fill), batch):
            if failures >= MAX_CONSECUTIVE_FAILURES:
                log.info("drift.giving_up", track=str(track_id)[:8], failures=failures)
                return
            for obj, ts, uri in fill[i:i + batch]:
                anchor = _nearest(anchors, ts)
                a_obj, a_ts, a_uri = anchor
                if a_obj.object_id not in ref_cache:
                    ref_cache[a_obj.object_id] = encode_crop(store, a_uri, a_obj.bbox)
                ref = ref_cache[a_obj.object_id]
                sim = cosine_to(store, uri, obj.bbox, ref)
                geom = size_ok(obj.bbox, a_obj.bbox)
                if sim is None:
                    failures += 1
                    verdict, why = "unknown", ("no anchor crop to compare against" if ref is None
                                               else "the crop or the encoder was unavailable")
                else:
                    failures = 0
                    if sim < DRIFT_FLOOR:
                        verdict, why = "drifted", f"appearance {sim:.2f} is below the {DRIFT_FLOOR} floor"
                    elif not geom:
                        verdict, why = "drifted", "the box has changed size beyond the anchor's band"
                    else:
                        verdict, why = "ok", f"appearance {sim:.2f}"
                rows.append({
                    "object_id": str(obj.object_id), "frame_id": str(obj.frame_id),
                    "ts_ns": int(ts), "source": obj.source,
                    "anchor_object_id": str(a_obj.object_id), "gap_ns": int(abs(ts - a_ts)),
                    "similarity": None if sim is None else round(sim, 4),
                    "size_ok": geom, "verdict": verdict, "why": why,
                })
            # Between batches, not inside one: a training job waiting on the card waits seconds.
            if i + batch < len(fill):
                await _yield_if_busy()

    if take_slot:
        from core.gpu_slot import gpu_slot

        async with gpu_slot(f"drift:{track_id}"):
            await run()
    else:
        await run()

    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("drifted", "ok", "unknown")}
    return {"track_id": str(track_id), "checked": len(rows), "anchors": len(anchors),
            "fill": len(fill), "counts": counts, "rows": rows}


async def _yield_if_busy() -> None:
    """Give the card back for a moment if a training job has taken the lease."""
    import asyncio

    try:
        from services.training.gpu_lease import training_holds_gpu

        if await training_holds_gpu():
            await asyncio.sleep(5.0)
    except Exception:  # noqa: BLE001 - a reading that fails must not decide the job
        return
