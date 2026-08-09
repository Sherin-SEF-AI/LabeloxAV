"""Embedding the false positives, so the failures nobody could see become minable.

`build_failure_clusters` groups an evaluation's misses by appearance, and on the champion it described 32 of
32,576 failures. Its own note said why: a false negative is a gold `Object` and already carries a DINOv3
vector, while a false positive is a `Prediction` and carries none, so 99.9% of what the model got wrong was
invisible to every downstream consumer.

That asymmetry mattered more than the count suggests. A false negative says "the model missed a thing that
was there", which is a recall story. A false positive says "the model saw a thing that was not there", which
is the story about what it hallucinates, and it is the one an operator is most likely to be able to act on:
32,544 spurious boxes on one gold set is a pattern, not noise.

So the crops are cut from the frames and encoded in the same DINOv3 space the gold objects live in, which is
the only way a failure centroid can be compared against the unlabeled pool at all.

Cached in `prediction_embedding` rather than recomputed. A prediction is immutable by this system's own
invariant, so its crop and therefore its vector can never change, and re-encoding thirty thousand crops on
every clustering run would make the feature too slow to use.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import EvalPatch, Frame, Prediction

log = get_logger("failure_embeddings")

# Below this a crop carries too few pixels for the encoder to say anything, and a distant spurious box is
# exactly where the vector would be noise dressed as signal.
MIN_CROP_PX = 16


def _crop(img: np.ndarray, bbox) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return img[y1:y2, x1:x2]


async def embed_false_positives(db: AsyncSession, eval_id: uuid.UUID | str, *,
                                limit: int = 4000, batch: int = 64) -> dict:
    """Encode this evaluation's false-positive crops, returning ids and vectors.

    Bounded by `limit` because a single evaluation here produced 32,544 of them and clustering does not need
    all of them to find the shape of the failure. The bound is reported rather than silently applied: a
    sample described as the whole set is the failure mode this codebase keeps finding.
    """
    from services.intelligence.embed import dinov3
    from services.recall.backends import load_image_bgr

    eid = eval_id if isinstance(eval_id, uuid.UUID) else uuid.UUID(str(eval_id))

    rows = (await db.execute(
        select(EvalPatch.patch_id, EvalPatch.prediction_id, Prediction.bbox, Prediction.class_id,
               Frame.img_uri)
        .join(Prediction, Prediction.prediction_id == EvalPatch.prediction_id)
        .join(Frame, Frame.frame_id == EvalPatch.frame_id)
        .where(EvalPatch.eval_id == eid, EvalPatch.outcome == "fp")
        .limit(limit))).all()
    total = (await db.execute(
        select(EvalPatch.patch_id).where(EvalPatch.eval_id == eid, EvalPatch.outcome == "fp"))).all()

    if not rows:
        return {"ids": [], "vectors": np.zeros((0, 0)), "class_ids": [],
                "encoded": 0, "available": len(total), "skipped": 0}

    store = get_object_store()
    # Grouped by frame so one decode serves every spurious box on it. These cluster heavily: a frame the
    # model misreads tends to be misread several times over.
    by_frame: dict[str, list] = defaultdict(list)
    for pid, pred_id, bbox, cid, uri in rows:
        by_frame[uri].append((pid, pred_id, bbox, cid))

    ids: list[str] = []
    class_ids: list[int] = []
    crops: list[np.ndarray] = []
    skipped = 0
    for uri, items in by_frame.items():
        try:
            img = load_image_bgr(store, uri)
        except Exception:  # noqa: BLE001 - a frame whose image is gone must not end the run
            skipped += len(items)
            continue
        for pid, _pred_id, bbox, cid in items:
            c = _crop(img, bbox)
            if c is None:
                skipped += 1
                continue
            crops.append(c)
            ids.append(str(pid))
            class_ids.append(int(cid) if cid is not None else -1)

    vecs: list[np.ndarray] = []
    for i in range(0, len(crops), batch):
        vecs.append(np.asarray(dinov3.encode_images(crops[i:i + batch])))
    X = np.vstack(vecs) if vecs else np.zeros((0, 0))

    log.info("failure_embeddings.encoded", eval_id=str(eid), encoded=len(ids),
             available=len(total), skipped=skipped)
    return {"ids": ids, "vectors": X, "class_ids": class_ids,
            "encoded": len(ids), "available": len(total), "skipped": skipped,
            # Named so a caller can say "described 4,000 of 32,544" instead of implying it saw them all.
            "truncated": len(total) > len(rows)}
