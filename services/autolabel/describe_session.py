"""Describe once, find it on every frame: one phrase run over a whole session, written for review.

`describe.py` answers "what in this frame matches the phrase" for a person looking at one frame. This runs
the same detector and segmenter (YOLO-World prompted with the phrase, SAM 2 box-prompted for the masks) over
every frame of a session and writes what it finds as proposals: `source='described'`, `state='review'`,
the phrase on the provenance. Nothing it writes is accepted; the review queue is where it is judged.

**Why the session floor is higher than the frame floor.** On one frame a person sees every proposal before
anything is written, so `describe.MIN_CONF` is deliberately low. Here nobody watches each frame, and an
open-vocabulary detector always finds something for a phrase, so the default floor is `SESSION_MIN_CONF`.
The caller can lower it, and the result says which floor ran.

**No double labels.** A proposal that overlaps an object already on the frame (same class, IoU at or above
`DUP_IOU`) is not written: the object is already labelled, and a second box on it would be a duplicate to
clean up, not a find. A re-run of the same phrase is therefore idempotent.

**Second opinion.** The plan called for Florence-2 open-vocabulary detection fused with the YOLO-World
boxes. Its weights are not in this machine's Hugging Face cache (the cache holds the config only, 44 KB),
and the tool does not download, so this ships YOLO-World + SAM only and the job result says so
(`second_opinion`). `fuse_agreement` is the fusion step, kept and tested, so a second detector is one
function away when its weights are present.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from uuid import UUID

import numpy as np

from core.logging import get_logger

log = get_logger("describe_session")

SESSION_MIN_CONF = 0.25
DUP_IOU = 0.5
MAX_FRAMES = 5000
SECOND_OPINION = ("unavailable: Florence-2 weights are not in the local Hugging Face cache "
                  "(microsoft/Florence-2-base has its config only); YOLO-World + SAM only")


def fuse_agreement(a: list[tuple[list[float], float]], b: list[tuple[list[float], float]],
                   iou_thr: float = 0.5) -> list[dict]:
    """Fuse two detectors' boxes: IoU-matched pairs become one score-weighted box marked as agreed.

    Unmatched boxes from either side are kept, marked by which detector saw them. Greedy matching in order of
    IoU (core/accel/boxes.box_iou_matrix), so each box is used at most once.
    """
    from core.accel.boxes import box_iou_matrix

    out: list[dict] = []
    if not a and not b:
        return out
    used_a: set[int] = set()
    used_b: set[int] = set()
    if a and b:
        iou = box_iou_matrix(np.asarray([x[0] for x in a], dtype=np.float32),
                             np.asarray([x[0] for x in b], dtype=np.float32))
        pairs = sorted(((float(iou[i, j]), i, j) for i in range(len(a)) for j in range(len(b))
                        if iou[i, j] >= iou_thr), reverse=True)
        for v, i, j in pairs:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            (ba, ca), (bb, cb) = a[i], b[j]
            w = (ca + cb) or 1.0
            box = [(ba[k] * ca + bb[k] * cb) / w for k in range(4)]
            out.append({"bbox": box, "conf": max(ca, cb), "agreed": True, "iou": round(v, 4), "from": "both"})
    out += [{"bbox": list(x[0]), "conf": x[1], "agreed": False, "from": "a"} for i, x in enumerate(a)
            if i not in used_a]
    out += [{"bbox": list(x[0]), "conf": x[1], "agreed": False, "from": "b"} for j, x in enumerate(b)
            if j not in used_b]
    return sorted(out, key=lambda d: -d["conf"])


def new_proposals(proposals: list, existing_boxes: list[list[float]], iou_thr: float = DUP_IOU) -> list:
    """The proposals that do not duplicate a box already on the frame."""
    if not proposals or not existing_boxes:
        return list(proposals)
    from core.accel.boxes import box_iou_matrix

    iou = box_iou_matrix(np.asarray([p.bbox for p in proposals], dtype=np.float32),
                         np.asarray(existing_boxes, dtype=np.float32))
    return [p for i, p in enumerate(proposals) if float(iou[i].max()) < iou_thr]


async def run_describe_session(params: dict, *, session_id: UUID | None, job_id: UUID | None = None,
                               user_id: UUID | None = None, progress: Callable | None = None,
                               propose_fn: Callable | None = None) -> dict:
    """The `describe_session` job body. params: phrase, class_name, min_conf?, max_frames?, cam_id?, dry_run?"""
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import Frame, Object
    from db.session import get_sessionmaker
    from services.api.routers.objects import _write_mask
    from services.autolabel.describe import normalize_phrase, propose
    from services.autolabel.ontology import get_ontology
    from services.temporal.sam2_video import _decode

    phrase = normalize_phrase(params.get("phrase", ""))
    if not phrase:
        raise ValueError("a phrase is required")
    onto = get_ontology()
    cname = params.get("class_name")
    if not cname or not onto.has_name(cname):
        raise ValueError(f"unknown class {cname!r}")
    class_id = onto.by_name(cname).id
    sid = UUID(str(session_id or params.get("session_id")))
    min_conf = float(params.get("min_conf", SESSION_MIN_CONF))
    max_frames = min(int(params.get("max_frames", MAX_FRAMES)), MAX_FRAMES)
    dry_run = bool(params.get("dry_run", False))
    propose_fn = propose_fn or propose
    store = get_object_store()

    async with get_sessionmaker()() as db:
        q = select(Frame).where(Frame.session_id == sid)
        if params.get("cam_id"):
            q = q.where(Frame.cam_id == params["cam_id"])
        frames = (await db.execute(q.order_by(Frame.cam_id, Frame.ts_ns).limit(max_frames))).scalars().all()
    if not frames:
        return {"phrase": phrase, "frames": 0, "written": 0, "note": "the session has no frames"}

    t0 = time.monotonic()
    written = skipped_dup = proposed = 0
    per_frame: list[dict] = []
    for n, fr in enumerate(frames):
        img = await asyncio.to_thread(_decode, store, fr.img_uri)
        props = await asyncio.to_thread(propose_fn, img, phrase, min_conf=min_conf)
        proposed += len(props)
        async with get_sessionmaker()() as db:
            existing = (await db.execute(select(Object.bbox).where(
                Object.frame_id == fr.frame_id, Object.class_id == class_id))).scalars().all()
            keep = new_proposals(props, [list(b) for b in existing if b])
            skipped_dup += len(props) - len(keep)
            if keep and not dry_run:
                for p in keep:
                    oid = uuid.uuid4()
                    mask_uri = (_write_mask(store, fr.session_id, fr.frame_id, oid, p.polygons, fr.width, fr.height)
                                if p.polygons else None)
                    db.add(Object(object_id=oid, frame_id=fr.frame_id, class_id=class_id,
                                  bbox=[float(v) for v in p.bbox], conf=round(float(p.conf), 4),
                                  source="described", state="review", mask_uri=mask_uri,
                                  mask_encoding="polygon" if mask_uri else None,
                                  provenance={"described_as": phrase, "job_id": str(job_id) if job_id else None,
                                              "method": "describe_session", "detector": "yolo-world",
                                              "segmenter": "sam2", "min_conf": min_conf,
                                              "requested_by": str(user_id) if user_id else None}))
                await db.commit()
            written += 0 if dry_run else len(keep)
        if keep:
            per_frame.append({"frame_id": str(fr.frame_id), "n": len(keep),
                              "best_conf": round(max(p.conf for p in keep), 4)})
        if progress is not None and (n % 5 == 0 or n == len(frames) - 1):
            await progress(0.02 + 0.97 * (n + 1) / len(frames), f"frame {n + 1}/{len(frames)}")

    result = {"phrase": phrase, "class_name": cname, "frames": len(frames), "proposed": proposed,
              "written": written, "skipped_duplicates": skipped_dup, "frames_with_finds": len(per_frame),
              "top_frames": sorted(per_frame, key=lambda d: -d["best_conf"])[:20], "min_conf": min_conf,
              "dry_run": dry_run, "second_opinion": SECOND_OPINION,
              "seconds": round(time.monotonic() - t0, 1)}
    log.info("describe_session.done", phrase=phrase, frames=len(frames), written=written)
    return result
