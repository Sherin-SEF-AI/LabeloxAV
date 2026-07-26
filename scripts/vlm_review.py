"""VLM-as-judge auto-review: turn unreviewed rare-class detections into usable training labels.

The closed loop stalls because rider and cattle have ~25 human/accepted/auto_accept labels each while tens of
thousands of their detections sit unreviewed in `review`. Training on the unreviewed ones poisoned iteration
5; a human has not reviewed them. This is the sanctioned automated middle path: the gate already trusts a VLM
confirmation for rare classes (rare_needs_agreement_and_vlm), so run the VLM verifier over the review-state
detections of a class and promote only the ones it confidently confirms to `accepted`, tagged as VLM-reviewed
so the provenance never claims a human did it.

This is a weaker signal than human review and it is selection-biased toward detections the detector and the
VLM already agree on, so a resulting metric gain is "under VLM review", not proof of the human loop. It is run
deliberately and labelled as such.

    python scripts/vlm_review.py --classes rider cattle --per-class 400 --min-conf 0.35
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import defaultdict
from uuid import UUID

from sqlalchemy import select, update

from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import Frame, Object
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_c_qwen3vl import VlmVerifier, make_vlm_client
from services.intelligence.embed.service import _decode

log = get_logger("vlm_review")


async def _candidates(db, class_id: int, min_conf: float, limit: int):
    # Highest-confidence review-state detections first: the detector is surest, so the VLM is most likely to
    # confirm, which is what we want for clean training labels (not the marginal ones).
    rows = (await db.execute(
        select(Object.object_id, Object.bbox, Frame.img_uri, Frame.frame_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.class_id == class_id, Object.state == "review", Object.conf >= min_conf)
        .order_by(Object.conf.desc()).limit(limit))).all()
    return rows


async def review_class(cn: str, per_class: int, min_conf: float, oversample: int, dry: bool) -> dict:
    onto = get_ontology()
    cid = onto.by_name(cn).id
    store = get_object_store()
    verifier = VlmVerifier(make_vlm_client(), onto)

    maker = get_sessionmaker()
    async with maker() as db:
        cands = await _candidates(db, cid, min_conf, per_class * oversample)

    # Group by frame so each image is fetched and decoded once.
    by_frame: dict = defaultdict(list)
    uri_of: dict = {}
    for oid, bbox, uri, fid in cands:
        by_frame[fid].append((oid, bbox))
        uri_of[fid] = uri

    async def _persist(ids: list[UUID]) -> None:
        if dry or not ids:
            return
        async with maker() as db:
            await db.execute(update(Object).where(Object.object_id.in_(ids)).values(
                state="accepted", source="vlm_review"))
            await db.commit()

    confirmed = 0
    seen = 0
    promoted = 0
    pending: list[UUID] = []
    for fid, items in by_frame.items():
        if confirmed >= per_class:
            break
        img = _decode(store, uri_of[fid])
        if img is None:
            continue
        for oid, bbox in items:
            if confirmed >= per_class:
                break
            seen += 1
            res = verifier.verify_object(img, tuple(bbox), cid)
            # Confirmed only when the VLM's own choice is this class: detector + VLM agree.
            if res.class_name == cn:
                confirmed += 1
                pending.append(oid)
                # Persist in small batches so a mid-run interruption keeps its progress.
                if len(pending) >= 50:
                    await _persist(pending)
                    promoted += len(pending)
                    pending = []
    await _persist(pending)
    promoted += len(pending)

    log.info("vlm_review.class", cls=cn, seen=seen, confirmed=confirmed, promoted=promoted)
    return {"class": cn, "seen": seen, "confirmed": confirmed, "promoted": promoted}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["rider", "cattle"])
    ap.add_argument("--per-class", type=int, default=400, help="stop after this many confirmed per class")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--oversample", type=int, default=6, help="candidates to scan per confirmed target")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    setup_logging()

    t0 = time.time()
    results = []
    for cn in args.classes:
        r = await review_class(cn, args.per_class, args.min_conf, args.oversample, args.dry_run)
        results.append(r)
        print(f"{cn}: scanned {r['seen']}, VLM-confirmed {r['confirmed']}, promoted {r['promoted']}", flush=True)
    print(f"\ndone in {time.time() - t0:.0f}s: {results}")


if __name__ == "__main__":
    asyncio.run(main())
