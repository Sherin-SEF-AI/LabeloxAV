"""Near-duplicate detection (M1.1). Two stages within a session:

  stage 1: a perceptual hash (imagehash.phash) per frame, grouped by Hamming distance as a cheap
           candidate filter (stop-and-go footage produces runs of near-identical frames).
  stage 2: confirm each candidate pair with DINOv3 cosine >= threshold (default 0.95).

Confirmed frames are grouped (union-find), one canonical per group is kept (highest quality), the rest
get selected=false so they are excluded from curation and sampling by default. Raw is never deleted: this
flags and selects only. Re-runnable: it resets the session's dup state first.
"""

from __future__ import annotations

import uuid as uuidlib
from collections import defaultdict
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select, update

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, FrameEmbedding
from db.session import get_sessionmaker

log = get_logger("dedup")


def _decode(store, uri):
    try:
        return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


async def dedup_session(session_id: UUID, phash_hamming: int | None = None, dino_cos: float | None = None) -> dict:
    import imagehash
    from PIL import Image

    cfg = get_settings().intel.dedup
    ph_thresh = cfg.phash_hamming if phash_hamming is None else phash_hamming
    cos_thresh = cfg.dino_cos if dino_cos is None else dino_cos
    store, maker = get_object_store(), get_sessionmaker()

    async with maker() as db:
        # reset prior dup state for the session (idempotent re-run)
        await db.execute(update(Frame).where(Frame.session_id == session_id, Frame.dup_group_id.isnot(None))
                         .values(dup_group_id=None, is_dup_canonical=False, dup_score=None, selected=True))
        await db.commit()
        rows = (await db.execute(
            select(Frame.frame_id, Frame.img_uri, Frame.quality, FrameEmbedding.dino_vec)
            .join(FrameEmbedding, FrameEmbedding.frame_id == Frame.frame_id)
            .where(Frame.session_id == session_id, FrameEmbedding.dino_vec.isnot(None))
            .order_by(Frame.ts_ns))).all()

    frames = [{"id": r[0], "uri": r[1], "q": r[2] or 0.0, "vec": np.asarray(r[3], dtype=np.float32)} for r in rows]
    total = len(frames)
    if total < 2:
        return {"session_id": str(session_id), "frames": total, "dup_groups": 0, "redundant": 0, "duplicate_rate": 0.0}

    for f in frames:
        img = _decode(store, f["uri"])
        f["ph"] = imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) if img is not None else None

    parent = {f["id"]: f["id"] for f in frames}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # stage 1 (pHash hamming) then stage 2 (DINOv3 cosine confirm)
    for i in range(total):
        fi = frames[i]
        if fi["ph"] is None:
            continue
        for j in range(i + 1, total):
            fj = frames[j]
            if fj["ph"] is None:
                continue
            if (fi["ph"] - fj["ph"]) <= ph_thresh and float(fi["vec"] @ fj["vec"]) >= cos_thresh:
                union(fi["id"], fj["id"])

    groups: dict = defaultdict(list)
    for f in frames:
        groups[find(f["id"])].append(f)

    updates, n_groups, n_redundant = [], 0, 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_groups += 1
        gid = uuidlib.uuid4()
        canonical = max(members, key=lambda m: m["q"])
        for m in members:
            is_can = m["id"] == canonical["id"]
            score = 1.0 if is_can else float(m["vec"] @ canonical["vec"])
            if not is_can:
                n_redundant += 1
            updates.append((m["id"], gid, is_can, score))

    async with maker() as db:
        for fid, gid, is_can, score in updates:
            await db.execute(update(Frame).where(Frame.frame_id == fid)
                             .values(dup_group_id=gid, is_dup_canonical=is_can, dup_score=score, selected=is_can))
        await db.commit()

    out = {"session_id": str(session_id), "frames": total, "dup_groups": n_groups,
           "redundant": n_redundant, "duplicate_rate": round(n_redundant / total, 3) if total else 0.0}
    log.info("dedup.done", **out)
    return out
