"""Near-duplicate consistency detector: two frames that look nearly identical should carry the same
objects. Using the DINOv3 frame embeddings, find each frame's nearest neighbour in the same session; when
that neighbour is a near-duplicate (cosine above the threshold) yet is missing a class this frame has, the
odd object out is suspect -- most often a false detection that fired on one frame but not its twin. Each is
emitted as a ranked ErrorCandidate for the fix queue. Skips when the neighbour is unlabelled (nothing to
compare) and stays at a high similarity floor so genuine scene change is not mistaken for an error.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, FrameEmbedding, Object

log = get_logger("ed.near_dup")


async def detect_near_dup_inconsistent(db: AsyncSession, session_id: str | None = None, *,
                                       sim_thresh: float = 0.96, limit_frames: int | None = None) -> list[dict]:
    from core.embeddings import frame_neighbors
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    q = select(FrameEmbedding.frame_id).join(Frame, Frame.frame_id == FrameEmbedding.frame_id).where(
        FrameEmbedding.dino_vec.isnot(None))
    if session_id:
        q = q.where(Frame.session_id == UUID(session_id))
    if limit_frames:
        q = q.limit(limit_frames)
    frame_ids = list((await db.execute(q)).scalars().all())

    out: list[dict] = []
    for fid in frame_ids:
        frame = await db.get(Frame, fid)
        emb = await db.get(FrameEmbedding, fid)
        if frame is None or emb is None or emb.dino_vec is None:
            continue
        nbrs = await frame_neighbors(db, emb.dino_vec, space="dino", k=2, exclude_frame_id=fid,
                                     session_id=frame.session_id)
        if not nbrs:
            continue
        nb_fid, sim = nbrs[0]
        if sim < sim_thresh:
            continue
        my_objs = (await db.execute(select(Object).where(Object.frame_id == fid, Object.source != "human"))).scalars().all()
        nb_classes = set((await db.execute(
            select(Object.class_id).where(Object.frame_id == UUID(nb_fid), Object.source != "human"))).scalars().all())
        if not nb_classes:  # neighbour unlabelled: nothing to compare against
            continue
        for o in my_objs:
            if int(o.class_id) not in nb_classes:
                try:
                    cname = onto.by_id(int(o.class_id)).name
                except Exception:  # noqa: BLE001
                    cname = str(o.class_id)
                out.append({"object_id": str(o.object_id), "kind": "near_dup_inconsistent", "score": round(float(sim), 4),
                            "proposed_label": None,
                            "detail": {"near_dup_frame": nb_fid, "similarity": round(float(sim), 4),
                                       "class": cname, "note": "absent in the near-identical frame"}})
    log.info("ed.near_dup.done", frames=len(frame_ids), flagged=len(out), scope=session_id or "corpus")
    return out
