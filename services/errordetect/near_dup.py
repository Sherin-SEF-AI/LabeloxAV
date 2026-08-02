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

# The similarity a pair must clear to count as a near-duplicate. Named rather than left as a bare default
# because the score is a margin measured across the band above it, so anything recomputing a stored score
# has to use the same value the candidate was produced under.
DEFAULT_SIM_THRESH = 0.96


def _suspicion(conf: float, dup_margin: float) -> float:
    """How much this object looks like the error, in [0, 1] and on the same scale as the other detectors.

    The score used to be the frame similarity. Because a candidate only exists when similarity already
    cleared the gate, every score landed between 0.96 and 1.0: 45,313 candidates with a mean of 0.992 and a
    tenth percentile of 0.986, a number that said nothing about whether the object was wrong. It mattered
    because `list_candidates` ranks the entire queue by score across detectors, so near-duplicate candidates
    occupied 98.5% of the top thousand and 4,608 of them outranked the best `confident_learning` candidate,
    which is the one detector reporting an actual probability. The queue was sorted by which frames looked
    alike.

    Two things explain a class present here and absent from the near-identical twin: a false positive here,
    or a missed detection there. This detector assumes the first, and the object's own confidence is the
    evidence separating them, so it carries the weight: a detection at 0.30 that its twin frame does not
    corroborate is the case worth reviewing, and one at 0.95 more likely means the twin missed it.

    The duplicate margin modulates rather than decides. At the gate the two frames can still differ enough
    for something to genuinely appear between them; at 1.0 they are the same picture and it cannot.

    This ranks, it does not calibrate. No score here can claim to be P(error) until confirmed and dismissed
    verdicts exist to fit against, and across 298,529 candidates there is currently one.
    """
    return round(max(0.0, min(1.0, 1.0 - conf)) * (0.5 + 0.5 * max(0.0, min(1.0, dup_margin))), 4)


async def detect_near_dup_inconsistent(db: AsyncSession, session_id: str | None = None, *,
                                       sim_thresh: float = DEFAULT_SIM_THRESH, limit_frames: int | None = None) -> list[dict]:
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
        # How far above the gate this pair sits, rescaled across the band the gate admits rather than across
        # [0, 1]. The raw similarity cannot fall below sim_thresh here by construction, so on its own it
        # varies too little to rank anything.
        margin = (float(sim) - sim_thresh) / max(1e-6, 1.0 - sim_thresh)
        for o in my_objs:
            if int(o.class_id) not in nb_classes:
                try:
                    cname = onto.by_id(int(o.class_id)).name
                except Exception:  # noqa: BLE001
                    cname = str(o.class_id)
                out.append({"object_id": str(o.object_id), "kind": "near_dup_inconsistent",
                            "score": _suspicion(float(o.conf or 0.0), margin),
                            "proposed_label": None,
                            "detail": {"near_dup_frame": nb_fid, "similarity": round(float(sim), 4),
                                       "dup_margin": round(margin, 4), "conf": round(float(o.conf or 0.0), 4),
                                       "class": cname, "note": "absent in the near-identical frame"}})
    log.info("ed.near_dup.done", frames=len(frame_ids), flagged=len(out), scope=session_id or "corpus")
    return out
