"""Embedding-outlier label-error detection (M4.1): an object whose DINOv3 embedding sits far from its own
class cluster is a likely mislabel. Per class, the centroid of accepted embeddings defines the cluster;
objects beyond a cosine-distance threshold (a high within-class percentile) are flagged."""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectEmbedding
from services.autolabel.ontology import get_ontology

log = get_logger("ed_outlier")
_ACCEPTED = ("accepted", "auto_accept")


async def detect_embedding_outliers(db: AsyncSession, session_id: str | None = None,
                                    min_per_class: int = 5, pct: float = 97.0) -> list[dict]:
    onto = get_ontology()
    q = (select(Object.object_id, Object.class_id, ObjectEmbedding.dino_vec)
         .join(ObjectEmbedding, ObjectEmbedding.object_id == Object.object_id)
         .where(Object.state.in_(_ACCEPTED)))
    if session_id:
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    rows = (await db.execute(q)).all()
    if not rows:
        return []

    by_class: dict[int, list] = {}
    for oid, cid, vec in rows:
        v = np.asarray(vec, dtype=np.float32)
        by_class.setdefault(cid, []).append((str(oid), v / (np.linalg.norm(v) + 1e-9)))

    out = []
    for cid, members in by_class.items():
        if len(members) < min_per_class:
            continue
        mat = np.stack([v for _, v in members])
        centroid = mat.mean(0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        dist = 1.0 - mat @ centroid  # cosine distance to the class centroid
        thresh = float(np.percentile(dist, pct))
        for (oid, _), d in zip(members, dist):
            if d > thresh and d > 0.35:  # also an absolute floor so tight clusters do not over-flag
                out.append({"object_id": oid, "kind": "embedding_outlier", "score": round(float(d), 4),
                            "proposed_label": None,
                            "detail": {"class_name": onto.by_id(cid).name, "centroid_distance": round(float(d), 4),
                                       "class_threshold": round(thresh, 4)}})
    out.sort(key=lambda x: x["score"], reverse=True)
    log.info("ed.outlier", flagged=len(out))
    return out
