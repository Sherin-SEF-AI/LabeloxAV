"""The Annotation Copilot: turns a reviewer's repeated correction into a one-click batch fix.

While a reviewer works, it watches their recent reclassifications, finds a repeated transition (e.g.
e_auto -> autorickshaw), pre-fetches the visually most similar cases still carrying the wrong label via the
DINOv3 object index, and offers to relabel them all as one reversible run routed through review. It extends
the find-similar primitive into a proactive assistant. It proposes only: the batch lands in review, never
auto-applied, and reverts exactly.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Object, ObjectEmbedding, Review

log = get_logger("agent.annotation_copilot")

_KIND = "copilot_batch"


async def detect_pattern(db: AsyncSession, user_id=None, *, lookback: int = 50, min_count: int = 3) -> dict | None:
    """The reviewer's most frequent recent class correction, if it repeats enough to be a pattern."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    q = select(Review.object_id, Review.before, Review.after).order_by(Review.ts_ns.desc()).limit(lookback)
    if user_id is not None:
        q = q.where(Review.user_id == user_id)
    rows = (await db.execute(q)).all()
    trans: Counter = Counter()
    examples: dict = defaultdict(list)
    for oid, before, after in rows:
        b = (before or {}).get("class_id")
        a = (after or {}).get("class_id")
        if b is not None and a is not None and int(b) != int(a):
            trans[(int(b), int(a))] += 1
            examples[(int(b), int(a))].append(str(oid))
    if not trans:
        return None
    (b, a), n = trans.most_common(1)[0]
    if n < min_count:
        return None

    def _nm(cid):
        try:
            return onto.by_id(cid).name
        except Exception:  # noqa: BLE001
            return str(cid)

    return {"from_class": b, "to_class": a, "from_name": _nm(b), "to_name": _nm(a),
            "count": n, "example_object_ids": examples[(b, a)][:10]}


async def find_similar(db: AsyncSession, pattern: dict, *, k: int = 60) -> list[str]:
    """The cases most similar to the reviewer's examples that still carry the wrong (from) label."""
    from core.embeddings import object_neighbors

    ex = [uuid.UUID(o) for o in pattern["example_object_ids"]]
    embs = (await db.execute(select(ObjectEmbedding.dino_vec).where(ObjectEmbedding.object_id.in_(ex)))).scalars().all()
    if not embs:
        return []
    centroid = np.mean([np.asarray(e, dtype=np.float32) for e in embs], axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    neigh = await object_neighbors(db, centroid.tolist(), k=k, class_id=pattern["from_class"])
    cand = [uuid.UUID(o) for o, _ in neigh]
    if not cand:
        return []
    still = (await db.execute(select(Object.object_id).where(
        Object.object_id.in_(cand), Object.class_id == pattern["from_class"], Object.source != "human"))).scalars().all()
    ex_set = set(pattern["example_object_ids"])
    return [str(o) for o in still if str(o) not in ex_set]


async def suggest_for_reviewer(db: AsyncSession, user_id=None, *, k: int = 60) -> dict:
    """The proactive suggestion: the detected pattern plus the similar cases a one-click batch would fix."""
    pattern = await detect_pattern(db, user_id)
    if pattern is None:
        return {"pattern": None, "candidates": []}
    candidates = await find_similar(db, pattern, k=k)
    return {"pattern": pattern, "candidates": candidates}


async def apply_batch(db: AsyncSession, object_ids: list[str], to_class: int, *, created_by: str | None = None) -> dict:
    """Relabel the chosen cases to the target class and route them to review, as one reversible run."""
    run_id = uuid.uuid4()
    changes: dict = {}
    for oid in object_ids:
        obj = await db.get(Object, uuid.UUID(oid))
        if obj is None or obj.source == "human" or int(obj.class_id) == int(to_class):
            continue
        changes[oid] = {"from_class": int(obj.class_id), "from_state": obj.state}
        obj.class_id = int(to_class)
        obj.state = "review"
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov.setdefault("copilot_batch", True)
        obj.provenance = prov
    db.add(AgentRun(run_id=run_id, kind=_KIND, scope={}, status="committed", policy={"to_class": int(to_class)},
                    counts={"relabeled": len(changes)}, changes=changes, critic={}, created_by=created_by or "copilot"))
    await db.commit()
    log.info("copilot.batch", run_id=str(run_id), relabeled=len(changes), to_class=to_class)
    return {"run_id": str(run_id), "relabeled": len(changes)}
