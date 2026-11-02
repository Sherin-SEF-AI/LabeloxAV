"""The order of the review queue ("My queue"), and the measurement that chose it.

The queue is ranked by one formula: how unsure the detector was, doubled for rare classes, boosted when the
mask and box disagree or the paths proposed different classes. It is computed in SQL over every object that
matches the queue's filters, so the queue is the top of the whole pool. It used to over-fetch an unordered
three times the page and rank that, which is the top of whatever Postgres read first.

Why this formula rather than the heavier active-learning value score (services/activelearn/selector.py) or
plain lowest-confidence-first: measured on gold, it finds the most wrong labels per review. `evaluate_orderings`
reruns that measurement on any gold set:

  The pool is every machine label on the gold set's frames that overlaps a gold object (IoU >= 0.3). Gold is
  not exhaustive (a few objects per frame), so a detection with no gold object near it is not known to be
  wrong and is left out; one that overlaps a gold object is right if its class matches and IoU >= 0.5, wrong
  otherwise. Each ordering is then scored by the share of wrong labels a reviewer meets in the first 10%, 25%
  and 50% of the queue, and by average precision. Random order is the baseline: on average it finds 10% of
  errors in the first 10% of reviews, by construction.

Measured 2026-11-02 on the dev corpus (one pool of 147 labels, 79% of them wrong; the two newest gold sets
share 150 frames, so they are one measurement):
this formula met 60% of the wrong labels in the first half of the queue against 50% at random, AP 0.90 and
0.94 where 95% of 200 random orders stay under 0.85; the AL value score 57-59% (AP 0.88-0.90); lowest
confidence first 53%, AP 0.83-0.85, inside the random band.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

import numpy as np
from sqlalchemy import Float, and_, case, cast, func, literal, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldSet, Object

LOW_CONF_FLOOR = 0.6
POOL_IOU = 0.3
MATCH_IOU = 0.5


def rare_class_ids(onto) -> list[int]:
    """Classes the queue doubles: the Indian long tail and the fallback classes."""
    return [c.id for c in onto.classes if c.india or c.l1 == "fallback"]


def mask_box_disagrees(prov: dict) -> bool:
    return prov.get("mask_box_disagree") in (True, "true")


def class_conflict(prov: dict) -> bool:
    """More than one path proposed, and one of them was overruled on class."""
    proposals = prov.get("proposals")
    if not isinstance(proposals, list) or len(proposals) <= 1:
        return False
    return any(isinstance(p, dict) and p.get("verdict") == "overruled" for p in proposals)


def priority_of(conf: float, class_id: int, provenance: dict | None, rare_ids: set[int] | frozenset[int]) -> float:
    """The queue priority of one object. `priority_sql` is the same number, computed in the database."""
    prov = provenance or {}
    rarity = 2.0 if class_id in rare_ids else 1.0
    boost = 1.0 + (0.5 if mask_box_disagrees(prov) else 0.0) + (0.5 if class_conflict(prov) else 0.0)
    return round((1.0 - float(conf)) * rarity * boost, 4)


def priority_sql(rare_ids: list[int]):
    """`priority_of` as a SQL expression over Object, for ORDER BY across the whole pool."""
    prov = Object.provenance
    proposals = prov["proposals"]
    mask_box = case((prov["mask_box_disagree"].astext == "true", 0.5), else_=0.0)
    # Nested rather than one AND: Postgres does not promise to evaluate AND left to right, and
    # jsonb_array_length raises on anything that is not an array. CASE does promise its order.
    overruled = literal_column("""'$.proposals[*] ? (@.verdict == "overruled")'::jsonpath""")
    conflict = case((func.jsonb_typeof(proposals) == "array",
                     case((and_(func.jsonb_array_length(proposals) > 1, func.jsonb_path_exists(prov, overruled)), 0.5),
                          else_=0.0)),
                    else_=0.0)
    rarity = case((Object.class_id.in_(rare_ids), 2.0), else_=1.0) if rare_ids else literal(1.0)
    return (literal(1.0) - cast(Object.conf, Float)) * rarity * (literal(1.0) + mask_box + conflict)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


async def gold_checked_pool(db: AsyncSession, gold_id: str) -> list[tuple[Any, bool]]:
    """Machine labels on the gold frames that overlap a gold object, each marked wrong or right."""
    g = await db.get(GoldSet, gold_id)
    if g is None:
        raise LookupError(f"gold set {gold_id} not found")
    gold = (await db.execute(select(Object).where(Object.object_id.in_(
        [uuid.UUID(str(i)) for i in g.object_ids])))).scalars().all()
    by_frame: dict = {}
    for o in gold:
        by_frame.setdefault(o.frame_id, []).append(o)
    machine = (await db.execute(select(Object).where(
        Object.frame_id.in_(list(by_frame)), Object.source != "human",
        Object.state.in_(("review", "annotate", "auto_accept"))))).scalars().all()
    pool = []
    for m in machine:
        best, match = 0.0, None
        for go in by_frame.get(m.frame_id, []):
            v = _iou(m.bbox, go.bbox)
            if v > best:
                best, match = v, go
        if match is not None and best >= POOL_IOU:
            pool.append((m, match.class_id != m.class_id or best < MATCH_IOU))
    return pool


def _found_within(wrong: list[bool], frac: float, total_wrong: int) -> float:
    k = max(1, int(round(len(wrong) * frac)))
    return sum(wrong[:k]) / total_wrong


def _average_precision(wrong: list[bool], total_wrong: int) -> float:
    hits, s = 0, 0.0
    for i, w in enumerate(wrong, 1):
        if w:
            hits += 1
            s += hits / i
    return s / total_wrong


async def evaluate_orderings(db: AsyncSession, gold_id: str, *, seeds: int = 200) -> dict:
    """How fast each ordering puts a reviewer in front of the labels that are wrong, on one gold set."""
    from services.activelearn.selector import score_candidates
    from services.autolabel.ontology import get_ontology

    pool = await gold_checked_pool(db, gold_id)
    total_wrong = sum(w for _, w in pool)
    out: dict = {"gold_id": gold_id, "pool": len(pool), "wrong": total_wrong,
                 "wrong_rate": round(total_wrong / len(pool), 4) if pool else None,
                 "fractions": [0.1, 0.25, 0.5], "orderings": {}}
    if not pool or total_wrong == 0 or total_wrong == len(pool):
        out["refused"] = "the pool needs both right and wrong labels to compare orderings"
        return out

    rare = frozenset(rare_class_ids(get_ontology()))
    al = await score_candidates(db, object_ids=[m.object_id for m, _ in pool], pool_limit=len(pool))
    value = {it["object_id"]: it["value"] for it in al}
    orders = {
        "queue_formula": sorted(pool, key=lambda t: (-priority_of(t[0].conf, t[0].class_id, t[0].provenance, rare),
                                                     str(t[0].object_id))),
        "al_value": sorted(pool, key=lambda t: (-value.get(str(t[0].object_id), -1.0), str(t[0].object_id))),
        "lowest_confidence": sorted(pool, key=lambda t: (t[0].conf, str(t[0].object_id))),
    }

    def summarise(wrong: list[bool]) -> dict:
        return {"found": [round(_found_within(wrong, f, total_wrong), 4) for f in out["fractions"]],
                "ap": round(_average_precision(wrong, total_wrong), 4)}

    for name, order in orders.items():
        out["orderings"][name] = summarise([w for _, w in order])
    rnd = []
    for seed in range(seeds):
        shuffled = [w for _, w in pool]
        random.Random(seed).shuffle(shuffled)
        rnd.append(summarise(shuffled))
    out["orderings"]["random"] = {
        "found": [round(float(np.mean([r["found"][i] for r in rnd])), 4) for i in range(len(out["fractions"]))],
        "ap": round(float(np.mean([r["ap"] for r in rnd])), 4),
        # How far a random order wanders: an ordering inside this band is not distinguishable from luck.
        "ap_p95": round(float(np.percentile([r["ap"] for r in rnd], 95)), 4),
    }
    out["best"] = max(orders, key=lambda n: out["orderings"][n]["ap"])
    return out


__all__ = ["LOW_CONF_FLOOR", "class_conflict", "evaluate_orderings", "mask_box_disagrees", "gold_checked_pool", "priority_of", "priority_sql",
           "rare_class_ids"]
