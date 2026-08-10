"""A batch that measures how good the labels are, rather than making them better.

The active-learning queue exists to improve the model, so it surfaces the most uncertain, most novel, most
error-prone objects it can find. Judging that batch answers "how accurate are the worst objects in the
corpus", which is not the question anybody actually asks. The question is what fraction of 570,379
auto-labels are correct, and that number bounds every claim downstream: model precision, dataset quality, a
promotion gate, anything a customer is asked to buy.

Answering it needs a sample that is random with respect to correctness, and nothing here produced one.

Three decisions worth stating.

**Random within class, not random overall.** A uniform sample of this corpus is 24% sedan and 0.02% cattle,
so a rare class would land two objects and its precision would be unknowable. Sampling per class gives every
class its own usable estimate, at the cost of needing to weight back up for a corpus-wide figure, which
`corpus_precision` does.

**Sampled from what was auto-accepted and what is awaiting review alike.** Restricting to auto-accepted
would measure the gate rather than the labels, and the gate is one of the things under test.

**Excludes anything already reviewed.** An object a human has ruled on is not evidence about unreviewed
labels; including it measures agreement with past decisions instead of accuracy.
"""

from __future__ import annotations

import random
import uuid as _uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Object, OntologyClass, Review

log = get_logger("labelops.precision_batch")

# States that represent a machine's assertion that an object is there. `rejected` is excluded because it is
# a human's assertion that it is not, and `accepted` because that is the same human agreeing.
MACHINE_STATES = ("auto_accept", "review", "annotate", "submitted")

MIN_SIDE_PX = 48.0

# The sample the corpus figure is planned around. 300 judged objects pins a rate near 0.9 to about +/-3.4%,
# which is tight enough to act on, and at a few seconds per crop in the grid it is roughly twenty-five
# minutes of somebody's time. Bigger is better and this is the point where the cost stops being trivial.
DEFAULT_TARGET = 300

# Below this a class has its own number that is too wide to mean anything, so spreading a fixed budget over
# forty classes buys forty useless estimates instead of ten usable ones.
MIN_PER_CLASS = 25


async def build_precision_batch(db: AsyncSession, *, batch_id: str | None = None,
                                target: int = DEFAULT_TARGET,
                                classes: list[str] | None = None,
                                seed: int = 7, min_side_px: float = MIN_SIDE_PX) -> dict:
    """Stamp a random, per-class sample of machine labels for review. Returns what was selected.

    Budgeted by `target`, the total number of objects somebody will actually judge, because that is the
    quantity a person can decide about. Spreading a budget across every class in the ontology produces a
    batch nobody finishes and per-class intervals too wide to act on, so the budget goes to the commonest
    classes first and the rest are left for a later, larger pass.

    Uses the same `provenance.flywheel.cycle_id` marker the mined batches use, so it opens in the crop grid
    with no new plumbing: /review/grid?flywheel=<batch_id>.
    """
    bid = batch_id or f"precision-{_uuid.uuid4().hex[:8]}"
    rng = random.Random(seed)

    wanted = classes
    if not wanted:
        rows = (await db.execute(
            select(OntologyClass.name, func.count(Object.object_id))
            .join(Object, Object.class_id == OntologyClass.id)
            .where(Object.state.in_(MACHINE_STATES))
            .group_by(OntologyClass.name)
            .having(func.count(Object.object_id) >= MIN_PER_CLASS)
            .order_by(func.count(Object.object_id).desc()))).all()
        # Only as many classes as the budget can give a usable estimate to.
        wanted = [r[0] for r in rows][:max(1, target // MIN_PER_CLASS)]

    per_class = max(MIN_PER_CLASS, target // max(1, len(wanted)))

    reviewed = select(Review.object_id).distinct().scalar_subquery()
    picked: list[str] = []
    per_class_counts: dict[str, int] = {}

    for name in wanted:
        # Ordered randomly in SQL rather than fetching the class and sampling in Python: some classes hold
        # 134,000 objects and the point is a sample, not a scan.
        ids = (await db.execute(
            select(Object.object_id)
            .join(OntologyClass, OntologyClass.id == Object.class_id)
            .where(OntologyClass.name == name,
                   Object.state.in_(MACHINE_STATES),
                   Object.object_id.notin_(reviewed),
                   # The same judgeability floor the mined batches use. A verdict on a 12px crop is a guess,
                   # and a guessed verdict poisons a precision estimate exactly as it poisons a label.
                   func.least(Object.bbox[3] - Object.bbox[1],
                              Object.bbox[4] - Object.bbox[2]) >= min_side_px)
            .order_by(func.random())
            .limit(per_class))).scalars().all()
        if ids:
            per_class_counts[name] = len(ids)
            picked.extend(str(i) for i in ids)

    rng.shuffle(picked)   # so a reviewer does not work through one class at a time and drift

    if picked:
        from sqlalchemy import text
        await db.execute(text("""
            update object
               set provenance = coalesce(provenance, '{}'::jsonb)
                                || jsonb_build_object('flywheel',
                                     coalesce(provenance->'flywheel', '{}'::jsonb)
                                     || jsonb_build_object('cycle_id', cast(:bid as text),
                                                           'reason', 'random sample to measure label precision'))
             where object_id::text = any(cast(:ids as text[]))"""), {"bid": bid, "ids": picked})
        await db.commit()

    log.info("labelops.precision_batch", batch_id=bid, objects=len(picked), classes=len(per_class_counts),
             target=target, per_class=per_class)
    return {"batch_id": bid, "objects": len(picked), "target": target,
            "classes_covered": len(per_class_counts), "per_class": per_class_counts,
            # The states matter in the link. Triage defaults to review and annotate, and this batch
            # deliberately includes auto_accept objects so the gate is measured too, so without them the
            # grid would silently hand back a subset and the precision figure would exclude exactly the
            # labels the gate was most confident about.
            "review_at": f"/review/grid?flywheel={bid}&states={','.join(MACHINE_STATES)}",
            "note": ("random within class, so each class gets its own estimate; use corpus_precision to "
                     "weight them back to a corpus-wide figure")}


async def corpus_precision(db: AsyncSession, batch_id: str, *, confidence: float = 0.95) -> dict:
    """Precision per class and for the corpus, from whatever of the batch has been reviewed.

    The corpus figure re-weights the per-class rates by how common each class actually is, because the batch
    deliberately over-samples rare classes and a straight average of the per-class rates would let cattle
    count as much as sedan.
    """
    from services.labelops.sampling import wilson_interval

    rows = (await db.execute(
        select(OntologyClass.name, Object.state, func.count(Object.object_id))
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        .where(Object.provenance["flywheel"]["cycle_id"].astext == batch_id)
        .group_by(OntologyClass.name, Object.state))).all()

    judged: dict[str, dict] = {}
    for name, state, n in rows:
        d = judged.setdefault(name, {"correct": 0, "wrong": 0, "unjudged": 0})
        if state in ("accepted", "submitted"):
            d["correct"] += int(n)
        elif state == "rejected":
            d["wrong"] += int(n)
        else:
            d["unjudged"] += int(n)

    # How common each class is among machine labels, for the re-weighting.
    pop = dict((await db.execute(
        select(OntologyClass.name, func.count(Object.object_id))
        .join(Object, Object.class_id == OntologyClass.id)
        .where(Object.state.in_(MACHINE_STATES))
        .group_by(OntologyClass.name))).all())
    pop_total = sum(pop.values()) or 1

    per_class = {}
    num = den = 0.0
    total_judged = total_correct = 0
    for name, d in sorted(judged.items()):
        n = d["correct"] + d["wrong"]
        ci = wilson_interval(d["correct"], n, confidence)
        per_class[name] = {**ci, "unjudged": d["unjudged"],
                           "corpus_share": round(pop.get(name, 0) / pop_total, 5)}
        total_judged += n
        total_correct += d["correct"]
        if n and ci["p"] is not None:
            w = pop.get(name, 0)
            num += ci["p"] * w
            den += w

    overall = wilson_interval(total_correct, total_judged, confidence)
    return {
        "batch_id": batch_id,
        "judged": total_judged,
        "remaining": sum(d["unjudged"] for d in judged.values()),
        # The unweighted figure is what the sample literally says; the weighted one is what the corpus is
        # like. They differ because the batch over-samples rare classes on purpose, and quoting the first as
        # if it were the second is the mistake this function exists to prevent.
        "sample_precision": overall,
        "corpus_precision": round(num / den, 4) if den else None,
        "per_class": per_class,
    }
