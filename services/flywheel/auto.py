"""Drive one adaptive flywheel cycle from real corpus signals (M17 closed on live data). Collects the real
class-starvation demands and ODD coverage gaps, runs the adaptive controller, records the cycle to the ledger,
and builds a concrete labeling work order: the actual candidate objects a human review would improve for each
starved safety class. The work order PROPOSES work (it never mutates a label), matching the app's
propose/dispose contract; a starved class with few labelable candidates left is itself the signal that
collection, not labeling, is the remaining lever."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import FlywheelCycle, Object
from services.flywheel.controller import adaptive_cycle
from services.flywheel.signals import class_starvation, corpus_totals, odd_gaps


async def candidate_counts(db: AsyncSession, class_ids: list[int]) -> dict[int, int]:
    """How many labelable candidates (objects not yet human-accepted) exist per class. Used to split the starved
    classes into those labeling can still improve and those that are empty and need collection."""
    if not class_ids:
        return {}
    rows = (await db.execute(
        select(Object.class_id, func.count(Object.object_id))
        .where(Object.class_id.in_(class_ids), Object.state != "accepted")
        .group_by(Object.class_id))).all()
    return {r[0]: r[1] for r in rows}


async def class_candidates(db: AsyncSession, class_id: int, limit: int) -> list[dict]:
    """The real objects of a class a human review would improve: not yet human-accepted, ranked most-uncertain
    first (confidence closest to 0.5). These are the labelable candidates for the class; when there are fewer
    than the budget asks for, the class is genuinely starved and needs collection."""
    if limit <= 0:
        return []
    uncertainty = func.abs(Object.conf - 0.5)
    rows = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.conf, Object.state, Object.source)
        .where(Object.class_id == class_id, Object.state != "accepted")
        .order_by(uncertainty.asc())
        .limit(min(limit, 500)))).all()
    return [{"object_id": str(r[0]), "frame_id": str(r[1]) if r[1] else None,
             "conf": round(r[2], 4), "state": r[3], "source": r[4]} for r in rows]


async def run_auto_cycle(db: AsyncSession, total_label_budget: int = 2000, safety_floor: int = 200,
                         min_share: float = 0.003, build_work_order: bool = True) -> dict:
    """Collect real signals, run the adaptive cycle, record it, and build the labeling work order.

    Returns the plan (allocation + collection tasks), the real signals that drove it, and a per-class work order
    of candidate objects to label, so the cycle is auditable end to end: a spend traces to the starved class or
    ODD gap that justified it, down to the object ids a reviewer would open."""
    starve = await class_starvation(db, min_share)
    gaps = await odd_gaps(db)
    totals = await corpus_totals(db)

    # Split the starved safety classes by whether labeling can still help: a class with candidates left goes to
    # the label budget; an empty class (no labelable instances) cannot be fixed by labeling, so it joins the
    # collection tasks alongside the ODD scene gaps. This is the honest routing the naive controller misses.
    demands = starve["demands"]
    cand_n = await candidate_counts(db, [d["class_id"] for d in demands])
    labelable = [d for d in demands if cand_n.get(d["class_id"], 0) > 0]
    empty = [d for d in demands if cand_n.get(d["class_id"], 0) == 0]

    regressions = labelable
    safety_slices = [d["slice"] for d in regressions if d["protected"]]
    # collection = scene ODD gaps + the empty safety classes expressed as class-collection gaps
    class_collection = [{"cell": d["slice"], "gap": abs(d["delta"]), "missing": True, "kind": "class"}
                        for d in empty]
    collection_signals = gaps["gaps"] + class_collection
    plan = adaptive_cycle(regressions, collection_signals, total_label_budget, totals["frames"],
                          safety_slices, safety_floor)

    # a lookup from allocated slice name back to its class id, so the work order can query real candidates
    cid_by_slice = {d["slice"]: d["class_id"] for d in regressions}
    work_order = []
    if build_work_order:
        for alloc in plan["allocation"]:
            cid = cid_by_slice.get(alloc["slice"])
            if cid is None or alloc["labels"] <= 0:
                continue
            cands = await class_candidates(db, cid, alloc["labels"])
            work_order.append({
                "slice": alloc["slice"], "class_id": cid, "labels_requested": alloc["labels"],
                "candidates_available": len(cands),
                "shortfall": max(0, alloc["labels"] - len(cands)),   # budget that labeling cannot fill: collect
                "candidates": cands[:50],
            })

    row = FlywheelCycle(
        label_budget=total_label_budget,
        signals={"regressions": regressions, "odd_gaps": gaps["gaps"], "empty_classes": empty,
                 "safety_slices": safety_slices, "source": "corpus", "totals": totals, "min_share": min_share},
        allocation=plan["allocation"],
        collection_tasks=plan["collection_tasks"],
        rationale=plan["rationale"],
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    return {"cycle_id": str(row.cycle_id), **plan,
            "signals": {"n_starved": starve["n_starved"], "total_objects": starve["total_objects"],
                        "n_labelable": len(labelable), "n_empty": len(empty),
                        "n_odd_gaps": gaps["n_gaps"], "frames": totals["frames"],
                        "labelable": regressions, "empty_classes": empty, "odd_gaps": gaps["gaps"]},
            "work_order": work_order}
