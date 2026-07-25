"""Gate-directed labeling: mine and materialize the review batch that would unblock a promotion.

This is the seam that was still manual. `services/govern/champion.py` decides a challenger cannot ship and
records why; `services/labelops` can hand a reviewer a job; nothing joined the two, so someone read the gate's
reasons and guessed which frames to mine. Five operational iterations died in that gap, and the guess was the
expensive part: iteration 5 trained on unreviewed machine labels and drove pedestrian recall from 0.73 to
0.004, which is what mining without a review step buys.

The chain here is short because every link already existed:

    demands_for_run          which safety classes are short, and by how much   (gate_signals)
      -> allocate_label_budget   split a finite budget across them, safety floored   (allocator)
      -> score_candidates        rank that class's unreviewed objects by value       (activelearn)
      -> create_task             materialize frames into reviewable jobs             (labelops)

What is new is only the join and the budgeting, so the ranking, the ontology and the job spine stay single-
sourced. The output is deliberately a *review* batch, never training data: a gate-directed job exists so a
human accepts or rejects these instances, and nothing here writes an accepted label.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.activelearn.selector import score_candidates
from services.flywheel.allocator import allocate_label_budget
from services.flywheel.gate_signals import demands_for_run

log = get_logger("gate_directed")

# Mining pool per blocked class. The pool is ranked and then truncated to the class's budget, so it must be
# comfortably larger than any realistic budget or the ranking has nothing to choose between.
POOL_PER_CLASS = 4000

# Every blocked class is safety-critical by construction, so each is guaranteed this many labels before the
# rest of the budget is apportioned by deficit. Without a floor, one severely starved class (cattle at 0.0
# recall) would absorb a whole budget and leave a merely-bad class with nothing.
SAFETY_FLOOR = 50


def _class_id(class_name: str) -> int | None:
    from services.autolabel.ontology import get_ontology

    try:
        return get_ontology().by_name(class_name).id
    except Exception:  # noqa: BLE001
        return None


async def mine_for_class(db: AsyncSession, class_name: str, budget: int) -> dict:
    """Rank the unreviewed objects of one class by active-learning value and take the top `budget`.

    Returns the objects kept and the distinct frames holding them. Frames are what a reviewer opens, and one
    frame often carries several instances of the blocked class, so the frame count is normally well below the
    object count. That is a feature: the reviewer sees several relevant instances per frame they open.
    """
    cid = _class_id(class_name)
    if cid is None:
        return {"class_name": class_name, "class_id": None, "objects": [], "frame_ids": [],
                "note": "class not in the ontology"}

    ranked = await score_candidates(db, class_ids=[cid], pool_limit=POOL_PER_CLASS)
    kept = ranked[:budget]
    # dict.fromkeys preserves value order, so the highest-value frames survive a later truncation
    frame_ids = list(dict.fromkeys(it["frame_id"] for it in kept))
    return {
        "class_name": class_name,
        "class_id": cid,
        "pool": len(ranked),
        "objects": kept,
        "frame_ids": frame_ids,
        "exhausted": len(ranked) < budget,
    }


async def plan_gate_batch(db: AsyncSession, run_id: str, *, budget: int = 500) -> dict:
    """Diagnose a blocked run and plan the batch, without writing anything.

    Kept separate from materialization so the plan can be read (and the projected impact argued with) before
    any job exists. A plan over a run the gate does not block returns no allocations rather than inventing
    work to do.
    """
    diag = await demands_for_run(db, run_id)
    if not diag["blocking"]:
        return {**diag, "budget": budget, "allocations": [], "total_frames": 0,
                "rationale": "recall is not blocking this run; no gate-directed batch needed"}

    alloc = allocate_label_budget(diag["demands"], budget, safety_floor=SAFETY_FLOOR)
    by_class = {a["slice"]: int(a["labels"]) for a in alloc}

    allocations = []
    for d in diag["demands"]:
        want = by_class.get(d["class_name"], 0)
        if want <= 0:
            continue
        mined = await mine_for_class(db, d["class_name"], want)
        allocations.append({
            "class_name": d["class_name"],
            "kind": d["kind"],
            "observed": d["observed"],
            "target": d["target"],
            "deficit": d["deficit"],
            "budget": want,
            "mined_objects": len(mined["objects"]),
            "frames": len(mined["frame_ids"]),
            "pool": mined.get("pool", 0),
            "exhausted": mined.get("exhausted", False),
            "frame_ids": mined["frame_ids"],
        })

    total_frames = len({f for a in allocations for f in a["frame_ids"]})
    short = [a["class_name"] for a in allocations if a["exhausted"]]
    rationale = (
        f"{len(allocations)} safety classes blocking; {budget} labels allocated by deficit "
        f"(safety floor {SAFETY_FLOOR}); {total_frames} distinct frames to review"
    )
    if short:
        # Say it out loud. A class whose pool is smaller than its budget cannot be fixed by reviewing harder,
        # and quietly returning fewer frames would read as "the batch covers it" when it does not.
        rationale += (f". Pool exhausted for {short}: the corpus does not hold enough unreviewed instances, "
                      f"so these need collection or autolabel coverage, not review")

    return {**diag, "budget": budget, "allocations": allocations,
            "total_frames": total_frames, "exhausted_classes": short, "rationale": rationale}


async def materialize_gate_batch(db: AsyncSession, run_id: str, project_id: str, *,
                                 budget: int = 500, jobs_of: int = 50) -> dict:
    """Plan the batch, then create one labelops task per blocked class.

    One task per class rather than one combined task, because the classes fail for different reasons and a
    reviewer working a cattle batch is doing different work from one working riders. Per-class tasks also keep
    the projected impact attributable: a class's recall moves or it does not.
    """
    from services.labelops.jobs import create_task

    plan = await plan_gate_batch(db, run_id, budget=budget)
    if not plan["allocations"]:
        return {**plan, "tasks": []}

    tasks = []
    for a in plan["allocations"]:
        if not a["frame_ids"]:
            continue
        name = f"gate: {a['class_name']} recall {a['observed']:.2f} -> {a['target']:.2f} ({run_id})"
        task = await create_task(db, project_id=project_id, name=name,
                                 predicate={"frame_ids": a["frame_ids"]}, jobs_of=jobs_of)
        tasks.append({"class_name": a["class_name"], "task_id": task["task_id"],
                      "n_frames": task["n_frames"], "n_jobs": task["n_jobs"]})
        log.info("gate_directed.task_created", run=run_id, cls=a["class_name"],
                 task=task["task_id"], frames=task["n_frames"])

    return {**plan, "tasks": tasks}
