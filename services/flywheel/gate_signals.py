"""Turn a blocked promotion into structured labeling demand.

The flywheel already routes a label budget at demands, but it builds those demands from corpus class *share*
(services/flywheel/signals.py): a class holding too small a fraction of the labeled corpus is treated as
starved. That signal is wrong for the case that actually blocks promotion. Across five operational retrain
iterations, pedestrian held 514 training instances (not starved by share) and still sat at 0.02 recall, so a
share-based allocator would have called it healthy while the safety gate blocked on it. The blocking metric is
per-class *recall against the floor*, not representation.

So read the gate's own arithmetic instead. These functions recompute the recall verdict from the stored metric
dicts rather than parsing the human-readable `reasons` strings, because those strings are for people and their
wording is free to change; the numbers are the contract.

`recall_demands` is pure over two metric dicts and the ontology, and emits demands in the shape
services/flywheel/allocator.py already consumes (`slice`, `weight`, `safety_weight`), enriched with the
diagnosis a reviewer needs: what the class scored, what it had to clear, and whether it missed a floor or
regressed against the incumbent.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ModelRegistry, ModelRun
from services.recall.gate import (
    _is_safety,  # noqa: PLC2701  (the gate's own definition of safety, not a copy)
)

log = get_logger("gate_signals")

# A floor miss and a regression are not equally urgent. A class under the absolute floor is unsafe on its own
# terms; a class that regressed is still above the floor and only lost ground, so it weighs less per unit of
# deficit. The multiplier keeps both comparable inside one budget.
FLOOR_MISS_URGENCY = 2.0
REGRESSION_URGENCY = 1.0


def recall_demands(challenger: dict, champion: dict | None, onto, rcfg=None) -> list[dict]:
    """Per-class labeling demands implied by the recall half of the champion gate.

    Returns one demand per safety class that either sits below `safety_recall_floor` or has lost more than
    `safety_recall_max_drop` against the incumbent, heaviest deficit first. An empty list means recall is not
    what is blocking promotion, and the caller should not manufacture work.
    """
    if rcfg is None:
        rcfg = get_settings().phase4.recall

    chal = challenger.get("per_class_recall") or {}
    champ = (champion or {}).get("per_class_recall") or {}
    if not chal:
        # Fail-closed, exactly as the gate does: no per-class recall means we cannot say which class is short,
        # and inventing a demand would send a human to label the wrong thing.
        return []

    floor = float(rcfg.safety_recall_floor)
    max_drop = float(rcfg.safety_recall_max_drop)

    demands: list[dict] = []
    for cn, raw in chal.items():
        if not _is_safety(cn, onto):
            continue
        observed = float(raw)
        baseline = float(champ[cn]) if cn in champ else None

        if observed < floor:
            kind, target = "floor_miss", floor
        elif baseline is not None and observed < baseline - max_drop:
            kind, target = "regression", baseline - max_drop
        else:
            continue

        deficit = round(target - observed, 4)
        urgency = FLOOR_MISS_URGENCY if kind == "floor_miss" else REGRESSION_URGENCY
        demands.append({
            # keys the existing allocator consumes
            "slice": cn,
            "weight": max(deficit * urgency, 1e-3),
            "safety_weight": 2.0,  # every class reaching here is VRU or animal, so all are protected
            "reason": (f"{cn} recall {observed:.3f} below floor {floor:.2f}" if kind == "floor_miss"
                       else f"{cn} recall {observed:.3f} regressed from {baseline:.3f}"),
            # the diagnosis, for the reviewer and the API
            "class_name": cn,
            "metric": "recall",
            "kind": kind,
            "observed": round(observed, 4),
            "target": round(target, 4),
            "baseline": round(baseline, 4) if baseline is not None else None,
            "deficit": deficit,
        })

    demands.sort(key=lambda d: (-d["weight"], d["class_name"]))
    return demands


async def demands_for_run(db: AsyncSession, run_id: str, task: str = "detection") -> dict:
    """Load a recorded training run and the serving champion, and diagnose what recall is blocking.

    The champion baseline comes from the registry rather than the run's own `baseline_metrics`, because a run
    is compared against whatever was champion when it trained; promotion is decided against whatever is
    champion now.
    """
    from services.autolabel.ontology import get_ontology

    run = await db.get(ModelRun, run_id)
    if run is None:
        raise ValueError(f"training run {run_id} not found")

    champ = (await db.execute(
        select(ModelRegistry).where(ModelRegistry.task == task, ModelRegistry.is_champion.is_(True))
    )).scalars().first()

    demands = recall_demands(run.metrics or {}, (champ.gold_metrics if champ else None), get_ontology())
    log.info("gate_signals.diagnosed", run=run_id, blocked_classes=[d["class_name"] for d in demands])
    return {
        "run_id": run_id,
        "champion": champ.model_version if champ else None,
        "blocking": bool(demands),
        "demands": demands,
    }


async def latest_blocked_run(db: AsyncSession) -> str | None:
    """The most recent unpromoted run, which is the one a gate-directed batch should be aimed at.

    Lets a caller ask "what should I label next" without first having to know which run last failed.
    """
    row = (await db.execute(
        select(ModelRun.run_id)
        .where(ModelRun.promoted.is_(False))
        .order_by(ModelRun.created_at.desc())
        .limit(1)
    )).scalars().first()
    return str(row) if row else None
