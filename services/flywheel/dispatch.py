"""Action a flywheel cycle's work order: route its real candidate objects into the review queue. This is the
dispose half of the loop the controller proposes: for each starved safety class the cycle allocated labels to,
the labelable candidates are bumped into 'review' and stamped with the cycle provenance, recorded as one
reversible AgentRun. It never touches a human-accepted object, and reverting the run restores every object's
prior state exactly. Only candidates that actually exist are dispatched; the shortfall stays a collection task,
not a phantom label."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun, FlywheelCycle, Object


async def _resolve_cycle(db: AsyncSession, cycle_id: str | None) -> FlywheelCycle:
    if cycle_id:
        row = await db.get(FlywheelCycle, uuid.UUID(cycle_id))
        if row is None:
            raise ValueError("cycle not found")
        return row
    row = (await db.execute(
        select(FlywheelCycle).order_by(FlywheelCycle.created_at.desc()).limit(1))).scalars().first()
    if row is None:
        raise ValueError("no flywheel cycle on record; run one first")
    return row


async def dispatch_worklist(db: AsyncSession, cycle_id: str | None = None, slices: list[str] | None = None,
                            per_slice_cap: int = 300, created_by: str = "flywheel") -> dict:
    """Send a cycle's labelable candidates to the review queue as one reversible run.

    cycle_id defaults to the latest cycle. slices restricts to specific classes (default: all the cycle
    allocated labels to). For each class, up to min(allocated labels, per_slice_cap) not-yet-accepted objects
    are bumped to 'review', stamped provenance.flywheel = {cycle_id, slice}, and recorded in a
    kind='flywheel' AgentRun so the dispatch is one click to revert."""
    cycle = await _resolve_cycle(db, cycle_id)
    cid = str(cycle.cycle_id)
    cid_by_slice = {d["slice"]: d["class_id"] for d in (cycle.signals or {}).get("regressions", [])}
    allocation = {a["slice"]: a["labels"] for a in (cycle.allocation or [])}
    wanted = slices or list(allocation.keys())

    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    by_slice: dict[str, int] = {}

    for slice_name in wanted:
        class_id = cid_by_slice.get(slice_name)
        if class_id is None:
            continue
        cap = min(allocation.get(slice_name, 0), per_slice_cap)
        if cap <= 0:
            continue
        objs = (await db.execute(
            select(Object).where(Object.class_id == class_id, Object.state != "accepted")
            .order_by(func.abs(Object.conf - 0.5).asc())   # most uncertain first
            .limit(cap))).scalars().all()
        n = 0
        for obj in objs:
            from_state = obj.state
            obj.state = "review"
            obj.provenance = {**(obj.provenance or {}),
                              "flywheel": {"cycle_id": cid, "slice": slice_name},
                              "agent_run_id": str(run_id)}
            changes[str(obj.object_id)] = {"from_state": from_state, "to_state": "review",
                                           "from_source": obj.source, "to_source": obj.source}
            n += 1
        if n:
            by_slice[slice_name] = n

    dispatched = sum(by_slice.values())
    run = AgentRun(run_id=run_id, kind="flywheel", scope={"cycle_id": cid},
                   status="committed", policy={"per_slice_cap": per_slice_cap, "slices": wanted},
                   counts={"routed_review": dispatched, "by_slice": by_slice},
                   changes=changes, created_by=created_by)
    db.add(run)
    await db.commit()
    return {"run_id": str(run_id), "cycle_id": cid, "dispatched": dispatched, "by_slice": by_slice,
            "revert": f"POST /api/agent/runs/{run_id}/revert"}
