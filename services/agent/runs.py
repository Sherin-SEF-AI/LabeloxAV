"""AgentRun helpers: serialize a run for the API, and revert one exactly. Revert restores each object's
prior state/source from the recorded transition, but skips any object a human has touched since the run
(source now "human", or the stamped agent_run_id no longer matches) -- the agent never overwrites a person.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Object

log = get_logger("agent.runs")


def run_dict(run: AgentRun) -> dict:
    return {
        "run_id": str(run.run_id), "kind": run.kind, "scope": run.scope, "status": run.status,
        "policy": run.policy, "counts": run.counts, "critic": run.critic,
        "changed": len(run.changes or {}), "created_by": run.created_by,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "reverted_at": run.reverted_at.isoformat() if run.reverted_at else None,
    }


async def revert_run(db: AsyncSession, run_id: uuid.UUID) -> dict:
    run = await db.get(AgentRun, run_id)
    if run is None:
        raise ValueError("run not found")
    if run.status != "committed":
        raise ValueError(f"run is {run.status}, only a committed run can be reverted")

    reverted = 0
    skipped = 0
    for oid, ch in (run.changes or {}).items():
        obj = await db.get(Object, uuid.UUID(oid))
        if obj is None:
            skipped += 1
            continue
        prov = obj.provenance or {}
        # A human took over, or a later agent run owns it now: leave it alone.
        if obj.source == "human" or str(prov.get("agent_run_id")) != str(run_id):
            skipped += 1
            continue
        # Objects the run CREATED (e.g. propagated boxes) are undone by deleting them.
        if ch.get("created"):
            await db.delete(obj)
            reverted += 1
            continue
        # Field-driven restore: put back whatever the run recorded a prior value for.
        if "from_state" in ch:
            obj.state = ch["from_state"]
        if "from_source" in ch:
            obj.source = ch["from_source"]
        if "from_class" in ch:            # a reconcile relabel: restore the original class
            obj.class_id = ch["from_class"]
        if "from_cuboid" in ch:           # an auto-cuboid: clear/restore the 3D box
            obj.cuboid_3d = ch["from_cuboid"]
        obj.version = (obj.version or 0) + 1
        prov = dict(prov)
        prov.pop("agent_run_id", None)
        prov.pop("agent_critic", None)
        prov.pop("agent_cuboid", None)
        obj.provenance = prov
        reverted += 1

    run.status = "reverted"
    run.reverted_at = datetime.now(timezone.utc)
    await db.commit()
    log.info("agent.run.revert", run_id=str(run_id), reverted=reverted, skipped=skipped)
    return {"run_id": str(run_id), "reverted": reverted, "skipped": skipped}


async def list_runs(db: AsyncSession, limit: int = 50) -> list[dict]:
    rows = await db.execute(select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit))
    return [run_dict(r) for r in rows.scalars().all()]
