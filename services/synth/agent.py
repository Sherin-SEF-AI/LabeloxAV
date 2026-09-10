"""Off-hours hook: build copy-paste positives for the classes the gate says are starved.

The signal is the same one gate-directed labelling reads (`gate_signals.demands_for_run` on the latest
blocked run): a class below its recall floor with a measured deficit. A person's verdicts fix that slowly;
this fills the training side in the meantime with real pixels of the class, and the next retrain decides
through the unchanged gate whether it helped. One build per class per week, so a build's effect is seen by
a retrain before the next one lands on top of it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun
from services.synth.copy_paste import KIND, MEMORY_CEILING_FRAC, build, plan_build

SYNTH_FRAMES_PER_CLASS = 500
SYNTH_CLASS_WINDOW = timedelta(days=7)
CREATED_BY = "synth_starved"


async def _classes_built_recently(db: AsyncSession) -> set[str]:
    since = datetime.now(UTC) - SYNTH_CLASS_WINDOW
    rows = (await db.execute(select(AgentRun.policy, AgentRun.status).where(
        AgentRun.kind == KIND, AgentRun.created_at >= since))).all()
    out: set[str] = set()
    for policy, status in rows:
        # A refused build counts too: the reason (no donors) does not change within the week.
        if status in ("running", "committed", "refused"):
            out.update((policy or {}).get("class_names") or [])
    return out


async def maybe_synth_starved(db: AsyncSession) -> dict:
    """Once per class per week, for the classes the latest blocked run is starved on and that have donors."""
    from services.agent.runtime.report import launch
    from services.flywheel.gate_signals import demands_for_run, latest_blocked_run
    from services.govern.killswitch import get_state
    from services.hardening.resources import host

    st = await get_state(db)
    if not st.loop_enabled:
        return {"ran": False, "reason": "loop disabled by the killswitch"}

    run_id = await latest_blocked_run(db)
    if run_id is None:
        return {"ran": False, "reason": "no blocked training run"}
    try:
        diag = await demands_for_run(db, run_id)
    except ValueError as exc:
        return {"ran": False, "reason": str(exc)}
    if not diag["blocking"]:
        return {"ran": False, "reason": "recall is not blocking the latest unpromoted run"}

    recent = await _classes_built_recently(db)
    wanted = [d["class_name"] for d in diag["demands"] if d["class_name"] not in recent]
    if not wanted:
        return {"ran": False, "reason": "every starved class had a build this week"}

    plan = await plan_build(db, class_names=wanted, n_frames=SYNTH_FRAMES_PER_CLASS)
    buildable = [c["class_name"] for c in plan["classes"] if c.get("donors")]
    if not buildable:
        reasons = {c["class_name"]: c.get("reason") for c in plan["classes"]}
        # Record the refusal as a run so the week's guard holds and the digest can show why.
        db.add(AgentRun(kind=KIND, scope={"target_run": run_id}, status="refused",
                        policy={"class_names": wanted, "n_frames": SYNTH_FRAMES_PER_CLASS},
                        counts={"reason": "no donors", "classes": reasons}, changes={}, critic={},
                        created_by=CREATED_BY))
        await db.commit()
        return {"ran": False, "reason": f"no starved class has a masked human donor: {reasons}"}

    mem = host().get("memory_used_frac")
    if mem is not None and mem >= MEMORY_CEILING_FRAC:
        return {"ran": False, "reason": f"host memory at {mem:.0%}; not starting a build"}

    seed = datetime.now(UTC).strftime("%Y%m%d")
    n_frames = SYNTH_FRAMES_PER_CLASS * len(buildable)

    async def worker(agent_run_id):
        await build(agent_run_id, class_names=buildable, n_frames=n_frames, seed=seed, created_by=CREATED_BY)

    res = await launch(db, KIND, worker, created_by=CREATED_BY,
                       policy={"class_names": buildable, "n_frames": n_frames, "seed": seed,
                               "target_run": run_id})
    return {"ran": True, "run_id": res["run_id"], "classes": buildable, "n_frames": n_frames,
            "skipped": {c["class_name"]: c.get("reason") for c in plan["classes"] if not c.get("donors")}}
