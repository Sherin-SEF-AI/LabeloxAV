"""A blocked gate fires its own unblock attempt, instead of waiting to be noticed.

Before this, a refused promotion was the loop's dead end: `evaluate_and_promote` recorded the refusal,
MLflow logged it, and the machinery built to answer it - the VLM re-review lever that cleared cattle in
iteration 6, and `materialize_gate_batch`, which turns per-class recall deficits into review tasks -
sat behind scripts and endpoints a person had to remember. The measured cost of that seam was months:
every model in this corpus was blocked for a period the flywheel spent idling.

One nightly attempt per blocked run, as one revertible AgentRun:

1. Diagnose: `demands_for_run` names the safety classes short of their recall floors.
2. Free lever first: VLM re-review of each starved class (services/labelops/vlm_promote.py), bounded
   per class, chunked, every write stamped with this run id so `revert_run` undoes the whole night.
3. Then humans: `materialize_gate_batch` mines the remaining deficit into per-class review tasks and
   `notify(gate_batch_ready)` tells whoever is on duty. The lever does not re-diagnose first - the
   deficit is measured against the run's recorded eval and only a retrain moves it - so the batch is
   aimed at the same deficit, minus whatever pool the VLM just promoted out of `review`.

Self-guards, in the fleet idiom: once per day; once per blocked run (a second attempt at the same run
would re-scan a pool the first attempt already drained); never while training holds the GPU (the VLM
is a local model on the same card); and the whole thing declines with a reason rather than raising.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("agent.gate_unblock")

KIND = "gate_unblock"

# Bounds, chosen from the iteration-6 run rather than invented: 400/class cleared cattle in about an
# hour on the local VLM; 200 keeps the nightly attempt under the off-hours window with several classes
# blocked. min_conf and oversample are the script's own defaults.
PER_CLASS_CAP = 200
MIN_CONF = 0.35
OVERSAMPLE = 6
BATCH_BUDGET = 500


async def _default_project_id(db: AsyncSession) -> str | None:
    """The project gate-directed tasks land in: the one named 'default', else the newest image project."""
    from sqlalchemy import select

    from db.models import LabelProject

    row = (await db.execute(select(LabelProject.project_id)
                            .where(LabelProject.name == "default"))).scalar_one_or_none()
    if row is None:
        row = (await db.execute(select(LabelProject.project_id)
                                .where(LabelProject.modality == "image")
                                .order_by(LabelProject.created_at.desc())
                                .limit(1))).scalar_one_or_none()
    return str(row) if row else None


async def maybe_unblock_gate(db: AsyncSession) -> dict:
    """Off-hours hook for the runtime scheduler: one unblock attempt per blocked run, per day."""
    from datetime import UTC, datetime

    from services.agent.runtime.report import latest_run, launch, ran_since
    from services.flywheel.gate_signals import demands_for_run, latest_blocked_run
    from services.training.gpu_lease import training_holds_gpu

    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, KIND, day_start):
        return {"ran": False, "reason": "already ran today"}

    run_id = await latest_blocked_run(db)
    if run_id is None:
        return {"ran": False, "reason": "no blocked training run"}

    last = await latest_run(db, KIND)
    if last and last["status"] == "committed" and (last["report"] or {}).get("target_run") == run_id:
        return {"ran": False, "reason": f"already attempted for run {run_id}; a retrain moves the "
                                        "deficit, not a second scan of the same pool"}

    if await training_holds_gpu(db):
        return {"ran": False, "reason": "training holds the GPU; the VLM shares the card"}

    try:
        diag = await demands_for_run(db, run_id)
    except ValueError as exc:
        return {"ran": False, "reason": str(exc)}
    if not diag["blocking"]:
        return {"ran": False, "reason": "recall is not blocking the latest unpromoted run"}

    project_id = await _default_project_id(db)
    demands = diag["demands"]

    async def worker(agent_run_id):
        from db.session import get_sessionmaker
        from services.agent.runtime.report import finish_run
        from services.flywheel.gate_directed import materialize_gate_batch
        from services.labelops.vlm_promote import promote_class

        report: dict = {"target_run": run_id, "champion": diag.get("champion"),
                        "blocked_classes": [d["class_name"] for d in demands], "levers": {}}
        changes: dict = {}
        status = "committed"
        try:
            for d in demands:
                res = await promote_class(d["class_name"], per_class=PER_CLASS_CAP,
                                          min_conf=MIN_CONF, oversample=OVERSAMPLE,
                                          agent_run_id=agent_run_id)
                changes.update(res.pop("changes", {}))
                report["levers"][d["class_name"]] = res

            maker = get_sessionmaker()
            async with maker() as wdb:
                if project_id is None:
                    report["batch"] = {"skipped": "no label project exists to hold the review tasks"}
                else:
                    mat = await materialize_gate_batch(wdb, run_id, project_id, budget=BATCH_BUDGET)
                    report["batch"] = {"tasks": mat.get("tasks", []),
                                       "total_frames": mat.get("total_frames"),
                                       "exhausted_classes": mat.get("exhausted_classes", []),
                                       "rationale": mat.get("rationale")}

                promoted = sum(v.get("promoted", 0) for v in report["levers"].values())
                n_tasks = len((report.get("batch") or {}).get("tasks", []))
                from services.notify import notify

                await notify(
                    wdb, kind="gate_batch_ready", severity="info",
                    title=f"gate unblock: {promoted} VLM-confirmed, {n_tasks} review tasks for "
                          f"{', '.join(report['blocked_classes'])}",
                    body=report.get("batch", {}).get("rationale"),
                    href="/flywheel", subject_type="training_run", subject_id=run_id,
                    meta={"run_id": run_id, "levers": {k: {kk: vv for kk, vv in v.items()
                                                           if kk != "changes"}
                                                      for k, v in report["levers"].items()}})
        except Exception as exc:  # noqa: BLE001 - the run records its failure; the daemon ticks on
            status = "error"
            report["error"] = str(exc)[:400]
            log.error("gate_unblock.failed", error=str(exc))
        await finish_run(agent_run_id, status=status, report=report, changes=changes)

    launched = await launch(db, KIND, worker, created_by="scheduler",
                            policy={"per_class": PER_CLASS_CAP, "min_conf": MIN_CONF,
                                    "budget": BATCH_BUDGET})
    return {"ran": True, "target_run": run_id, **launched}
