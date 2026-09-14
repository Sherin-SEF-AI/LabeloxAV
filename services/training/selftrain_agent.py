"""Off-hours hook: retrain the detector on what the machine already agrees on, once a week at most.

The trigger is new consensus, not new time. Self-training only says something the last run did not if the
evidence has moved, so the hook counts the pseudo-labels available now against what the last run trained
on and declines with both numbers when the difference is too small to matter.

It never promotes. The candidate goes through the same champion gate and the same propose-by-approval
path as a hand-launched retrain, which is where a person decides.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import TrainingJob

log = get_logger("selftrain_agent")

PURPOSE = "selftrain"
CREATED_BY = "selftrain_agent"
# New consensus pseudo-labels needed since the last run before another is worth the GPU hours.
SELFTRAIN_MIN_NEW = 5000
# One a week at most, matching the detector cadence the measurement agent already keeps.
SELFTRAIN_EVERY_DAYS = 7


async def _last_run(db: AsyncSession) -> TrainingJob | None:
    return (await db.execute(
        select(TrainingJob).where(TrainingJob.purpose == PURPOSE)
        .order_by(TrainingJob.created_at.desc()).limit(1))).scalars().first()


async def maybe_selftrain(db: AsyncSession) -> dict:
    """Launch one self-training job when the consensus has grown enough since the last one."""
    from services.govern.killswitch import get_state
    from services.training.gpu_lease import training_holds_gpu
    from services.training.jobs import TrainJobSpec, enqueue_job
    from services.training.tasks.selftrain import gather_pseudo_labels

    st = await get_state(db)
    if not st.loop_enabled:
        return {"ran": False, "reason": "loop disabled by the killswitch"}
    if await training_holds_gpu(db):
        return {"ran": False, "reason": "training already holds the GPU"}

    last = await _last_run(db)
    if last is not None and last.created_at and \
            last.created_at > datetime.now(UTC) - timedelta(days=SELFTRAIN_EVERY_DAYS):
        return {"ran": False,
                "reason": f"a self-training run started within the last {SELFTRAIN_EVERY_DAYS} days"}
    if last is not None and last.status in ("pending", "running", "queued-cloud"):
        return {"ran": False, "reason": f"the last self-training job is still {last.status}"}

    pseudo = await gather_pseudo_labels()
    if not pseudo["n"]:
        return {"ran": False, "reason": pseudo.get("reason", "no consensus pseudo-label is available")}
    # What the last run actually trained on, read from the spec it was launched with. The executor's
    # `counts` carries image and class totals rather than the consensus size, so recording the number at
    # launch is what lets this compare like with like instead of always seeing the whole corpus as new.
    before = int((((last.config or {}) if last else {}).get("dataset_spec") or {})
                 .get("pseudo_labels_at_launch") or 0)
    grown = pseudo["n"] - before
    if grown < SELFTRAIN_MIN_NEW:
        return {"ran": False,
                "reason": (f"consensus has grown by {grown} since the last run "
                           f"({before} then, {pseudo['n']} now); {SELFTRAIN_MIN_NEW} is the threshold"),
                "pseudo_labels": pseudo["n"], "source": pseudo["source"]}

    name = f"selftrain-{datetime.now(UTC):%Y%m%d}-{uuid.uuid4().hex[:6]}"
    job_id = await enqueue_job(TrainJobSpec(
        purpose=PURPOSE, task_type="selftrain", compute_target="local",
        dataset_spec={"name": name, "pseudo_labels_at_launch": pseudo["n"]}, promote=False,
        notes=(f"self-training on {pseudo['n']} consensus pseudo-labels from {pseudo['source']}; "
               f"{grown} more than the last run")))
    log.info("selftrain.launched", job_id=str(job_id), pseudo=pseudo["n"], source=pseudo["source"])
    return {"ran": True, "job_id": str(job_id), "pseudo_labels": pseudo["n"],
            "source": pseudo["source"], "new_since_last": grown}
