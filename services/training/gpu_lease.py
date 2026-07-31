"""Whether a training job is really holding the GPU right now.

Single-GPU discipline means every GPU-using feature yields to training: loading SAM or an embedder on top of
a running train would OOM and kill a multi-hour job. Ten call sites enforced that by asking whether any
`TrainingJob` row said `status = "running"`.

A status column is not a lease. It says what a process intended, not whether that process still exists. A
training run killed by a crash, an OOM, a reboot or a stopped container leaves its row at `running` forever,
and from that moment interactive segmentation, autolabel, embedding, redetection, VLM QA and the relabel
agent all refuse work, each with a message promising the GPU will free up "until it finishes". It never
finishes. The only recovery is someone noticing and editing the database.

So the lease is taken from the heartbeat instead. `_apply_progress` commits once per epoch, and `updated_at`
carries `onupdate=func.now()`, so a live run touches its row continuously. A row that has not moved in a
long time belongs to a process that is gone, and the GPU it claimed is free.

This is deliberately generous rather than clever. Reading the GPU directly would be more direct and puts an
nvidia-smi call on a request path; a shorter timeout would risk releasing a lease held by a genuinely slow
epoch. An hour is far longer than any epoch this corpus produces and far shorter than the forever the status
column was granting.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.logging import get_logger

log = get_logger("gpu_lease")

# How long a job may go without touching its row before it is presumed dead. Epochs on this corpus take
# seconds to a few minutes; an hour leaves enormous headroom while still recovering automatically.
STALE_AFTER = timedelta(hours=1)


async def training_holds_gpu(db) -> bool:
    """Whether a live training job currently owns the GPU.

    A job counts only while its heartbeat is fresh. Rows left at `running` by a process that died are
    ignored, and reported once so the staleness is visible rather than silently routed around.
    """
    from sqlalchemy import select

    from db.models import TrainingJob

    rows = (await db.execute(
        select(TrainingJob.job_id, TrainingJob.updated_at, TrainingJob.purpose)
        .where(TrainingJob.status == "running"))).all()
    if not rows:
        return False

    now = datetime.now(UTC)
    for job_id, updated_at, purpose in rows:
        if updated_at is None:
            continue
        # A row written by a database default may be naive; compare in UTC either way rather than raising.
        seen = updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=UTC)
        if now - seen <= STALE_AFTER:
            return True
        log.warning("gpu_lease.stale_running_job", job_id=str(job_id), purpose=purpose,
                    last_seen=seen.isoformat(), stale_for_s=int((now - seen).total_seconds()),
                    detail="row still says running but its heartbeat stopped; treating the GPU as free")
    return False


async def gpu_busy_detail(db) -> str | None:
    """The message to refuse with, or None when the GPU is free.

    Returned rather than raised so non-HTTP callers, which skip a job instead of erroring, can share the
    same liveness rule.
    """
    if await training_holds_gpu(db):
        return ("GPU reserved for an active training job. This is paused until it finishes; box review "
                "(accept/reject/reclassify) does not need the GPU and still works.")
    return None
