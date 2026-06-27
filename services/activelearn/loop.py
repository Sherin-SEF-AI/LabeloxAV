"""The retrain-selection loop driver (M4.0). Counts the new training signal that has accrued since the
last closed-loop fine-tune (human corrections, confirmed error candidates, and human-verdicted control
samples), and when enough accumulates (or on demand) fires a train_finetune burst job, whose result the
M4.4 champion/challenger gate decides on. Assembling the training set itself reuses the existing dataset
build (accepted objects); this driver decides WHEN to retrain and on what fresh signal.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ControlSample, ErrorCandidate, Review, TrainingJob

log = get_logger("al_loop")
LOOP_PURPOSE = "closed-loop"


async def _last_retrain_ts(db: AsyncSession):
    return (await db.execute(
        select(func.max(TrainingJob.created_at)).where(TrainingJob.purpose == LOOP_PURPOSE))).scalar_one_or_none()


async def new_signal_count(db: AsyncSession) -> dict:
    """Fresh supervision since the last closed-loop retrain: corrections + confirmed errors + control verdicts.
    Review carries an int64 ns timestamp; the governance tables carry created_at datetimes."""
    since = await _last_retrain_ts(db)
    since_ns = int(since.timestamp() * 1e9) if since is not None else None

    rev_q = select(func.count()).select_from(Review)
    if since_ns is not None:
        rev_q = rev_q.where(Review.ts_ns > since_ns)
    corrections = (await db.execute(rev_q)).scalar_one()

    err_q = select(func.count()).select_from(ErrorCandidate).where(ErrorCandidate.status == "confirmed_error")
    ctl_q = select(func.count()).select_from(ControlSample).where(ControlSample.human_verdict.isnot(None))
    if since is not None:
        err_q = err_q.where(ErrorCandidate.created_at > since)
        ctl_q = ctl_q.where(ControlSample.created_at > since)
    confirmed_errors = (await db.execute(err_q)).scalar_one()
    control_verdicts = (await db.execute(ctl_q)).scalar_one()

    total = int(corrections) + int(confirmed_errors) + int(control_verdicts)
    return {"since": since.isoformat() if since else None, "corrections": int(corrections),
            "confirmed_errors": int(confirmed_errors), "control_verdicts": int(control_verdicts), "total": total}


async def maybe_retrain(db: AsyncSession, compute_target: str = "cloud", force: bool = False,
                        base_weights: str | None = None) -> dict:
    """Fire a closed-loop fine-tune if enough new signal has accrued. Returns the trigger decision."""
    cfg = get_settings().phase4
    signal = await new_signal_count(db)
    if not force and signal["total"] < cfg.activelearn.retrain_min_new:
        return {"triggered": False, "new_signal": signal, "threshold": cfg.activelearn.retrain_min_new}

    from services.training.jobs import TrainJobSpec, enqueue_job

    spec = TrainJobSpec(
        purpose=LOOP_PURPOSE, task_type="detection", compute_target=compute_target,
        base_weights=base_weights, promote=False,  # promotion is the M4.4 champion gate, not here
        gate={"min_map_delta": cfg.govern.min_map_uplift, "min_safe_miou": None,
              "max_class_drop": 0.15},
        notes=f"closed-loop retrain on {signal['total']} new signals")
    job_id = await enqueue_job(spec)
    log.info("al.retrain_triggered", job_id=job_id, **signal)
    return {"triggered": True, "job_id": job_id, "compute_target": compute_target, "new_signal": signal}
