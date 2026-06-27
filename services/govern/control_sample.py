"""Control sample (M4.4): a small random ungated stream that is always sent to human review even when
auto-accepted. The fraction judged incorrect among the auto-accepted controls is the MEASURED true
precision of the gate, which the drift detector watches and a buyer can trust over a self-reported number.
The control sample is never skipped."""

from __future__ import annotations

import random
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ControlSample, Object

log = get_logger("govern_control")


async def maybe_sample(db: AsyncSession, object_id: str, was_auto_accepted: bool,
                       rate: float | None = None, rng: random.Random | None = None) -> str | None:
    """Mirror an object into the control stream with probability `rate` (the gate calls this on each
    auto-accept). Returns the sample id when sampled."""
    rate = rate if rate is not None else get_settings().phase4.govern.control_sample_rate
    draw = (rng.random() if rng is not None else random.random())
    if draw > rate:
        return None
    cs = ControlSample(object_id=UUID(object_id), was_auto_accepted=was_auto_accepted)
    db.add(cs)
    await db.flush()
    return str(cs.sample_id)


async def seed_from_recent_auto_accepts(db: AsyncSession, limit: int = 500, rate: float | None = None,
                                        seed: int = 0) -> dict:
    """Mirror a random fraction of recent auto-accepted objects into the control stream (a batch mirror)."""
    rate = rate if rate is not None else get_settings().phase4.govern.control_sample_rate
    rng = random.Random(seed)
    oids = (await db.execute(
        select(Object.object_id).where(Object.state == "auto_accept").limit(limit))).scalars().all()
    n = 0
    for oid in oids:
        if rng.random() <= rate:
            db.add(ControlSample(object_id=oid, was_auto_accepted=True))
            n += 1
    await db.commit()
    log.info("control.seeded", mirrored=n, pool=len(oids))
    return {"mirrored": n, "pool": len(oids), "rate": rate}


async def record_verdict(db: AsyncSession, sample_id: str, verdict: str) -> dict:
    cs = await db.get(ControlSample, UUID(sample_id))
    if cs is None:
        return {"error": "sample not found"}
    cs.human_verdict = verdict  # correct | incorrect
    await db.commit()
    return {"sample_id": sample_id, "human_verdict": verdict}


async def measured_precision(db: AsyncSession) -> dict:
    """True auto-accept precision = correct / reviewed among auto-accepted controls with a human verdict."""
    reviewed = (await db.execute(select(func.count()).select_from(ControlSample).where(
        ControlSample.was_auto_accepted.is_(True), ControlSample.human_verdict.isnot(None)))).scalar_one()
    incorrect = (await db.execute(select(func.count()).select_from(ControlSample).where(
        ControlSample.was_auto_accepted.is_(True), ControlSample.human_verdict == "incorrect"))).scalar_one()
    precision = None if reviewed == 0 else round(1.0 - incorrect / reviewed, 4)
    pending = (await db.execute(select(func.count()).select_from(ControlSample).where(
        ControlSample.human_verdict.is_(None)))).scalar_one()
    return {"reviewed": int(reviewed), "incorrect": int(incorrect), "precision": precision, "pending": int(pending)}
