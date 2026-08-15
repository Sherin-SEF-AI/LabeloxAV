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
    # Flush the caller's pending object first, then confirm it is actually persisted before FK-referencing
    # it. A control sample is a governance nicety (mirror 2% of auto-accepts to measure precision); it must
    # NEVER abort a labeling run, so if the object is not present we skip rather than raise a FK violation
    # that poisons the whole autolabel transaction.
    oid = UUID(object_id)
    await db.flush()
    if await db.get(Object, oid) is None:
        log.warning("control.sample_skipped_missing_object", object_id=object_id)
        return None
    cs = ControlSample(object_id=oid, was_auto_accepted=was_auto_accepted)
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


VERDICTS = ("correct", "incorrect")


async def record_verdict(db: AsyncSession, sample_id: str, verdict: str) -> dict:
    """Rule on one control sample. The verdict is the measurement, so a bad value cannot be stored.

    It used to take any string. `human_verdict` is counted by `measured_precision`, which treats anything
    that is not the literal "incorrect" as correct, so a typo silently reported the gate as more accurate
    than it is. The number this table exists to produce is one a buyer is meant to trust over a
    self-reported figure, which is exactly the number that must not be quietly wrong.
    """
    if verdict not in VERDICTS:
        return {"error": f"verdict must be one of {VERDICTS}"}
    cs = await db.get(ControlSample, UUID(sample_id))
    if cs is None:
        return {"error": "sample not found"}
    cs.human_verdict = verdict
    await db.commit()
    log.info("control.verdict", sample_id=sample_id, verdict=verdict)
    return {"sample_id": sample_id, "human_verdict": verdict}


async def pending_samples(db: AsyncSession, limit: int = 100) -> dict:
    """Control samples awaiting a verdict, with enough of the object to rule on one.

    Seeding has worked all along: this corpus holds 601 samples. Every one of them is unjudged, because the
    verdict route had no caller and there was no way to list what needed judging, so the measured precision
    the drift detector watches has been null since the feature shipped. A worklist nobody can see is the
    same as no worklist.
    """
    from db.models import Frame

    rows = (await db.execute(
        select(ControlSample.sample_id, ControlSample.was_auto_accepted, ControlSample.created_at,
               Object.object_id, Object.class_id, Object.conf, Object.state, Object.frame_id, Frame.session_id)
        .join(Object, Object.object_id == ControlSample.object_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(ControlSample.human_verdict.is_(None))
        .order_by(ControlSample.created_at.desc()).limit(limit))).all()

    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    def _name(cid: int) -> str:
        try:
            return onto.by_id(int(cid)).name
        except KeyError:
            return f"class {cid}"

    return {"count": len(rows), "samples": [
        {"sample_id": str(sid), "was_auto_accepted": bool(auto),
         "object_id": str(oid), "frame_id": str(fid), "session_id": str(sess),
         "class_name": _name(cid), "conf": float(conf) if conf is not None else None, "state": state,
         "crop_url": f"/api/objects/{oid}/crop",
         "created_at": created.isoformat() if created else None}
        for sid, auto, created, oid, cid, conf, state, fid, sess in rows]}


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
