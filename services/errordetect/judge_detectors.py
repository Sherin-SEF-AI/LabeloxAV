"""Measuring what a detector's flag is worth, using the judge instead of a person.

298,528 candidates carry one human verdict between them, so `detector_precision` reports every detector as
unmeasured and the active-learning selector weights all of them at 0.5, which is a placeholder standing in
for the thing the ranking most needs to know. Nobody is going to rule on 298,528 crops.

The judge already answers the question in a different shape. A detector's flag claims "this label is
suspect"; the judge answers "is this label correct". Judge says incorrect, the detector was right. Judge
says correct, the detector was wrong. That is per-detector precision without anybody clicking anything.

Four things this gets right, and each of them is a way it could have been quietly wrong.

**A random sample, never the top-scored candidates.** Judging a detector's most confident flags measures its
best work, not its work. The same reason `precision_batch` samples randomly rather than reusing the
active-learning queue.

**Machine verdicts stay out of the human plane.** These write to `machine_verdict`, never to
`error_candidate.status` and never to `review`. A machine confirming its own detector into the queue's
verdict column would corrupt the one honest signal in the table, and the human-verdict precision would then
be measuring the machine.

**Strict and superclass, separately.** A detector flagging a sedan that is really an SUV is technically
right and useless to a safety gate. This corpus is 128 within-superclass refinements to 2 cross-superclass
errors, so a strict-only reading would rate every detector far higher than its usefulness.

**Corrected for the judge's own error, and marked as an estimate.** The judge is not ground truth. What
comes out is labelled a machine estimate throughout, and `detector_precision` keeps reporting the
human-verdict figure separately, because a number derived from a model grading a model is a weaker claim
than a number derived from a person and the two must not be added together.
"""

from __future__ import annotations

import uuid as _uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ErrorCandidate, MachineVerdict, Object, OntologyClass
from services.labelops.sampling import wilson_interval

log = get_logger("errordetect.judge")

# The batch id machine-judged detector samples are stamped with, so they are identifiable as an estimate
# rather than mixed into the measurement batches.
BATCH_PREFIX = "detector-judge"

# Below this many judged candidates a detector's estimated precision is too wide to rank on, the same floor
# the human-verdict path uses. Reported as unmeasured rather than as poor.
MIN_JUDGED = 20


async def sample_candidates(db: AsyncSession, kind: str, n: int, *, seed: int = 7) -> list[dict]:
    """A random sample of one detector's pending candidates.

    Random rather than highest-scored. A detector's top flags are its best case, and measuring those reports
    how good it is when it is most confident, which is not what the queue needs to know.
    """
    rows = (await db.execute(
        select(ErrorCandidate.candidate_id, ErrorCandidate.object_id, ErrorCandidate.score,
               OntologyClass.name)
        .join(Object, Object.object_id == ErrorCandidate.object_id)
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        .where(ErrorCandidate.kind == kind, ErrorCandidate.status == "pending")
        .order_by(func.random())
        .limit(n))).all()
    return [{"candidate_id": str(c), "object_id": str(o), "score": float(s), "class_name": cn}
            for c, o, s, cn in rows]


async def judge_detector(db: AsyncSession, kind: str, *, n: int = 50, client=None,
                         model_version: str | None = None) -> dict:
    """Judge a random sample of one detector's candidates and estimate its precision."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from core.config import get_settings
    from core.timebase import now_ns
    from services.autolabel.ontology import get_ontology
    from services.labelops.judge_calibration import _is_refinement
    from services.labelops.vlm_review import (
        JUDGE,
        _alternatives,
        _ask,
        _load_crop,
        _model_version_for,
        parse_judge_reply,
    )
    from services.llm.router import make_vlm_client

    settings = get_settings()
    onto = get_ontology()
    client = client or make_vlm_client(settings)
    provider = getattr(settings.models.vlm, "vision_provider", "ollama")
    model_version = model_version or _model_version_for(settings, provider)
    margin = settings.models.vlm.crop_margin
    batch_id = f"{BATCH_PREFIX}:{kind}"

    items = await sample_candidates(db, kind, n)
    confirmed = dismissed = unsure = unreadable = refinements = cross = failed = 0

    for it in items:
        obj = await db.get(Object, _uuid.UUID(it["object_id"]))
        if obj is None:
            continue
        crop = await _load_crop(db, obj, margin)
        if crop is None or crop.size == 0:
            unreadable += 1
            continue
        given = it["class_name"]
        try:
            given_id = onto.by_name(given).id
        except Exception:  # noqa: BLE001
            continue

        reply = _ask(client, crop, given, _alternatives(onto, given_id), model=model_version)
        if reply is None:
            # The judge was never reached, so there is no verdict. Skipped rather than recorded as `unsure`:
            # an outage must not read as this detector's candidates being unjudgeable.
            failed += 1
            continue
        parsed = parse_judge_reply(reply, onto, given_class=given)

        if parsed["verdict"] == "unsure":
            unsure += 1
        elif parsed["verdict"] == "incorrect":
            # The judge agrees the label is wrong, so the detector was right to flag it.
            confirmed += 1
            if _is_refinement(onto, given, parsed["proposed_class_id"]):
                refinements += 1
            elif parsed["proposed_class_id"] is not None:
                cross += 1
        else:
            dismissed += 1

        await db.execute(pg_insert(MachineVerdict).values(
            object_id=obj.object_id, judge=JUDGE, provider=provider, model_version=model_version,
            verdict=parsed["verdict"], proposed_class_id=parsed["proposed_class_id"],
            confidence=parsed["confidence"], agreement=None,
            detail={"given_class": given, "reason": parsed["reason"], "detector": kind,
                    "detector_score": it["score"], "candidate_id": it["candidate_id"]},
            batch_id=batch_id, ts_ns=now_ns(),
        ).on_conflict_do_update(
            constraint="uq_machine_verdict_object_judge_batch",
            set_={"verdict": parsed["verdict"], "proposed_class_id": parsed["proposed_class_id"],
                  "confidence": parsed["confidence"], "batch_id": batch_id, "ts_ns": now_ns()}))

    await db.commit()

    decided = confirmed + dismissed
    strict = wilson_interval(confirmed, decided) if decided else None
    # A flag whose only fault is fine class is not a flag a safety gate wants raised, so the useful reading
    # counts only cross-superclass confirmations.
    superclass = wilson_interval(cross, decided) if decided else None

    out = {
        "kind": kind, "sampled": len(items), "judged": decided, "unsure": unsure,
        "unreadable": unreadable, "failed": failed,
        "confirmed": confirmed, "dismissed": dismissed,
        "refinements_within_superclass": refinements, "cross_superclass": cross,
        "precision_strict": strict,
        "precision_cross_superclass": superclass,
        "usable": decided >= MIN_JUDGED,
        "estimate": True,
        "judge_model": model_version,
        "note": ("machine estimate: a judge grading a detector, not a human verdict. Reported separately "
                 "from detector_precision, which counts human rulings only, and never written to "
                 "error_candidate.status"),
    }
    log.info("errordetect.judge_detector", **{k: v for k, v in out.items()
                                              if k not in ("precision_strict", "precision_cross_superclass",
                                                           "note")})
    return out


async def judge_all_detectors(db: AsyncSession, *, n: int = 50,
                              kinds: list[str] | None = None) -> dict:
    """Estimate every detector's precision from a judged sample of its candidates."""
    if kinds is None:
        kinds = [k for (k,) in (await db.execute(
            select(ErrorCandidate.kind).where(ErrorCandidate.status == "pending").distinct())).all()]

    per_kind = {}
    for kind in sorted(kinds):
        per_kind[kind] = await judge_detector(db, kind, n=n)
    return {"per_kind": per_kind, "sample_per_detector": n, "estimate": True}


async def machine_detector_weights(db: AsyncSession) -> dict[str, float]:
    """Ranking weight per detector from the machine estimate, for detectors no human has ruled on.

    The lower bound of the cross-superclass rate, for two reasons stacked. The lower bound because a small
    sample should not outrank a large one at the same rate, as with every other weight here. And
    cross-superclass rather than strict because a detector whose flags are all fine-class refinements is
    not finding the errors the queue exists to surface, however often it is technically correct.
    """
    rows = (await db.execute(
        select(MachineVerdict.detail["detector"].astext, MachineVerdict.verdict,
               MachineVerdict.proposed_class_id, MachineVerdict.detail["given_class"].astext)
        .where(MachineVerdict.batch_id.startswith(BATCH_PREFIX)))).all()
    if not rows:
        return {}

    from services.autolabel.ontology import get_ontology
    from services.labelops.judge_calibration import _is_refinement

    onto = get_ontology()
    tally: dict[str, dict] = {}
    for kind, verdict, proposed, given in rows:
        if not kind:
            continue
        d = tally.setdefault(kind, {"cross": 0, "decided": 0})
        if verdict == "unsure":
            continue
        d["decided"] += 1
        if verdict == "incorrect" and proposed is not None and not _is_refinement(onto, given, proposed):
            d["cross"] += 1

    return {k: wilson_interval(v["cross"], v["decided"])["lo"]
            for k, v in tally.items() if v["decided"] >= MIN_JUDGED}
