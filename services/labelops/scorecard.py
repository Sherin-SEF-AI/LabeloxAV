"""One view of how good a labeller is, per class, whether they are a person or a vendor.

Four scores existed and none of them added up.

`annotator_scorecards` gives throughput and honeypot accuracy per person, and no class dimension.
`workforce_rating` gives batch accept rate per vendor, no class dimension, and is computed and rendered
nowhere: no page in `web/` calls `/api/workforce` at all. `op_precision.measure_all` scores agent
operations, not labellers. `control_sample.measured_precision` scores the auto-accept gate.

So the question you have to answer before you can price work or route it (how good is this labeller at this
class) had four partial answers on four surfaces, and the arithmetic between them was left to whoever was
looking.

Two things this refuses to do.

**It does not average a rate over classes.** A labeller who is excellent at `car` and hopeless at
`traffic_sign` has no single accuracy, and the mean of the two is a number about neither. The per-class
table is the answer and the overall figure is reported beside its sample count, not instead of it.

**It does not rank on a point estimate.** Three correct out of three is 1.0 and means very little; ninety
out of a hundred is 0.9 and means a great deal. Every rate here carries its Wilson interval and ranking uses
the lower bound, which is the same discipline `workforce_rating` and the error detectors already use. A
labeller nobody has checked is reported as unproven rather than perfect.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import (
    JobAgreement,
    LabelJob,
    Object,
    OntologyClass,
    Review,
    User,
    Workforce,
    WorkforceAssignment,
)
from services.labelops.sampling import wilson_interval

log = get_logger("labelops.scorecard")

# A human ruling that endorses the label, and one that corrects it. Same split op_precision uses, and for
# the same reason: `adjust_geometry` is in neither, because a nudged box is a right answer with an imprecise
# outline rather than a wrong one.
ENDORSING = ("confirm", "accept")
CORRECTING = ("reject", "reclassify", "reclassify_track", "error_fix")

# Below this many judged labels in a class, the rate is an anecdote. Matches op_precision's floor so two
# surfaces cannot disagree about what counts as evidence.
MIN_JUDGED = 25


def _rate(correct: int, judged: int, confidence: float = 0.95) -> dict:
    ci = wilson_interval(correct, judged, confidence)
    return {**ci, "proven": judged >= MIN_JUDGED,
            "note": None if judged >= MIN_JUDGED else
            f"only {judged} judged labels: unproven, which is not the same as poor"}


async def _per_class_for_people(db: AsyncSession) -> dict[str, dict[str, dict]]:
    """For each annotator, how their labels fared when somebody later ruled on them, by class.

    The join is Object -> Review: the annotator is on the object (since 0091), the verdict is on the review.
    A label nobody ever looked at counts toward neither side, which is what stops an unread labeller from
    measuring at 100%.
    """
    rows = (await db.execute(
        select(Object.annotator_id, OntologyClass.name, Review.action, func.count())
        .join(Review, Review.object_id == Object.object_id)
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        .where(Object.annotator_id.isnot(None),
               Review.action.in_([*ENDORSING, *CORRECTING]),
               # A person confirming their own label is not evidence about that label.
               Review.user_id != Object.annotator_id)
        .group_by(Object.annotator_id, OntologyClass.name, Review.action))).all()

    out: dict[str, dict[str, dict]] = {}
    for annotator_id, class_name, action, n in rows:
        per = out.setdefault(str(annotator_id), {}).setdefault(class_name, {"correct": 0, "judged": 0})
        per["judged"] += int(n)
        if action in ENDORSING:
            per["correct"] += int(n)
    return out


async def _agreement_for_people(db: AsyncSession) -> dict[str, dict]:
    """How often this annotator's replica partner agreed with them.

    Distinct evidence from review: a reviewer sees one answer and rules on it, while a replica partner
    produced their own answer without seeing this one. It is the only measurement here that does not depend
    on somebody senior having found time.
    """
    rows = (await db.execute(
        select(LabelJob.assignee_id, func.count(JobAgreement.agreement_id),
               func.avg(JobAgreement.n_disagreements))
        .join(JobAgreement, (JobAgreement.job_a_id == LabelJob.job_id)
              | (JobAgreement.job_b_id == LabelJob.job_id))
        .where(LabelJob.assignee_id.isnot(None))
        .group_by(LabelJob.assignee_id))).all()
    return {str(uid): {"frames_compared": int(n), "mean_disagreements": round(float(d or 0.0), 2)}
            for uid, n, d in rows}


async def people(db: AsyncSession, *, confidence: float = 0.95) -> list[dict]:
    """One row per person who has labelled or reviewed anything, with their per-class record.

    The union matters. `annotator_scorecards` lists people by their Review trail, which is the trail a
    *reviewer* leaves; a person who draws boxes all day and rules on nobody else's work has no reviews at
    all and was missing from it entirely. Their labels are exactly what this is supposed to score.
    """
    from services.labelops.quality import annotator_scorecards

    base = {row["user_id"]: row for row in await annotator_scorecards(db)}
    per_class = await _per_class_for_people(db)
    agreement = await _agreement_for_people(db)

    # Everyone who has attributed labels, whether or not they have ever reviewed anything.
    labellers = (await db.execute(
        select(User.user_id, User.name, User.role)
        .join(Object, Object.annotator_id == User.user_id).distinct())).all()
    for uid, name, role in labellers:
        base.setdefault(str(uid), {"user_id": str(uid), "name": name, "role": role, "reviews": 0,
                                   "total_time_min": 0.0, "mean_time_ms": 0, "median_time_ms": 0,
                                   "jobs": 0, "honeypot_accuracy": None})

    out = []
    for uid, row in base.items():
        classes = per_class.get(uid, {})
        judged = sum(c["judged"] for c in classes.values())
        correct = sum(c["correct"] for c in classes.values())
        out.append({
            **row,
            "kind": "person",
            "judged": judged,
            "accuracy": _rate(correct, judged, confidence),
            "per_class": sorted(
                ({"class_name": name, "judged": c["judged"], "correct": c["correct"],
                  **_rate(c["correct"], c["judged"], confidence)} for name, c in classes.items()),
                key=lambda c: (-c["judged"], c["class_name"])),
            "agreement": agreement.get(uid),
        })
    # Worst proven first: a board exists to say where to look, and an unproven labeller is not a problem to
    # solve, it is work to send.
    out.sort(key=lambda r: (not r["accuracy"]["proven"], r["accuracy"]["lo"]))
    return out


async def _per_class_for_vendors(db: AsyncSession) -> dict[str, dict[str, dict]]:
    """The same question for vendors, joined through the assignment that delivered the labels.

    Vendor labels carry their assignment in provenance (return_ingest) and their job in the column, so the
    path from a label back to the vendor who produced it runs object -> job -> assignment -> workforce.
    """
    rows = (await db.execute(
        select(WorkforceAssignment.workforce_id, OntologyClass.name, Review.action, func.count())
        .join(LabelJob, LabelJob.job_id == WorkforceAssignment.job_id)
        .join(Object, Object.job_id == LabelJob.job_id)
        .join(Review, Review.object_id == Object.object_id)
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        .where(Review.action.in_([*ENDORSING, *CORRECTING]))
        .group_by(WorkforceAssignment.workforce_id, OntologyClass.name, Review.action))).all()

    out: dict[str, dict[str, dict]] = {}
    for wid, class_name, action, n in rows:
        per = out.setdefault(str(wid), {}).setdefault(class_name, {"correct": 0, "judged": 0})
        per["judged"] += int(n)
        if action in ENDORSING:
            per["correct"] += int(n)
    return out


async def vendors(db: AsyncSession, *, confidence: float = 0.95) -> list[dict]:
    """One row per workforce: its batch record, and now its per-class record too.

    The batch rate answers "does this vendor deliver acceptable batches" and the per-class rate answers
    "what are they good at". A vendor can pass every honeypot gate and still be reliably wrong about one
    class, and only the second question finds that.
    """
    from services.labelops.workforce import workforce_rating

    rating = (await workforce_rating(db, confidence=confidence)).get("per_workforce", {})
    per_class = await _per_class_for_vendors(db)
    rows = (await db.execute(select(Workforce))).scalars().all()

    out = []
    for wf in rows:
        wid = str(wf.workforce_id)
        classes = per_class.get(wid, {})
        judged = sum(c["judged"] for c in classes.values())
        correct = sum(c["correct"] for c in classes.values())
        batch = rating.get(wf.name, {})
        out.append({
            "kind": "vendor",
            "workforce_id": wid,
            "name": wf.name,
            "active": wf.active,
            "capabilities": wf.capabilities or {},
            "min_honeypot_accuracy": wf.min_honeypot_accuracy,
            # What the routing already uses, surfaced for the first time: no page in the app has ever
            # called this.
            "batch": {k: batch.get(k) for k in ("decided", "accepted", "rejected", "accept_rate",
                                                "proven", "routing_weight", "note")},
            "judged": judged,
            "accuracy": _rate(correct, judged, confidence),
            "per_class": sorted(
                ({"class_name": name, "judged": c["judged"], "correct": c["correct"],
                  **_rate(c["correct"], c["judged"], confidence)} for name, c in classes.items()),
                key=lambda c: (-c["judged"], c["class_name"])),
        })
    out.sort(key=lambda r: (not r["accuracy"]["proven"], r["accuracy"]["lo"]))
    return out


async def scorecards(db: AsyncSession, *, confidence: float = 0.95) -> dict:
    """People and vendors in one answer, because the question is about labellers, not about employment."""
    ppl = await people(db, confidence=confidence)
    vnd = await vendors(db, confidence=confidence)
    judged = sum(p["judged"] for p in ppl) + sum(v["judged"] for v in vnd)
    log.info("labelops.scorecards", people=len(ppl), vendors=len(vnd), judged=judged)
    return {
        "people": ppl,
        "vendors": vnd,
        "confidence": confidence,
        "min_judged": MIN_JUDGED,
        # The honest caveat, carried with the numbers rather than left to a footnote. Objects reach review
        # because something drew attention to them, so this is a rate over judged labels and not over the
        # labeller's output.
        "caveat": ("accuracy is measured over labels a second person later ruled on; review is not a random "
                   "sample of anybody's work, so this compares labellers rather than estimating their true "
                   "error rate"),
        "judged_total": judged,
    }


async def scorecard_for(db: AsyncSession, user_id: str, *, confidence: float = 0.95) -> dict | None:
    """One person, for their own page. Everyone can see their own record; that is not a ranking."""
    uid = str(uuid.UUID(str(user_id)))
    row = next((r for r in await people(db, confidence=confidence) if r["user_id"] == uid), None)
    if row is None:
        user = await db.get(User, uuid.UUID(uid))
        if user is None:
            return None
        row = {"user_id": uid, "name": user.name, "role": user.role, "kind": "person", "judged": 0,
               "accuracy": _rate(0, 0, confidence), "per_class": [], "agreement": None}
    if not row["judged"]:
        # An empty card and a bad card look the same to the person reading it, and only one of them is
        # about them. Said plainly rather than left as a zero.
        row = {**row, "detail": "nothing you labelled has been ruled on by anybody else yet"}
    return row
