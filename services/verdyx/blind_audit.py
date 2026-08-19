"""Blind audits: measuring recall against a denominator that was not built by the model.

Every recall number this engine has ever reported is recall against objects somebody already found, and
finding is not symmetric. Confirming a machine box costs one click; drawing a box the machine missed costs
thirty seconds of looking at empty road. So a review pass produces a label set biased toward what the model
already sees, the gold set sealed from it inherits that bias, and gold recall is an overestimate by an
amount nobody can state. Fixing the prediction plane made the numerator honest. It did nothing to the
denominator, because the denominator was never the predictions.

This module treats the model and a blind human as two independent observers of one population, in the
Lincoln-Petersen sense. The human labels a sample of frames from scratch, never having seen a prediction or
an existing label. Objects found by both, by the model alone, and by the human alone then determine, in
closed form, how many were found by neither, and hence what recall actually is. The arithmetic is
core/accel/recapture.py; this module produces the counts it consumes and stores what comes back.

THE BLINDNESS IS SERVER-SIDE AND MUST STAY THERE. services/api/routers/objects.py::frame_objects refuses to
return anything on an audit frame except the auditor's own audit boxes. Hiding predictions in the editor
would still ship them to the browser, a hidden label is one keystroke from an unhidden one, and afterwards
nothing could distinguish an audit that was blind from one that was not. An audit that leaked is not a
degraded measurement, it is no measurement, because the second observer stops being independent.

TWO QUESTIONS, TWO MATCHING RULES, and they do not sum to each other:

  * The pooled row (class_id null) is CLASS-AGNOSTIC. It asks how many objects are out there, so any
    sufficient overlap counts as a capture whatever either observer called it. This is the standard
    two-observer setup and it is what `population` means.
  * A per-class row is CLASS-AWARE. It asks how many objects of class c each observer correctly
    identified, so a box the model found but called something else is a miss for that class. This is the
    number that is comparable to per-class gold recall, which is why it is computed this way.

Summing the per-class populations therefore does not reproduce the pooled population, and it should not.
They answer different questions and both rows are stored.

Independence is assumed and is not fully met: a small, occluded, badly lit object is harder for both
observers, so the captures correlate positively and the estimated population is biased DOWN. Every number
here is a lower bound on what was missed and an upper bound on recall, and is reported that way.
"""

from __future__ import annotations

import uuid as uuidlib
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.recapture import lincoln_petersen, stratified_recapture
from core.logging import get_logger
from db.models import (
    BlindAudit,
    BlindAuditFrame,
    EvalPatch,
    InferenceRun,
    LabelJob,
    LabelProject,
    LabelTask,
    Object,
    Prediction,
    RecaptureEstimateRow,
)
from services.analytics.evaluation import _greedy_match

log = get_logger("blind_audit")

ESTIMATOR = "chapman-lp-v1"

# Density strata, by how many predictions the run left on the frame at the audit's operating point. Capture
# probability is not constant across the corpus: an empty highway and a crowded junction do not share a
# detection rate, and pooling one collapsed count over both assumes they do.
#
# The boundaries are fixed rather than fitted to each run's histogram, because quantile boundaries would
# make two audits of two runs incomparable, which is most of what an audit is for. They are set where they
# are because they describe Indian road scenes: under ten objects is a quiet frame, ten to thirty is
# ordinary traffic, thirty and up is a crowded junction. On the champion's gold run that splits 157 frames
# 13/82/62, which is a real three-way split; boundaries an order of magnitude lower (the natural choice for
# a sparse highway corpus) put 138 of those frames in one bucket and stratify nothing.
_DENSITY_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("sparse", 0, 10),        # under 10 predictions
    ("moderate", 10, 30),     # 10 to 29
    ("dense", 30, 10**9),     # 30 and up
)

# Below this, a stratum's estimate is arithmetic on too little to mean anything. It is still computed and
# stored, because refusing to store it would leave the caller unable to tell "thin" from "not run", but the
# support count travels with it so a reader can see what it rests on.
MIN_STRATUM_FRAMES = 10


def _stratum_of(n_pred: int, stratify_by: str) -> str:
    if stratify_by != "density":
        return "all"
    for name, lo, hi in _DENSITY_BOUNDS:
        if lo <= n_pred < hi:
            return name
    return "dense"


async def seed_audit(db: AsyncSession, *, run_id: str, n_frames: int = 200,
                     stratify_by: str = "density", score_thr: float = 0.25, iou_thr: float = 0.5,
                     project_id: str | None = None, notes: str | None = None) -> dict:
    """Choose the frames, create the audit and the annotation job that serves them blind.

    Frames are drawn from the ones the run actually scored, so the model's observation exists for every
    frame in the sample. When the run scored a gold set, that is the gold frames, which is what makes the
    resulting estimate directly comparable to the gold recall for the same run.

    Sampling is even across strata rather than proportional to how common each stratum is. A proportional
    sample of a corpus that is mostly empty road spends the audit budget where the model is already right;
    the strata that carry the risk are the crowded ones, and they need enough frames to be measurable at
    all. Stratified estimation corrects for the uneven sampling by construction, which is the other reason
    to stratify.
    """
    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"error": "inference run not found", "run_id": run_id}
    if run.status != "complete":
        return {"error": f"run status is '{run.status}'; a partial run is not an observer", "run_id": run_id}
    if n_frames < 1:
        return {"error": "n_frames must be at least 1"}

    # Prediction count per frame at the operating point. This is the stratification key and it is entirely
    # model-side, so choosing frames leaks nothing about what a human would find.
    counts = (await db.execute(
        select(Prediction.frame_id, func.count(Prediction.prediction_id))
        .where(Prediction.run_id == run.run_id, Prediction.conf >= score_thr)
        .group_by(Prediction.frame_id))).all()
    counted = {fid: int(n) for fid, n in counts}

    # Frames the run scored but left empty at this threshold still belong in the sample. Dropping them would
    # define the audit over the frames the model already fired on, which is the same selection bias in a
    # different place: a frame the model saw nothing in is exactly where a human-only object is most likely.
    scored = (await db.execute(
        select(Prediction.frame_id).where(Prediction.run_id == run.run_id).distinct())).scalars().all()
    for fid in scored:
        counted.setdefault(fid, 0)
    if not counted:
        return {"error": "the run has no predictions, so there is nothing to audit against",
                "run_id": run_id}

    by_stratum: dict[str, list] = {}
    for fid, n in counted.items():
        by_stratum.setdefault(_stratum_of(n, stratify_by), []).append(fid)
    # Deterministic order, so re-seeding the same run with the same size picks the same frames and two
    # audits of one run can be compared rather than merely both existing.
    for v in by_stratum.values():
        v.sort(key=str)

    names = sorted(by_stratum)
    # Water-filling: hand the next frame to the least-served stratum that still has one, until the budget
    # is spent or every stratum is exhausted. An even split when the strata are all large enough, and a
    # stratum too small for its share spills its shortfall onto the others rather than shrinking the audit.
    # Ties break on the stratum name so the same run and budget always select the same frames.
    take = dict.fromkeys(names, 0)
    for _ in range(n_frames):
        candidates = [n for n in names if take[n] < len(by_stratum[n])]
        if not candidates:
            break
        take[min(candidates, key=lambda k: (take[k], k))] += 1
    chosen = [(fid, name) for name in names for fid in by_stratum[name][:take[name]]]

    if not chosen:
        return {"error": "no frames available to audit", "run_id": run_id}

    strata_summary = {n: {"available": len(by_stratum[n]),
                          "sampled": sum(1 for _, s in chosen if s == n)} for n in names}
    audit = BlindAudit(run_id=run.run_id, gold_id=run.gold_id, n_frames=len(chosen),
                       stratify_by=stratify_by, strata=strata_summary, score_thr=float(score_thr),
                       iou_thr=float(iou_thr), status="seeded", notes=notes)
    db.add(audit)
    await db.flush()

    db.add_all([BlindAuditFrame(audit_id=audit.audit_id, frame_id=fid, stratum=stratum)
                for fid, stratum in chosen])

    job_id = await _create_audit_job(db, audit, [fid for fid, _ in chosen], project_id)
    audit.job_id = job_id
    await db.commit()

    log.info("blind_audit.seeded", audit=str(audit.audit_id), run=run_id, gold=run.gold_id,
             frames=len(chosen), strata=strata_summary, job=str(job_id) if job_id else None)
    return {"audit_id": str(audit.audit_id), "run_id": run_id, "gold_id": run.gold_id,
            "n_frames": len(chosen), "strata": strata_summary, "score_thr": score_thr,
            "iou_thr": iou_thr, "job_id": str(job_id) if job_id else None, "status": "seeded"}


async def _create_audit_job(db: AsyncSession, audit: BlindAudit, frame_ids: list[UUID],
                            project_id: str | None) -> UUID | None:
    """The annotation job that serves the audit frames. None when there is no project to hang it on.

    No honeypots are seeded. A honeypot is a gold frame mixed in to measure the annotator, and a gold frame
    arrives with its labels already drawn; on an audit that is the exact leak this whole design exists to
    prevent.
    """
    if project_id:
        project = await db.get(LabelProject, UUID(project_id))
    else:
        project = (await db.execute(
            select(LabelProject).order_by(LabelProject.created_at).limit(1))).scalars().first()
    if project is None:
        log.warning("blind_audit.no_project", audit=str(audit.audit_id))
        return None

    task = LabelTask(project_id=project.project_id,
                     name=f"blind audit {str(audit.audit_id)[:8]}",
                     predicate={"blind_audit_id": str(audit.audit_id)}, replicas=1)
    db.add(task)
    await db.flush()
    job = LabelJob(task_id=task.task_id, frame_ids=[str(f) for f in frame_ids],
                   stage="annotation", state="new")
    db.add(job)
    await db.flush()
    return job.job_id


async def audit_for_job(db: AsyncSession, job_id: UUID) -> BlindAudit | None:
    """The audit a job serves, or None. This is what the frame fetch handler asks before returning anything."""
    return (await db.execute(
        select(BlindAudit).where(BlindAudit.job_id == job_id))).scalars().first()


async def active_audit_id(db: AsyncSession, *, frame_id: UUID, job_id: UUID | None,
                          user_id: UUID | None) -> UUID | None:
    """The audit this request is acting under, or None. ONE rule, used by both the read and the write path.

    Read and write must agree exactly. If the fetch hid the predictions but the create failed to stamp the
    box (because the annotator's editor did not pass a job_id, say), the audit would collect no human
    observations and score as the human having found nothing, which reports the model's recall as perfect
    precisely where it was actually being tested. A single function is the only way to keep the two sides
    from drifting apart.

    Resolution order: the job named by the request, then the audit this user is assigned to on this frame.
    The second clause is what makes dropping the job_id useless as a way to see the answers, and equally
    what makes it harmless: the boxes still get stamped.

    Only an audit still being labelled counts. Once scored, the frames go back to behaving like any other
    frame, because the measurement has already been taken.
    """
    if job_id is not None:
        audit = (await db.execute(
            select(BlindAudit).where(BlindAudit.job_id == job_id,
                                     BlindAudit.status.in_(("seeded", "labeling"))))).scalars().first()
        if audit is not None:
            return audit.audit_id
    if user_id is None:
        return None
    return (await db.execute(
        select(BlindAudit.audit_id)
        .join(BlindAuditFrame, BlindAuditFrame.audit_id == BlindAudit.audit_id)
        .join(LabelJob, LabelJob.job_id == BlindAudit.job_id)
        .where(BlindAuditFrame.frame_id == frame_id,
               BlindAudit.status.in_(("seeded", "labeling")),
               LabelJob.assignee_id == user_id)
        .limit(1))).scalars().first()


async def mark_frames_labeled(db: AsyncSession, audit_id: UUID,
                              frame_ids: list[UUID] | None = None) -> int:
    """Record that the auditor is finished with these frames (all of them when frame_ids is None).

    This is what makes an audit scoreable, and it cannot be inferred from whether the auditor drew anything.
    A frame with no audit boxes is either a frame with nothing in it or a frame nobody opened, and scoring
    the second as the first reports the model as having missed nothing precisely where it was never checked.
    """
    q = select(BlindAuditFrame).where(BlindAuditFrame.audit_id == audit_id,
                                      BlindAuditFrame.labeled_at.is_(None))
    if frame_ids is not None:
        q = q.where(BlindAuditFrame.frame_id.in_(frame_ids))
    rows = (await db.execute(q)).scalars().all()
    now = datetime.now(UTC)
    for r in rows:
        r.labeled_at = now
    if rows:
        audit = await db.get(BlindAudit, audit_id)
        if audit is not None and audit.status == "seeded":
            audit.status = "labeling"
        await db.commit()
    return len(rows)


async def _gold_recall(db: AsyncSession, run_id: UUID, gold_id: str | None) -> tuple[float | None, dict]:
    """Recall against the sealed gold denominator, for the same run: the number this is measured against.

    Read from EvalPatch rather than recomputed, so it is the same number the Quality page shows and not a
    second opinion that happens to be close. Two conventions, matching the two matching rules above:

      class-agnostic  a gold object counts as found if any prediction matched it, whatever class it claimed
      class-aware     it counts as found only when the classes agree

    EvalPatch writes `fn` only for a gold object no prediction matched at all, and records a cross-class
    match as an `fp` carrying both class ids. So the agnostic denominator is (matched gold) + (fn), and the
    aware numerator is the `tp` rows.
    """
    if not gold_id:
        return None, {}
    rows = (await db.execute(
        select(EvalPatch.outcome, EvalPatch.gt_class_id, func.count(EvalPatch.patch_id))
        .where(EvalPatch.run_id == run_id, EvalPatch.gold_id == gold_id)
        .group_by(EvalPatch.outcome, EvalPatch.gt_class_id))).all()
    if not rows:
        return None, {}

    tp: dict[int, int] = {}
    cross: dict[int, int] = {}   # matched, classes disagreed
    fn: dict[int, int] = {}
    for outcome, gcid, n in rows:
        if gcid is None:
            continue             # a prediction on empty ground; no gold object behind it
        n = int(n)
        if outcome == "tp":
            tp[gcid] = tp.get(gcid, 0) + n
        elif outcome == "fp":
            cross[gcid] = cross.get(gcid, 0) + n
        elif outcome == "fn":
            fn[gcid] = fn.get(gcid, 0) + n

    classes = set(tp) | set(cross) | set(fn)
    per_class = {}
    for c in classes:
        total = tp.get(c, 0) + cross.get(c, 0) + fn.get(c, 0)
        per_class[c] = round(tp.get(c, 0) / total, 6) if total else None
    total_all = sum(tp.values()) + sum(cross.values()) + sum(fn.values())
    agnostic = ((sum(tp.values()) + sum(cross.values())) / total_all) if total_all else None
    return (round(agnostic, 6) if agnostic is not None else None), per_class


async def score_audit(db: AsyncSession, audit_id: str, *, min_frames: int = 1) -> dict:
    """Match the two observations frame by frame, estimate the population, and persist every slice.

    Only frames the auditor finished are scored. An unlabelled frame is not a frame where the human found
    nothing, and counting it as one would make the model's recall look better the less of the audit was
    done, which is the exact failure mode this is meant to detect.
    """
    audit = await db.get(BlindAudit, UUID(audit_id))
    if audit is None:
        return {"error": "audit not found", "audit_id": audit_id}

    af_rows = (await db.execute(
        select(BlindAuditFrame).where(BlindAuditFrame.audit_id == audit.audit_id))).scalars().all()
    done = [r for r in af_rows if r.labeled_at is not None]
    if len(done) < max(1, min_frames):
        return {"error": f"only {len(done)} of {len(af_rows)} audit frames are labelled; "
                         "an unlabelled frame is not a frame where the human found nothing",
                "audit_id": audit_id, "n_labeled": len(done), "n_frames": len(af_rows),
                "status": audit.status}

    frame_ids = [r.frame_id for r in done]
    stratum_of = {r.frame_id: r.stratum for r in done}

    preds = (await db.execute(
        select(Prediction.frame_id, Prediction.class_id, Prediction.bbox, Prediction.conf)
        .where(Prediction.run_id == audit.run_id, Prediction.frame_id.in_(frame_ids),
               Prediction.conf >= audit.score_thr))).all()
    by_frame_pred: dict[uuidlib.UUID, list] = {}
    for fid, cid, bbox, conf in preds:
        by_frame_pred.setdefault(fid, []).append((int(cid), list(bbox), float(conf or 0.0)))

    humans = (await db.execute(
        select(Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.blind_audit_id == audit.audit_id, Object.frame_id.in_(frame_ids)))).all()
    by_frame_human: dict[uuidlib.UUID, list] = {}
    for fid, cid, bbox in humans:
        by_frame_human.setdefault(fid, []).append((int(cid), list(bbox)))

    # (both, model_only, human_only), class-agnostic, per stratum and per frame.
    per_stratum: dict[str, list[int]] = {}
    # Class-aware, per class: an object the model found under the wrong name is a miss for that class.
    per_class: dict[int, list[int]] = {}
    frame_counts: dict[uuidlib.UUID, tuple[int, int, int]] = {}

    for fid in frame_ids:
        p = by_frame_pred.get(fid, [])
        h = by_frame_human.get(fid, [])
        p_boxes = np.asarray([x[1] for x in p], dtype=float).reshape(-1, 4)
        p_scores = np.asarray([x[2] for x in p], dtype=float).reshape(-1)
        h_boxes = np.asarray([x[1] for x in h], dtype=float).reshape(-1, 4)

        match_h, _iou, h_matched = _greedy_match(p_boxes, p_scores, h_boxes, audit.iou_thr)
        both = int(h_matched.sum())
        model_only = len(p) - both
        human_only = len(h) - both
        frame_counts[fid] = (both, model_only, human_only)

        acc = per_stratum.setdefault(stratum_of[fid], [0, 0, 0])
        acc[0] += both
        acc[1] += model_only
        acc[2] += human_only

        for i in range(len(p)):
            j = int(match_h[i])
            p_class = p[i][0]
            if j < 0:
                per_class.setdefault(p_class, [0, 0, 0])[1] += 1        # model claims it, no human box
                continue
            h_class = h[j][0]
            if h_class == p_class:
                per_class.setdefault(p_class, [0, 0, 0])[0] += 1        # both, and they agree
            else:
                # One box, two names. For the model's class it is an unconfirmed claim; for the human's it
                # is an object of that class the model did not find under that name.
                per_class.setdefault(p_class, [0, 0, 0])[1] += 1
                per_class.setdefault(h_class, [0, 0, 0])[2] += 1
        for j in range(len(h)):
            if not h_matched[j]:
                per_class.setdefault(h[j][0], [0, 0, 0])[2] += 1

    for r in done:
        b, m, hh = frame_counts[r.frame_id]
        r.n_both, r.n_model_only, r.n_human_only = b, m, hh

    names = sorted(per_stratum)
    strat = stratified_recapture([per_stratum[n] for n in names], labels=names)
    gold_agnostic, gold_per_class = await _gold_recall(db, audit.run_id, audit.gold_id)

    # Recompute from scratch: scoring is a pure function of the labels present, so a rescore after more
    # frames were labelled replaces the estimate rather than accumulating a second one beside it.
    await db.execute(delete(RecaptureEstimateRow).where(
        RecaptureEstimateRow.audit_id == audit.audit_id))

    rows: list[RecaptureEstimateRow] = []
    for entry in strat["per_stratum"]:
        rows.append(_row(audit, stratum=entry["stratum"], class_id=None, est=entry,
                         gold_recall=gold_agnostic))
    if strat["pooled"] is not None:
        rows.append(_row(audit, stratum=None, class_id=None, est=strat["pooled"],
                         gold_recall=gold_agnostic))
    for cid, (b, m, hh) in sorted(per_class.items()):
        est = lincoln_petersen(n_both=b, n_model_only=m, n_human_only=hh)
        rows.append(_row(audit, stratum=None, class_id=cid, est=est.__dict__,
                         gold_recall=gold_per_class.get(cid)))
    db.add_all(rows)

    audit.status = "scored"
    audit.scored_at = datetime.now(UTC)
    await db.commit()

    pooled = strat["pooled"]
    log.info("blind_audit.scored", audit=audit_id, run=str(audit.run_id), gold=audit.gold_id,
             frames_scored=len(done), measured=strat["measured"],
             population=(pooled or {}).get("population"),
             model_recall=(pooled or {}).get("model_recall"), gold_recall=gold_agnostic,
             unmeasured_strata=strat["unmeasured"])
    return {
        "audit_id": audit_id, "run_id": str(audit.run_id), "gold_id": audit.gold_id,
        "n_frames": len(af_rows), "n_labeled": len(done),
        "measured": strat["measured"], "reason": strat.get("reason"),
        "pooled": pooled, "per_stratum": strat["per_stratum"], "unmeasured": strat["unmeasured"],
        "per_class": {str(c): v for c, v in sorted(per_class.items())},
        "gold_recall": gold_agnostic,
        # The whole point, stated rather than left to the reader to subtract.
        "recall_overstatement": (round(gold_agnostic - pooled["model_recall"], 6)
                                 if pooled and gold_agnostic is not None else None),
        "caveat": ("positively correlated captures bias the population down, so this is a lower bound on "
                   "what was missed and an upper bound on recall"),
        "estimator": ESTIMATOR,
    }


def _row(audit: BlindAudit, *, stratum: str | None, class_id: int | None, est: dict,
         gold_recall: float | None) -> RecaptureEstimateRow:
    """One persisted slice. `est` is a RecaptureEstimate as a dict or a stratified_recapture entry."""
    return RecaptureEstimateRow(
        audit_id=audit.audit_id, run_id=audit.run_id, gold_id=audit.gold_id,
        stratum=stratum, class_id=class_id,
        n_both=int(est.get("n_both") or 0), n_model_only=int(est.get("n_model_only") or 0),
        n_human_only=int(est.get("n_human_only") or 0),
        measured=bool(est.get("measured")), reason=est.get("reason"),
        population=est.get("population"), population_lo=est.get("lo"), population_hi=est.get("hi"),
        variance=est.get("variance"),
        model_recall=est.get("model_recall"), recall_lo=est.get("recall_lo"),
        recall_hi=est.get("recall_hi"), human_recall=est.get("human_recall"),
        gold_recall=gold_recall, estimator=ESTIMATOR,
        n_strata_pooled=est.get("n_strata_pooled"))


async def audit_progress(db: AsyncSession, audit_id: str) -> dict:
    audit = await db.get(BlindAudit, UUID(audit_id))
    if audit is None:
        return {"error": "audit not found", "audit_id": audit_id}
    n_total, n_done = (await db.execute(
        select(func.count(BlindAuditFrame.audit_frame_id),
               func.count(BlindAuditFrame.labeled_at))
        .where(BlindAuditFrame.audit_id == audit.audit_id))).one()
    n_boxes = (await db.execute(select(func.count(Object.object_id)).where(
        Object.blind_audit_id == audit.audit_id))).scalar_one()
    return {"audit_id": audit_id, "run_id": str(audit.run_id), "gold_id": audit.gold_id,
            "job_id": str(audit.job_id) if audit.job_id else None, "status": audit.status,
            "n_frames": int(n_total), "n_labeled": int(n_done), "n_audit_objects": int(n_boxes),
            "strata": audit.strata, "score_thr": audit.score_thr, "iou_thr": audit.iou_thr,
            "scored_at": audit.scored_at.isoformat() if audit.scored_at else None}


async def pooled_estimate(db: AsyncSession, *, run_id: str,
                          gold_id: str | None = None) -> dict | None:
    """The corpus-wide estimate for a run, or None when no scored audit exists for it.

    This is what the champion gate reads. None and "measured false" are different answers and the gate
    treats them differently: the first means nobody checked, the second means somebody checked and the
    check could not conclude.
    """
    q = (select(RecaptureEstimateRow)
         .where(RecaptureEstimateRow.run_id == UUID(run_id),
                RecaptureEstimateRow.stratum.is_(None), RecaptureEstimateRow.class_id.is_(None))
         .order_by(RecaptureEstimateRow.created_at.desc()).limit(1))
    if gold_id:
        q = q.where(RecaptureEstimateRow.gold_id == gold_id)
    row = (await db.execute(q)).scalars().first()
    if row is None:
        return None
    return {"estimate_id": str(row.estimate_id), "audit_id": str(row.audit_id),
            "run_id": str(row.run_id), "gold_id": row.gold_id,
            "measured": row.measured, "reason": row.reason,
            "population": row.population, "population_lo": row.population_lo,
            "population_hi": row.population_hi,
            "model_recall": row.model_recall, "recall_lo": row.recall_lo, "recall_hi": row.recall_hi,
            "human_recall": row.human_recall, "gold_recall": row.gold_recall,
            "n_both": row.n_both, "n_model_only": row.n_model_only, "n_human_only": row.n_human_only,
            "estimator": row.estimator,
            "created_at": row.created_at.isoformat() if row.created_at else None}


async def list_audits(db: AsyncSession, *, run_id: str | None = None, limit: int = 50) -> list[dict]:
    q = select(BlindAudit).order_by(BlindAudit.created_at.desc()).limit(max(1, min(limit, 200)))
    if run_id:
        q = q.where(BlindAudit.run_id == UUID(run_id))
    audits = (await db.execute(q)).scalars().all()
    out = []
    for a in audits:
        n_total, n_done = (await db.execute(
            select(func.count(BlindAuditFrame.audit_frame_id), func.count(BlindAuditFrame.labeled_at))
            .where(BlindAuditFrame.audit_id == a.audit_id))).one()
        out.append({"audit_id": str(a.audit_id), "run_id": str(a.run_id), "gold_id": a.gold_id,
                    "job_id": str(a.job_id) if a.job_id else None, "status": a.status,
                    "n_frames": int(n_total), "n_labeled": int(n_done),
                    "created_at": a.created_at.isoformat() if a.created_at else None})
    return out


async def audit_frame_ids(db: AsyncSession, audit_id: UUID) -> list[UUID]:
    return list((await db.execute(select(BlindAuditFrame.frame_id).where(
        BlindAuditFrame.audit_id == audit_id))).scalars().all())


__all__ = ["seed_audit", "score_audit", "audit_progress", "pooled_estimate", "list_audits",
           "mark_frames_labeled", "audit_for_job", "active_audit_id", "audit_frame_ids", "ESTIMATOR"]
