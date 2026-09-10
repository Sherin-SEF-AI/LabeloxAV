"""Comparing two models on the same frames, and turning where they part company into a worklist.

Gold is frozen the day it is sealed, so it can only ever say how a model does on the frames somebody
chose months ago. The corpus keeps ingesting sessions nobody compares any model on, and the frames worth
a person's attention are precisely the ones two models read differently: agreement teaches nothing,
whether both are right or both are wrong.

The matcher is deliberately small and pure. It takes two runs' predictions on one frame and returns the
disagreements. Everything about scheduling, GPU slots and who gets asked lives in `govern/shadow_agent`.

Two decisions carry the reasoning:

* **Both models are cut at the champion's operating point.** Inference writes down to a 0.001 floor so a
  PR curve can be drawn later, and comparing raw floors would make every low-confidence tail of the
  challenger a "champion miss". The comparison a promotion decision needs is between models as they would
  actually be run, which is at the threshold the champion serves at.
* **A miss is only a miss above the threshold.** A box the other model found at 0.2 when the cut is 0.5
  is not a disagreement about what is there, it is agreement with different confidence, and that is what
  `conf_gap` is for.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import InferenceRun, Prediction, ShadowDisagreement

log = get_logger("shadow_run")

CHAMPION_MISS = "champion_miss"
CHALLENGER_MISS = "challenger_miss"
CLASS_FLIP = "class_flip"
CONF_GAP = "conf_gap"

# Boxes this close are the same box. The same value the blind audit and the recall harness pair at, so a
# disagreement here means the same thing as a miss there.
MATCH_IOU = 0.5
# Below this the two models agree about the object and differ only in how sure they are, which is not
# worth a person's minute on its own. Above it, one of them is close to the cut and the other is not.
CONF_GAP_MIN = 0.25


@dataclass(frozen=True)
class Det:
    """One prediction, reduced to what the comparison needs."""

    prediction_id: uuid.UUID
    class_id: int
    bbox: list[float]
    conf: float


@dataclass(frozen=True)
class Disagreement:
    kind: str
    bbox: list[float]
    score: float
    champion: Det | None = None
    challenger: Det | None = None
    iou: float | None = None
    conf_gap: float | None = None


def _iou(a: list[float], b: list[float]) -> float:
    from core.accel.boxes import box_iou_matrix

    return float(box_iou_matrix([a], [b])[0][0])


def _pair(champion: list[Det], challenger: list[Det], iou_thr: float) -> tuple[list[tuple[int, int, float]],
                                                                              set[int], set[int]]:
    """Greedy highest-IoU pairing, class-agnostic so a class flip pairs instead of reading as two misses.

    Returns the matched (champion index, challenger index, iou) triples and the unmatched indices on each
    side. Greedy by IoU rather than by confidence: the question is which boxes are the same box, and
    confidence is the thing being compared, so letting it decide the pairing would beg the question.
    """
    from core.accel.boxes import box_iou_matrix

    if not champion or not challenger:
        return [], set(range(len(champion))), set(range(len(challenger)))
    m = box_iou_matrix([d.bbox for d in champion], [d.bbox for d in challenger])
    order = sorted(((float(m[i][j]), i, j) for i in range(len(champion)) for j in range(len(challenger))
                    if float(m[i][j]) >= iou_thr), key=lambda t: -t[0])
    used_c: set[int] = set()
    used_x: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for iou, i, j in order:
        if i in used_c or j in used_x:
            continue
        used_c.add(i)
        used_x.add(j)
        pairs.append((i, j, iou))
    return pairs, set(range(len(champion))) - used_c, set(range(len(challenger))) - used_x


def compare_frame(champion: list[Det], challenger: list[Det], *, threshold: float,
                  iou_thr: float = MATCH_IOU, conf_gap_min: float = CONF_GAP_MIN,
                  shared_classes: set[int] | None = None) -> list[Disagreement]:
    """Every place two models' detections on one frame disagree, at the champion's operating point.

    Pure: no database, no model, no clock. The scores are what ranks a worklist, and each kind earns its
    score differently. A miss is scored by how confident the model that found it was, because a confident
    detection the other model has nothing for is the strongest evidence either way. A class flip is scored
    by the lower of the two confidences, because the interesting flips are the ones where both models are
    sure and they are sure of different things. A confidence gap is scored by the gap itself.

    `shared_classes` is the set of ontology ids both models can emit. A detection of a class the other
    model was never trained on is dropped rather than counted, because that is a difference in vocabulary
    and not a disagreement about what is in the picture: the first real sweep on this corpus compared a
    12-class champion with a 9-class challenger and 95% of the result was the three missing classes. None
    means the vocabularies are unknown and every class is compared, which is the older behaviour.
    """
    kept_c = [d for d in champion if d.conf >= threshold]
    kept_x = [d for d in challenger if d.conf >= threshold]
    if shared_classes is not None:
        kept_c = [d for d in kept_c if d.class_id in shared_classes]
        kept_x = [d for d in kept_x if d.class_id in shared_classes]
    pairs, lone_c, lone_x = _pair(kept_c, kept_x, iou_thr)
    out: list[Disagreement] = []

    for i, j, iou in pairs:
        c, x = kept_c[i], kept_x[j]
        if c.class_id != x.class_id:
            out.append(Disagreement(kind=CLASS_FLIP, bbox=c.bbox, score=min(c.conf, x.conf),
                                    champion=c, challenger=x, iou=iou,
                                    conf_gap=abs(c.conf - x.conf)))
            continue
        gap = abs(c.conf - x.conf)
        if gap >= conf_gap_min:
            out.append(Disagreement(kind=CONF_GAP, bbox=c.bbox, score=gap, champion=c, challenger=x,
                                    iou=iou, conf_gap=gap))

    # A box only the challenger has above the cut is something the champion missed, and the other way
    # round. The name is from the point of view of the model that has nothing there.
    for j in sorted(lone_x):
        x = kept_x[j]
        out.append(Disagreement(kind=CHAMPION_MISS, bbox=x.bbox, score=x.conf, challenger=x))
    for i in sorted(lone_c):
        c = kept_c[i]
        out.append(Disagreement(kind=CHALLENGER_MISS, bbox=c.bbox, score=c.conf, champion=c))

    out.sort(key=lambda d: -d.score)
    return out


async def _threshold_for(db: AsyncSession, model_version: str, default: float) -> float:
    """The champion's serving threshold, or the supplied default when nothing was ever fitted.

    An unfitted threshold is not zero and not the ship default pretending to be measured: the caller
    passes what it wants used and this says whether a measured one exists.
    """
    from db.models import ThresholdFit

    row = (await db.execute(
        select(func.min(ThresholdFit.threshold)).where(
            ThresholdFit.active.is_(True), ThresholdFit.measured.is_(True),
            ThresholdFit.model_version == model_version,
            ThresholdFit.threshold.isnot(None)))).scalar_one_or_none()
    return float(row) if row is not None else float(default)


async def _dets_by_frame(db: AsyncSession, run_id: uuid.UUID,
                         frame_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[Det]]:
    rows = (await db.execute(
        select(Prediction.prediction_id, Prediction.frame_id, Prediction.class_id, Prediction.bbox,
               Prediction.conf)
        .where(Prediction.run_id == run_id, Prediction.frame_id.in_(frame_ids)))).all()
    out: dict[uuid.UUID, list[Det]] = {}
    for pid, fid, cid, bbox, conf in rows:
        if conf is None:
            # A reconstructed run has no score, so it cannot be cut at an operating point and cannot be
            # compared to one that can. Skipping is the honest answer; a substituted 1.0 would not be.
            continue
        out.setdefault(fid, []).append(Det(prediction_id=pid, class_id=int(cid),
                                           bbox=[float(v) for v in bbox], conf=float(conf)))
    return out


async def match_predictions(db: AsyncSession, *, sweep_run_id: uuid.UUID, champion_run: str,
                            challenger_run: str, threshold: float = 0.5,
                            iou_thr: float = MATCH_IOU, chunk: int = 200) -> dict:
    """Write a `ShadowDisagreement` row for every disagreement between two complete inference runs.

    Frames are read in chunks so a sweep over thousands never holds every prediction in memory at once,
    and each chunk commits, so an interrupted match leaves the rows it already found rather than nothing.
    """
    c_run = await db.get(InferenceRun, uuid.UUID(champion_run))
    x_run = await db.get(InferenceRun, uuid.UUID(challenger_run))
    for name, run in (("champion", c_run), ("challenger", x_run)):
        if run is None:
            return {"error": f"{name} inference run not found"}
        if run.status != "complete":
            return {"error": f"{name} run status is '{run.status}'; a partial run is not a comparison"}

    # The classes both models could have named. A run that never declared a vocabulary (recorded before
    # 0111) leaves this None, and the comparison then covers every class as it used to.
    shared: set[int] | None = None
    if c_run.class_vocab is not None and x_run.class_vocab is not None:
        shared = {int(i) for i in c_run.class_vocab} & {int(i) for i in x_run.class_vocab}

    frames = sorted({f for f in (await db.execute(
        select(Prediction.frame_id).where(
            Prediction.run_id.in_((c_run.run_id, x_run.run_id))).distinct())).scalars().all()},
        key=str)

    counts: dict[str, int] = {CHAMPION_MISS: 0, CHALLENGER_MISS: 0, CLASS_FLIP: 0, CONF_GAP: 0}
    written = 0
    for i in range(0, len(frames), chunk):
        window = frames[i:i + chunk]
        by_c = await _dets_by_frame(db, c_run.run_id, window)
        by_x = await _dets_by_frame(db, x_run.run_id, window)
        for fid in window:
            for d in compare_frame(by_c.get(fid, []), by_x.get(fid, []), threshold=threshold,
                                   iou_thr=iou_thr, shared_classes=shared):
                db.add(ShadowDisagreement(
                    sweep_run_id=sweep_run_id, frame_id=fid,
                    champion_run_id=c_run.run_id, challenger_run_id=x_run.run_id,
                    champion_prediction_id=d.champion.prediction_id if d.champion else None,
                    challenger_prediction_id=d.challenger.prediction_id if d.challenger else None,
                    kind=d.kind,
                    champion_class_id=d.champion.class_id if d.champion else None,
                    challenger_class_id=d.challenger.class_id if d.challenger else None,
                    iou=d.iou, conf_gap=d.conf_gap, score=d.score, bbox=d.bbox, state="pending"))
                counts[d.kind] += 1
                written += 1
        await db.commit()

    log.info("shadow.matched", sweep_run_id=str(sweep_run_id), frames=len(frames), written=written,
             shared_classes=(len(shared) if shared is not None else None),
             **{k: v for k, v in counts.items()})
    return {"frames": len(frames), "written": written, "by_kind": counts, "threshold": threshold,
            "shared_classes": (sorted(shared) if shared is not None else None),
            "champion_only_classes": (sorted(set(c_run.class_vocab or []) - shared)
                                      if shared is not None else None),
            "challenger_only_classes": (sorted(set(x_run.class_vocab or []) - shared)
                                        if shared is not None else None)}


async def win_share(db: AsyncSession, challenger_run_id: uuid.UUID) -> dict:
    """How often the challenger was right where the two models disagreed, with its interval.

    Only the discordant pairs count: a disagreement both models got wrong, or both got right, says
    nothing about which is better, and folding those in would drag every share toward the middle and make
    a decisive challenger look marginal. Returns `measured: False` with a reason rather than a share when
    nobody has adjudicated enough of them, because an unmeasured share is not 0.5.
    """
    from services.labelops.sampling import wilson_interval

    rows = (await db.execute(
        select(ShadowDisagreement.verdict, func.count())
        .where(ShadowDisagreement.challenger_run_id == challenger_run_id,
               ShadowDisagreement.verdict.isnot(None))
        .group_by(ShadowDisagreement.verdict))).all()
    by = {k: int(v) for k, v in rows}
    champ = by.get("champion_right", 0)
    chall = by.get("challenger_right", 0)
    discordant = champ + chall
    out = {"n_adjudicated": sum(by.values()), "champion_right": champ, "challenger_right": chall,
           "both_right": by.get("both_right", 0), "both_wrong": by.get("both_wrong", 0),
           "discordant": discordant}
    if discordant == 0:
        return {**out, "measured": False,
                "reason": "no disagreement has been adjudicated in the challenger's favour or against it"}
    ci = wilson_interval(chall, discordant)
    return {**out, "measured": True, "share": ci["p"], "lo": ci["lo"], "hi": ci["hi"]}


async def adjudicate_for_job(db: AsyncSession, job_id: uuid.UUID) -> dict:
    """Turn the human labels on a job's frames into verdicts on that job's disagreements.

    A disagreement is a claim about one box. The person who labelled the frame did not answer it directly,
    they drew what is actually there, so the verdict is read off their work: the disagreement's box is
    matched against the frame's human objects, and whoever agreed with the person won.
    """
    from db.models import LabelJob, Object

    job = await db.get(LabelJob, job_id)
    if job is None:
        return {"error": "job not found"}
    pend = (await db.execute(
        select(ShadowDisagreement).where(ShadowDisagreement.task_id == job.task_id,
                                         ShadowDisagreement.state.in_(("pending", "queued"))))).scalars().all()
    if not pend:
        return {"adjudicated": 0, "reason": "no pending disagreement is attached to this job's task"}

    frame_ids = sorted({d.frame_id for d in pend}, key=str)
    humans = (await db.execute(
        select(Object).where(Object.frame_id.in_(frame_ids), Object.source == "human"))).scalars().all()
    by_frame: dict[uuid.UUID, list[Object]] = {}
    for o in humans:
        by_frame.setdefault(o.frame_id, []).append(o)

    now = datetime.now(UTC)
    n = 0
    for d in pend:
        best, best_iou = None, 0.0
        for o in by_frame.get(d.frame_id, []):
            v = _iou(d.bbox, [float(x) for x in o.bbox])
            if v > best_iou:
                best, best_iou = o, v
        if best_iou < MATCH_IOU or best is None:
            # Nothing is there. Whichever model claimed a box was wrong, and the model that claimed
            # nothing was right. The kind is named for the model that has nothing, so a `champion_miss`
            # with no object present means the challenger invented one and the champion was right to
            # have missed it. A flip or a gap has a box from both, so both were wrong.
            d.verdict = ("champion_right" if d.kind == CHAMPION_MISS
                         else "challenger_right" if d.kind == CHALLENGER_MISS
                         else "both_wrong")
            d.verdict_object_id = None
        else:
            c_ok = d.champion_class_id is not None and d.champion_class_id == best.class_id
            x_ok = d.challenger_class_id is not None and d.challenger_class_id == best.class_id
            if d.kind == CHAMPION_MISS:
                # The champion had nothing here. The challenger is right if what it drew is what is there.
                d.verdict = "challenger_right" if x_ok else "both_wrong"
            elif d.kind == CHALLENGER_MISS:
                d.verdict = "champion_right" if c_ok else "both_wrong"
            elif c_ok and x_ok:
                d.verdict = "both_right"
            elif c_ok:
                d.verdict = "champion_right"
            elif x_ok:
                d.verdict = "challenger_right"
            else:
                d.verdict = "both_wrong"
            d.verdict_object_id = best.object_id
        d.state = "adjudicated"
        d.adjudicated_at = now
        n += 1
    await db.commit()
    log.info("shadow.adjudicated", job_id=str(job_id), adjudicated=n)
    return {"adjudicated": n, "frames": len(frame_ids)}


# How many disagreement frames one sweep puts in front of a person. The sweep can file thousands of rows;
# the worklist is bounded by what a night's labelling can actually absorb, and the rest wait in `pending`
# for a later sweep to promote them. Ranked by the worst disagreement on each frame, so the frames that
# earn a minute come first.
QUEUE_FRAMES = 40


async def queue_for_labelling(db: AsyncSession, *, sweep_run_id: uuid.UUID,
                              project_id: str | None = None, n_frames: int = QUEUE_FRAMES) -> dict:
    """File one labelling task over the worst disagreement frames of a sweep, and mark those rows queued.

    A frame is the unit, not a disagreement, because a `champion_miss` has no object for a person to rule
    on: the answer is drawn, not voted. Everything pending on a queued frame is queued with it, so one
    person opening one frame settles every disagreement on it at once.
    """
    from db.models import LabelProject

    rows = (await db.execute(
        select(ShadowDisagreement.frame_id, func.max(ShadowDisagreement.score))
        .where(ShadowDisagreement.sweep_run_id == sweep_run_id,
               ShadowDisagreement.state == "pending")
        .group_by(ShadowDisagreement.frame_id)
        .order_by(func.max(ShadowDisagreement.score).desc()).limit(n_frames))).all()
    if not rows:
        return {"queued": 0, "reason": "the sweep found no pending disagreement to queue"}
    frame_ids = [r[0] for r in rows]

    if project_id:
        project = await db.get(LabelProject, uuid.UUID(project_id))
    else:
        project = (await db.execute(
            select(LabelProject).order_by(LabelProject.created_at).limit(1))).scalars().first()
    if project is None:
        return {"queued": 0, "reason": "no labelling project exists to hang the task on"}

    from services.labelops.jobs import create_task

    task = await create_task(db, project_id=str(project.project_id),
                             name=f"shadow disagreements {str(sweep_run_id)[:8]}",
                             predicate={"frame_ids": [str(f) for f in frame_ids]},
                             jobs_of=max(1, min(50, len(frame_ids))))
    task_id = uuid.UUID(task["task_id"])
    pend = (await db.execute(
        select(ShadowDisagreement).where(ShadowDisagreement.sweep_run_id == sweep_run_id,
                                         ShadowDisagreement.frame_id.in_(frame_ids),
                                         ShadowDisagreement.state == "pending"))).scalars().all()
    for d in pend:
        d.state = "queued"
        d.task_id = task_id
    await db.commit()
    log.info("shadow.queued", sweep_run_id=str(sweep_run_id), task_id=str(task_id),
             frames=len(frame_ids), disagreements=len(pend))
    return {"queued": len(pend), "frames": len(frame_ids), "task_id": str(task_id),
            "jobs": task.get("n_jobs")}


# Agreement between two independently trained models is the consensus signal this corpus actually has.
# `oraclyx.record_consensus` is the designed source and it is written one object at a time from a router
# call, so `pseudo_label` holds nothing; the prediction plane meanwhile holds hundreds of thousands of
# detections from several models over the same frames. Where two models that were not trained together
# put the same box on the same class, that is a stronger label than either alone.
AGREE_CONF_FLOOR = 0.35


async def agreements_for_runs(db: AsyncSession, *, run_a: uuid.UUID, run_b: uuid.UUID,
                              iou_thr: float = MATCH_IOU, conf_floor: float = AGREE_CONF_FLOOR,
                              chunk: int = 200, limit: int | None = None) -> dict:
    """Boxes two inference runs agree on, with a soft target from how sure both of them were.

    The soft target is the product of the two confidences rather than the mean or the maximum. A product
    is the one of the three that cannot be carried by a single confident model: 0.95 and 0.4 gives 0.38,
    where a mean would give 0.68 and dress one model's uncertainty as consensus.

    Restricted to the classes both models can emit, for the reason migration 0111 exists: a class one of
    them was never trained on cannot be agreed upon, and counting its absence would silently reweight
    every remaining class.
    """
    a = await db.get(InferenceRun, run_a)
    b = await db.get(InferenceRun, run_b)
    for name, run in (("a", a), ("b", b)):
        if run is None:
            return {"error": f"inference run {name} not found"}
        if run.status != "complete":
            return {"error": f"inference run {name} is '{run.status}'; a partial run is not a consensus"}

    shared: set[int] | None = None
    if a.class_vocab is not None and b.class_vocab is not None:
        shared = {int(i) for i in a.class_vocab} & {int(i) for i in b.class_vocab}

    frames = sorted({f for f in (await db.execute(
        select(Prediction.frame_id).where(
            Prediction.run_id.in_((a.run_id, b.run_id))).distinct())).scalars().all()}, key=str)

    out: list[dict] = []
    for i in range(0, len(frames), chunk):
        window = frames[i:i + chunk]
        by_a = await _dets_by_frame(db, a.run_id, window)
        by_b = await _dets_by_frame(db, b.run_id, window)
        for fid in window:
            da = [d for d in by_a.get(fid, []) if d.conf >= conf_floor
                  and (shared is None or d.class_id in shared)]
            dbx = [d for d in by_b.get(fid, []) if d.conf >= conf_floor
                   and (shared is None or d.class_id in shared)]
            pairs, _lone_a, _lone_b = _pair(da, dbx, iou_thr)
            for ia, ib, iou in pairs:
                if da[ia].class_id != dbx[ib].class_id:
                    continue          # they found the same object and disagree about what it is
                out.append({"frame_id": str(fid), "class_id": da[ia].class_id,
                            # The higher-confidence model's geometry, not an average: averaging two boxes
                            # produces a box neither model proposed.
                            "bbox": (da[ia].bbox if da[ia].conf >= dbx[ib].conf else dbx[ib].bbox),
                            "soft_target": round(da[ia].conf * dbx[ib].conf, 6),
                            "iou": round(iou, 4)})
                if limit and len(out) >= limit:
                    return {"n": len(out), "frames": len(frames), "manifest": out,
                            "source": "model_agreement", "truncated": True,
                            "shared_classes": (sorted(shared) if shared is not None else None)}
    return {"n": len(out), "frames": len(frames), "manifest": out, "source": "model_agreement",
            "truncated": False, "conf_floor": conf_floor,
            "shared_classes": (sorted(shared) if shared is not None else None)}
