"""Honeypot quality control: measure an annotator against hidden ground truth.

The only way to know whether labels are good is to mix in frames whose correct answer is already known and
score silently. A fraction of each job's frames are drawn from a sealed GoldSet; the annotator cannot tell
them apart from real work, so the measurement is not gameable.

Scoring reuses the same class-agnostic IoU matching as the evaluation drill-down
(services/analytics/evaluation.py), so "accuracy" means the same thing to an annotator scorecard as it does
to a model scorecard. Two different definitions of correct across the two would make them incomparable and
quietly misleading.
"""

from __future__ import annotations

import random
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import GoldSet, LabelJob, LabelProject, Object, Review, User

log = get_logger("labelops_quality")


async def seed_honeypots(db: AsyncSession, job: LabelJob, project: LabelProject) -> int:
    """Mix gold frames into a job at the project's configured rate. Returns how many were added.

    Deterministic per job id so re-running task creation does not reshuffle which frames are hidden gold.
    """
    if not project.gold_id or project.honeypot_frac <= 0:
        return 0
    gold = await db.get(GoldSet, project.gold_id)
    if gold is None or not gold.object_ids:
        log.warning("labelops.honeypot_gold_missing", gold_id=project.gold_id)
        return 0

    gold_obj_ids = [UUID(str(o)) for o in gold.object_ids]
    gold_frames = (await db.execute(
        select(Object.frame_id).where(Object.object_id.in_(gold_obj_ids)).distinct())).scalars().all()
    if not gold_frames:
        # A gold set is a list of object ids, so it rots when those objects are deleted and keeps claiming
        # its original size. Returning 0 in silence made a project configured for quality measurement look
        # identical to one that was not, and no honeypot in it would ever be scored.
        log.warning("labelops.honeypot_gold_rotted", gold_id=project.gold_id,
                    listed=len(gold_obj_ids),
                    detail="the gold set's objects no longer exist; nothing can be seeded from it")
        return 0

    n = max(1, int(round(len(job.frame_ids or []) * project.honeypot_frac)))
    rng = random.Random(str(job.job_id))          # stable choice per job
    picked = rng.sample([str(f) for f in gold_frames], min(n, len(gold_frames)))

    existing = {str(f) for f in (job.frame_ids or [])}
    # A gold frame the job ALREADY contains is a perfectly good honeypot: we know the truth for it, and it is
    # indistinguishable from the rest of the work. Grade it in place rather than skipping it (skipping meant a
    # task drawn from the same session as the gold set got no quality measurement at all) and append only the
    # gold frames the job does not already have.
    already = [f for f in picked if f in existing]
    fresh = [f for f in picked if f not in existing]
    marked = [*already, *fresh]
    if not marked:
        return 0
    prev = {str(f) for f in (job.honeypot_frame_ids or [])}
    job.frame_ids = [*(job.frame_ids or []), *fresh]
    job.honeypot_frame_ids = [*(job.honeypot_frame_ids or []), *[f for f in marked if f not in prev]]
    return len([f for f in marked if f not in prev])


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


async def score_honeypots(db: AsyncSession, job: LabelJob, project: LabelProject | None,
                          iou_thr: float = 0.5) -> dict:
    """Grade the annotator's labels on this job's hidden gold frames against the sealed truth.

    accuracy = matched-with-correct-class / gold objects on those frames. A missed gold object and a
    wrong-class match both count against, which is what an annotator scorecard should mean.
    """
    hp = [UUID(str(f)) for f in (job.honeypot_frame_ids or [])]
    if not hp:
        return {"accuracy": None, "detail": "no honeypots in this job"}
    gold_id = project.gold_id if project else None
    if not gold_id:
        return {"accuracy": None, "detail": "project has no gold set"}
    gold = await db.get(GoldSet, gold_id)
    if gold is None or not gold.object_ids:
        return {"accuracy": None, "detail": "gold set missing"}

    gold_ids = {UUID(str(o)) for o in gold.object_ids}

    gold_rows = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.frame_id.in_(hp), Object.object_id.in_(gold_ids)))).all()
    if not gold_rows:
        return {"accuracy": None, "detail": "no gold objects on the honeypot frames"}

    # The annotator's work: human-sourced objects on those frames that are not themselves the gold rows.
    #
    # Scoped to this job once the job is known. Without that scope, two annotators working replicas of the
    # same frames are graded against each other's boxes as well as their own, so one person's honeypot
    # score moves when a different person draws. Objects predating job attribution carry no job_id and are
    # matched by the fallback, which is every job that existed before replicas did.
    ann_q = (select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox)
             .where(Object.frame_id.in_(hp), Object.source == "human",
                    Object.object_id.notin_(gold_ids)))
    if job.replica_group is not None:
        ann_q = ann_q.where(Object.job_id == job.job_id)
    ann_rows = (await db.execute(ann_q)).all()

    by_frame_gold: dict = {}
    for _oid, fid, cid, bbox in gold_rows:
        by_frame_gold.setdefault(fid, []).append((cid, list(bbox)))
    by_frame_ann: dict = {}
    for _oid, fid, cid, bbox in ann_rows:
        by_frame_ann.setdefault(fid, []).append((cid, list(bbox)))

    n_gold = sum(len(v) for v in by_frame_gold.values())
    correct = wrong_class = missed = 0
    for fid, golds in by_frame_gold.items():
        anns = list(by_frame_ann.get(fid, []))
        used = set()
        for g_cid, g_box in golds:
            best_j, best_iou = -1, 0.0
            for j, (_a_cid, a_box) in enumerate(anns):
                if j in used:
                    continue
                v = _iou(g_box, a_box)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_thr:
                used.add(best_j)
                if int(anns[best_j][0]) == int(g_cid):
                    correct += 1
                else:
                    wrong_class += 1
            else:
                missed += 1

    accuracy = (correct / n_gold) if n_gold else None
    out = {"accuracy": round(accuracy, 4) if accuracy is not None else None,
           "gold_objects": n_gold, "correct": correct, "wrong_class": wrong_class, "missed": missed,
           "honeypot_frames": len(hp), "iou_thr": iou_thr}
    log.info("labelops.honeypots_scored", job=str(job.job_id), **{k: out[k] for k in ("accuracy", "correct", "missed")})
    return out


async def annotator_scorecards(db: AsyncSession, *, project_id: str | None = None,
                               limit: int = 50) -> list[dict]:
    """Per-annotator throughput and quality.

    Throughput and time come from the Review trail that already exists (Review.time_spent_ms); quality comes
    from submitted jobs' honeypot accuracy. Median time is reported alongside the mean because annotation
    times are heavily skewed: one interrupted session drags a mean far off what the work actually costs.
    """
    rows = (await db.execute(
        select(User.user_id, User.name, User.role,
               func.count(Review.review_id), func.sum(Review.time_spent_ms),
               func.avg(Review.time_spent_ms),
               func.percentile_cont(0.5).within_group(Review.time_spent_ms.asc()))
        .join(Review, Review.user_id == User.user_id, isouter=True)
        .group_by(User.user_id, User.name, User.role)
        .order_by(func.count(Review.review_id).desc()).limit(limit))).all()

    jstmt = (select(LabelJob.assignee_id, func.count(), func.avg(LabelJob.honeypot_accuracy))
             .where(LabelJob.assignee_id.isnot(None)))
    if project_id:
        from db.models import LabelTask

        jstmt = jstmt.join(LabelTask, LabelTask.task_id == LabelJob.task_id).where(
            LabelTask.project_id == UUID(project_id))
    jrows = {a: (int(n), float(acc) if acc is not None else None)
             for a, n, acc in (await db.execute(jstmt.group_by(LabelJob.assignee_id))).all()}

    out = []
    for uid, name, role, n_rev, total_ms, avg_ms, med_ms in rows:
        n_jobs, acc = jrows.get(uid, (0, None))
        out.append({
            "user_id": str(uid), "name": name, "role": role,
            "reviews": int(n_rev or 0),
            "total_time_min": round(float(total_ms or 0) / 60000.0, 1),
            "mean_time_ms": int(avg_ms) if avg_ms else 0,
            "median_time_ms": int(med_ms) if med_ms else 0,
            "jobs": n_jobs,
            "honeypot_accuracy": round(acc, 4) if acc is not None else None,
        })
    return out
