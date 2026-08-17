"""How much two annotators agreed, measured on frames they labelled independently.

Everything needed for this existed except the ability to tell the two apart. `services/quality/iaa.py` has
the matching and the statistics, `Issue` has the escalation, `LabelJob` has the work. What was missing was
any record of which annotator produced which box, so two people's labels on one frame were one pile.

**No winner is chosen.** This measures and escalates; a human settles. Automatic winner selection over a
corpus where 702 of 576,393 objects carry a human verdict would be two unverified opinions outvoting a
third, and the name for that is not consensus. (`services/oraclyx/consensus.py` does vote and auto-accept,
which is why this module is called agreement: the two must not be confused.)

**Disagreements become Issues, not a state change.** Flipping `Object.state` to `review` would hide them
among the 519,550 objects already in that state, would mutate a column every export selects on, and would
be irreversible in the sense that matters: afterwards nobody can tell a routed disagreement from a label
that was always in review. An `Issue` is anchored, has an open/resolved lifecycle, and already renders in
the editor beside the frame it is about.

**Blind, or it measures nothing.** A replica job hides existing labels. Two annotators correcting the same
machine proposals are two editors of one label set, and 82.6% of this corpus is pre-labelled, so this is
the normal case rather than an edge one.
"""

from __future__ import annotations

import uuid
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, JobAgreement, LabelJob, LabelTask, Object
from services.quality.iaa import iaa_score, match_boxes

log = get_logger("labelops.agreement")

# The states a replica job must be in before its work can be compared. Scoring a job somebody is still
# working produces a number that changes under the reader and an Issue on a box about to be redrawn.
SUBMITTED_STATES = ("submitted", "completed", "accepted")

DISAGREEMENT_KIND = "disagreement"


class AgreementError(RuntimeError):
    pass


def _pair_key(a: LabelJob, b: LabelJob) -> tuple[LabelJob, LabelJob]:
    """Order a pair the same way every time, so recomputing updates one row rather than writing a mirror."""
    return (a, b) if str(a.job_id) <= str(b.job_id) else (b, a)


async def _objects_for(db: AsyncSession, job_id: uuid.UUID, frame_id: uuid.UUID) -> list[Object]:
    return list((await db.execute(
        select(Object).where(Object.job_id == job_id, Object.frame_id == frame_id)
        .order_by(Object.object_id))).scalars().all())


def _as_sets(objs: list[Object], onto) -> list[dict]:
    out = []
    for o in objs:
        try:
            name = onto.by_id(int(o.class_id)).name
        except Exception:  # noqa: BLE001
            name = str(o.class_id)
        out.append({"bbox": list(o.bbox or []), "class_name": name})
    return out


async def score_replica_group(db: AsyncSession, replica_group: str, *, iou_thresh: float = 0.5,
                              raise_issues: bool = True) -> dict:
    """Compare every pair of jobs in a replica group, frame by frame.

    Only the frames the replicas share are compared. Their `frame_ids` are not equal even at creation:
    `seed_honeypots` appends gold frames chosen from `random.Random(job_id)`, so each replica carries
    honeypots the other does not, and comparing a frame only one of them holds would score its annotator
    against an empty set.
    """
    from services.autolabel.ontology import get_ontology

    gid = uuid.UUID(str(replica_group))
    jobs = list((await db.execute(
        select(LabelJob).where(LabelJob.replica_group == gid)
        .order_by(LabelJob.replica_index))).scalars().all())
    if len(jobs) < 2:
        raise AgreementError(f"replica group {replica_group} has {len(jobs)} job(s); agreement needs two")

    unfinished = [str(j.job_id) for j in jobs if j.state not in SUBMITTED_STATES]
    if unfinished:
        raise AgreementError(
            f"{len(unfinished)} job(s) in this group are still open; a partial set is not a disagreement")

    onto = get_ontology()
    task_id = jobs[0].task_id
    counts = {"pairs": 0, "frames_compared": 0, "disagreements": 0, "issues_opened": 0}
    per_frame: list[dict] = []

    for ja, jb in combinations(jobs, 2):
        a, b = _pair_key(ja, jb)
        counts["pairs"] += 1
        shared = sorted({str(f) for f in (a.frame_ids or [])} & {str(f) for f in (b.frame_ids or [])})
        for fid in shared:
            frame_id = uuid.UUID(fid)
            objs_a = await _objects_for(db, a.job_id, frame_id)
            objs_b = await _objects_for(db, b.job_id, frame_id)
            if not objs_a and not objs_b:
                # Both annotators said the frame is empty. That is agreement, but storing a row per empty
                # frame would bury the frames worth looking at under the ones nobody drew on.
                continue
            counts["frames_compared"] += 1

            set_a, set_b = _as_sets(objs_a, onto), _as_sets(objs_b, onto)
            metrics = iaa_score(set_a, set_b, iou_thresh)

            matched = match_boxes([s["bbox"] for s in set_a], [s["bbox"] for s in set_b], iou_thresh)
            matched_a = {i for i, _, _ in matched}
            matched_b = {j for _, j, _ in matched}
            # Two ways to disagree, and they mean different things. One drew a box the other did not is a
            # detection disagreement, and only that annotator has an object to anchor to. Both drew it and
            # named it differently is a class disagreement, and BOTH labels are in dispute: flagging one
            # side would pick a winner by implication, and which side that was would depend on how two
            # random job ids happened to sort.
            unmatched = ([(a, objs_a[i], "only this annotator drew it") for i in range(len(objs_a))
                          if i not in matched_a]
                         + [(b, objs_b[j], "only this annotator drew it") for j in range(len(objs_b))
                            if j not in matched_b])
            wrong_class: list[tuple] = []
            n_class_conflicts = 0
            for i, j, _iou in matched:
                if set_a[i]["class_name"] == set_b[j]["class_name"]:
                    continue
                n_class_conflicts += 1
                wrong_class.append((a, objs_a[i], f"the other annotator called it {set_b[j]['class_name']}"))
                wrong_class.append((b, objs_b[j], f"the other annotator called it {set_a[i]['class_name']}"))
            disputed = unmatched + wrong_class
            # One conflict, however many labels it touches: a class disagreement is one thing two people
            # disagree about, and counting it twice would inflate every board built on this.
            counts["disagreements"] += len(unmatched) + n_class_conflicts
            n_disagreements = len(unmatched) + n_class_conflicts

            row = JobAgreement(task_id=task_id, replica_group=gid, frame_id=frame_id,
                               job_a_id=a.job_id, job_b_id=b.job_id, metrics=metrics,
                               n_disagreements=n_disagreements)
            existing = (await db.execute(
                select(JobAgreement).where(JobAgreement.frame_id == frame_id,
                                           JobAgreement.job_a_id == a.job_id,
                                           JobAgreement.job_b_id == b.job_id))).scalars().first()
            if existing is not None:
                existing.metrics = metrics
                existing.n_disagreements = n_disagreements
            else:
                db.add(row)

            if raise_issues and disputed:
                counts["issues_opened"] += await _open_issues(db, disputed, frame_id)
            per_frame.append({"frame_id": fid, "job_a": str(a.job_id), "job_b": str(b.job_id),
                              **metrics, "n_disagreements": n_disagreements})

    await db.commit()
    log.info("labelops.agreement_scored", replica_group=str(gid), **counts)
    return {"replica_group": str(gid), "task_id": str(task_id), "counts": counts, "frames": per_frame}


async def _open_issues(db: AsyncSession, disputed: list[tuple], frame_id: uuid.UUID) -> int:
    """One issue per disputed object, and never a second one for the same object.

    Recomputing a group must not multiply the queue somebody has to work.
    """
    from db.models import Issue
    from services.labelops.issues import create_issue

    opened = 0
    for job, obj, why in disputed:
        already = (await db.execute(
            select(Issue).where(Issue.object_id == obj.object_id, Issue.kind == DISAGREEMENT_KIND,
                                Issue.status == "open"))).scalars().first()
        if already is not None:
            continue
        # Through create_issue rather than a bare insert: the reason belongs in an IssueComment (Issue
        # itself carries no text), and the notification and webhook are what make an opened issue something
        # a person hears about rather than a row.
        await create_issue(db, kind=DISAGREEMENT_KIND, body=f"annotators disagree: {why}",
                           object_id=str(obj.object_id), frame_id=str(frame_id),
                           job_id=str(job.job_id), region=list(obj.bbox or []))
        opened += 1
    return opened


async def group_agreement(db: AsyncSession, task_id: str) -> dict:
    """The board read: every scored frame in a task, and the roll-up over them.

    The roll-up is reported beside the frame count it came from, because a task with three compared frames
    and a task with three hundred both produce a number between zero and one and only one of them means
    anything.
    """
    tid = uuid.UUID(str(task_id))
    task = await db.get(LabelTask, tid)
    if task is None:
        raise AgreementError("task not found")
    rows = list((await db.execute(
        select(JobAgreement).where(JobAgreement.task_id == tid))).scalars().all())
    if not rows:
        return {"task_id": str(tid), "replicas": task.replicas, "frames_compared": 0,
                "detail": "no agreement has been scored for this task"}

    def _mean(key: str) -> float:
        vals = [float(r.metrics.get(key, 0.0)) for r in rows if r.metrics]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "task_id": str(tid),
        "replicas": task.replicas,
        "frames_compared": len(rows),
        "detection_agreement": _mean("detection_agreement"),
        "class_agreement": _mean("class_agreement"),
        "mean_iou": _mean("mean_iou"),
        "cohen_kappa": _mean("cohen_kappa"),
        "disagreements": sum(r.n_disagreements for r in rows),
        # Worst first: the point of a board is to say where to look.
        "worst_frames": [
            {"frame_id": str(r.frame_id), "n_disagreements": r.n_disagreements,
             "detection_agreement": (r.metrics or {}).get("detection_agreement")}
            for r in sorted(rows, key=lambda r: (-r.n_disagreements, str(r.frame_id)))[:20]],
    }


async def frames_of_group(db: AsyncSession, replica_group: str) -> list[str]:
    """The frames every replica in a group holds. Exposed because the intersection is not obvious."""
    gid = uuid.UUID(str(replica_group))
    jobs = list((await db.execute(
        select(LabelJob).where(LabelJob.replica_group == gid))).scalars().all())
    if not jobs:
        return []
    shared: set[str] | None = None
    for j in jobs:
        ids = {str(f) for f in (j.frame_ids or [])}
        shared = ids if shared is None else (shared & ids)
    return sorted(shared or set())


async def frame_exists(db: AsyncSession, frame_id: uuid.UUID) -> bool:
    return await db.get(Frame, frame_id) is not None
