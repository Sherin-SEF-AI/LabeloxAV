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

**The consensus merge is a separate, explicit step** (`merge_replica_group`, tool #11), run by a reviewer
after scoring. It does not choose between two annotators: it writes a third object, `source='consensus'`,
only where a configurable quorum of independent voters (every replica, plus the frame's machine labels as
one more voter when asked) drew the same thing with the same class, and by default only when at least one
of those voters is a person. The annotators' own labels are never edited. Everything short of a quorum
goes to review as an Issue, as scoring already did, and a merged object nobody has touched since is
replaced, not duplicated, when the group is merged again.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from sqlalchemy import func, select
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
        # `job` is None for a model vote in the consensus merge: a machine label belongs to no job.
        await create_issue(db, kind=DISAGREEMENT_KIND, body=f"annotators disagree: {why}",
                           object_id=str(obj.object_id), frame_id=str(frame_id),
                           job_id=str(job.job_id) if job is not None else None,
                           region=list(obj.bbox or []))
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


# ---- consensus merge (tool #11) -------------------------------------------------------------------------

CONSENSUS_SOURCE = "consensus"
# A vote tally is a machine decision even when the voters were people, so the merged object sits in the
# machine-acceptance state. `accepted` means a person accepted this object, and nobody did.
CONSENSUS_STATE = "auto_accept"
MODEL_VOTER = "model"
# What the model voter is made of: the frame's machine labels. Derived labels (interpolated, propagated)
# restate another label rather than observing the frame, imported ones are somebody else's ground truth,
# and synthetic ones were never on the frame, so none of them is an independent vote.
MODEL_SOURCES = ("fused", "auto_accept", "relabel", "recall", "vlm_review")


@dataclass
class Vote:
    voter: str                 # "r0", "r1", ... per replica, or MODEL_VOTER
    object_id: str
    bbox: list[float]
    class_id: int
    class_name: str
    conf: float
    human: bool


def _mean_box(votes: list[Vote]) -> list[float]:
    n = len(votes)
    return [round(sum(v.bbox[i] for v in votes) / n, 2) for i in range(4)]


def cluster_votes(votes_by_voter: dict[str, list[Vote]], iou_thresh: float = 0.5) -> list[list[Vote]]:
    """Group every voter's boxes into clusters, one per physical thing, at most one box per voter each.

    Class-agnostic on purpose: two people drawing the same car and naming it differently is one thing they
    disagree about, and matching by class would split it into two things each only one of them saw. Each
    voter is matched one-to-one against the running cluster means, greedily by IoU (the same rule as
    `iaa.match_boxes`), and its unmatched boxes open new clusters. Voters are taken in the order given,
    which the caller makes the replica order with the model last, so the people define the clusters and
    the model joins them.
    """
    from core.accel.boxes import box_iou_matrix

    clusters: list[list[Vote]] = []
    for voter in votes_by_voter:
        votes = [v for v in votes_by_voter[voter] if len(v.bbox) >= 4]
        if not votes:
            continue
        reps = [_mean_box(c) for c in clusters]
        iou = box_iou_matrix([v.bbox[:4] for v in votes], reps, device="cpu") if reps else None
        cand = []
        if iou is not None:
            cand = sorted(((float(iou[i, j]), i, j) for i in range(len(votes)) for j in range(len(reps))
                           if iou[i, j] >= iou_thresh), reverse=True)
        used_v: set[int] = set()
        used_c: set[int] = set()
        for _v, i, j in cand:
            if i in used_v or j in used_c:
                continue
            used_v.add(i)
            used_c.add(j)
            clusters[j].append(votes[i])
        for i, v in enumerate(votes):
            if i not in used_v:
                clusters.append([v])
    return clusters


def decide_cluster(cluster: list[Vote], voters: list[str], *, quorum: int, iou_thresh: float,
                   require_human: bool = True) -> dict:
    """Accept, drop or dispute one cluster.

    accepted   at least `quorum` voters drew it with one class, that class is a strict plurality, and (by
               default) a person is among them. Agreement is re-checked against the merged box with
               `oraclyx.consensus.vote`, so a box that joined the cluster through a chain of overlaps but
               sits off the merged box does not count towards the quorum.
    dropped    no person drew it and a quorum of voters left it out: the model's box alone, which every
               annotator independently declined. Nothing to merge and nothing to review.
    disputed   everything else, which goes to review.
    """
    from services.oraclyx.consensus import vote

    n_voters = len(voters)
    counts = Counter(v.class_name for v in cluster)
    ranked = counts.most_common()
    top, n_top = ranked[0]
    unique = len(ranked) == 1 or n_top > ranked[1][1]
    agreeing = [v for v in cluster if v.class_name == top]
    merged = _mean_box(agreeing)
    top_id = agreeing[0].class_id
    tally = vote({"path": "fusion", "class_id": top_id, "bbox": merged, "conf": 0.0},
                 [{"path": v.voter, "class_id": v.class_id, "bbox": v.bbox[:4], "conf": v.conf} for v in cluster],
                 iou_tol=iou_thresh, min_agree=quorum + 1)
    # `vote` files the reference itself under "fusion"; only the real voters count.
    agree_voters = [k for k, d in tally["voters"].items() if k != "fusion" and d["agree"]]
    agree_n = len(agree_voters)
    human_agrees = any(v.human for v in cluster if v.voter in agree_voters)
    drew = {v.voter for v in cluster}
    any_human = any(v.human for v in cluster)

    if agree_n >= quorum and unique and (human_agrees or not require_human):
        status = "accepted"
        reason = f"{agree_n} of {n_voters} voters drew it as {top}"
    elif not any_human and n_voters - len(drew) >= quorum:
        status = "dropped"
        reason = f"only the model drew it; {n_voters - len(drew)} of {n_voters} voters left it out"
    else:
        status = "disputed"
        if len(ranked) > 1:
            reason = "classes differ: " + ", ".join(f"{c} ({n})" for c, n in ranked)
        elif len(drew) < n_voters:
            missing = [x for x in voters if x not in drew]
            reason = f"{len(drew)} of {n_voters} voters drew it; not drawn by {', '.join(missing)}"
        elif agree_n < quorum:
            reason = f"boxes do not overlap enough: {agree_n} of {n_voters} within IoU {iou_thresh}"
        else:
            reason = "no annotator agrees with the model on it"
    return {
        "status": status, "class_name": top, "class_id": top_id, "bbox": merged, "reason": reason,
        "agree_n": agree_n, "n_voters": n_voters, "consensus_score": tally["consensus_score"],
        "class_votes": dict(counts),
        "members": [{"voter": v.voter, "object_id": v.object_id, "class_name": v.class_name,
                     "bbox": [round(float(x), 2) for x in v.bbox[:4]], "human": v.human,
                     "agree": bool(tally["voters"].get(v.voter, {}).get("agree")),
                     "iou": tally["voters"].get(v.voter, {}).get("iou")} for v in cluster],
    }


async def _group_jobs(db: AsyncSession, replica_group: str) -> list[LabelJob]:
    gid = uuid.UUID(str(replica_group))
    jobs = list((await db.execute(
        select(LabelJob).where(LabelJob.replica_group == gid)
        .order_by(LabelJob.replica_index))).scalars().all())
    if len(jobs) < 2:
        raise AgreementError(f"replica group {replica_group} has {len(jobs)} job(s); consensus needs two")
    unfinished = [j for j in jobs if j.state not in SUBMITTED_STATES]
    if unfinished:
        raise AgreementError(
            f"{len(unfinished)} job(s) in this group are still open; merge after every replica is submitted")
    return jobs


def _vote_of(o: Object, voter: str, onto, human: bool) -> Vote:
    try:
        name = onto.by_id(int(o.class_id)).name
    except Exception:  # noqa: BLE001
        name = str(o.class_id)
    return Vote(voter=voter, object_id=str(o.object_id), bbox=[float(x) for x in (o.bbox or [])],
                class_id=int(o.class_id), class_name=name, conf=1.0 if human else float(o.conf or 0.0),
                human=human)


async def _frame_votes(db: AsyncSession, jobs: list[LabelJob], frame_id: uuid.UUID, onto,
                       include_model: bool) -> tuple[dict[str, list[Vote]], dict[str, Object]]:
    by_voter: dict[str, list[Vote]] = {}
    objs: dict[str, Object] = {}
    for j in jobs:
        voter = f"r{j.replica_index}"
        rows = [o for o in await _objects_for(db, j.job_id, frame_id) if o.source != CONSENSUS_SOURCE]
        by_voter[voter] = [_vote_of(o, voter, onto, True) for o in rows]
        objs.update({str(o.object_id): o for o in rows})
    if include_model:
        rows = list((await db.execute(
            select(Object).where(Object.frame_id == frame_id, Object.job_id.is_(None),
                                 Object.source.in_(MODEL_SOURCES), Object.state != "rejected")
            .order_by(Object.object_id))).scalars().all())
        by_voter[MODEL_VOTER] = [_vote_of(o, MODEL_VOTER, onto, False) for o in rows]
        objs.update({str(o.object_id): o for o in rows})
    return by_voter, objs


def _signature(cluster: dict) -> list[str]:
    return sorted(m["object_id"] for m in cluster["members"] if m["agree"])


async def merge_replica_group(db: AsyncSession, replica_group: str, *, iou_thresh: float = 0.5,
                              quorum: int = 2, include_model: bool = True, require_human: bool = True,
                              dry_run: bool = False, raise_issues: bool = True,
                              created_by: str | None = None) -> dict:
    """Merge a submitted replica group into consensus objects, and route what did not agree to review.

    Per shared frame: every replica's boxes and (optionally) the frame's machine labels are clustered,
    each cluster is decided by `decide_cluster`, an accepted cluster becomes one `source='consensus'`
    object carrying its voters in `provenance.consensus`, and a disputed one opens a `disagreement` Issue
    on each of its human members (on the model's box when no person drew it).

    Re-running is safe. A consensus object whose agreeing members are unchanged is left as it is; one that
    no longer has a quorum is removed, but only while nobody has edited it (version 1, source still
    consensus): an object a person has touched is theirs now. `dry_run` writes nothing at all, which is what
    the board uses to show per-class agreement without committing to a merge.
    """
    from services.autolabel.ontology import get_ontology

    jobs = await _group_jobs(db, replica_group)
    gid = jobs[0].replica_group
    voters = [f"r{j.replica_index}" for j in jobs] + ([MODEL_VOTER] if include_model else [])
    if not 2 <= quorum <= len(voters):
        raise AgreementError(f"quorum must be between 2 and the {len(voters)} voters")
    if not 0.05 <= iou_thresh <= 0.95:
        raise AgreementError("iou_thresh must be between 0.05 and 0.95")

    onto = get_ontology()
    shared = await frames_of_group(db, str(gid))
    job_by_voter = {f"r{j.replica_index}": j for j in jobs}
    counts = {"frames": 0, "clusters": 0, "accepted": 0, "created": 0, "unchanged": 0, "removed": 0,
              "kept_edited": 0, "disputed": 0, "dropped": 0, "issues_opened": 0}
    per_class: dict[str, dict[str, int]] = {}
    frames_out: list[dict] = []
    disputed_all: list[tuple] = []

    for fid in shared:
        frame_id = uuid.UUID(fid)
        by_voter, objs = await _frame_votes(db, jobs, frame_id, onto, include_model)
        if not any(by_voter.values()):
            continue
        counts["frames"] += 1
        decided = [decide_cluster(c, voters, quorum=quorum, iou_thresh=iou_thresh, require_human=require_human)
                   for c in cluster_votes(by_voter, iou_thresh)]
        existing = [] if dry_run else list((await db.execute(
            select(Object).where(Object.frame_id == frame_id, Object.source == CONSENSUS_SOURCE,
                                 Object.provenance["consensus"]["replica_group"].astext == str(gid))
        )).scalars().all())
        existing_sig = {tuple((o.provenance or {}).get("consensus", {}).get("signature") or []): o for o in existing}
        keep_sigs: set[tuple] = set()

        for d in decided:
            counts["clusters"] += 1
            counts[d["status"]] += 1
            pc = per_class.setdefault(d["class_name"], {"accepted": 0, "disputed": 0, "dropped": 0})
            pc[d["status"]] += 1
            if d["status"] == "accepted":
                sig = tuple(_signature(d))
                keep_sigs.add(sig)
                if dry_run:
                    continue
                if sig in existing_sig:
                    counts["unchanged"] += 1
                    d["object_id"] = str(existing_sig[sig].object_id)
                    continue
                obj = Object(object_id=uuid.uuid4(), frame_id=frame_id, class_id=int(d["class_id"]),
                             bbox=[float(x) for x in d["bbox"]], conf=float(d["agree_n"]) / len(voters),
                             source=CONSENSUS_SOURCE, state=CONSENSUS_STATE, attrs={},
                             provenance={"consensus": {
                                 "replica_group": str(gid), "task_id": str(jobs[0].task_id),
                                 "signature": list(sig), "voters": voters, "agree_n": d["agree_n"],
                                 "quorum": quorum, "iou_thresh": iou_thresh, "include_model": include_model,
                                 "consensus_score": d["consensus_score"], "members": d["members"],
                                 "merged_by": created_by}})
                db.add(obj)
                counts["created"] += 1
                d["object_id"] = str(obj.object_id)
            elif d["status"] == "disputed":
                anchors = [m for m in d["members"] if m["human"]] or d["members"]
                for m in anchors:
                    o = objs.get(m["object_id"])
                    if o is not None:
                        disputed_all.append((job_by_voter.get(m["voter"]), o, d["reason"], frame_id))

        if not dry_run:
            for sig, o in existing_sig.items():
                if sig in keep_sigs:
                    continue
                if o.source == CONSENSUS_SOURCE and (o.version or 1) == 1:
                    await db.delete(o)
                    counts["removed"] += 1
                else:
                    counts["kept_edited"] += 1
        frames_out.append({"frame_id": fid, "clusters": decided})

    # A dry run added nothing, so there is nothing to roll back; rolling back anyway would expire every row
    # the caller holds (the board walks several groups in one session).
    if not dry_run:
        await db.commit()
        if raise_issues:
            by_frame: dict[uuid.UUID, list[tuple]] = {}
            for job, o, why, frame_id in disputed_all:
                by_frame.setdefault(frame_id, []).append((job, o, why))
            for frame_id, items in by_frame.items():
                counts["issues_opened"] += await _open_issues(db, items, frame_id)
    log.info("labelops.consensus_merged", replica_group=str(gid), dry_run=dry_run, **counts)
    return {"replica_group": str(gid), "task_id": str(jobs[0].task_id), "voters": voters,
            "params": {"iou_thresh": iou_thresh, "quorum": quorum, "include_model": include_model,
                       "require_human": require_human},
            "dry_run": dry_run, "counts": counts, "per_class": per_class, "frames": frames_out}


async def score_and_merge(db: AsyncSession, replica_group: str, **merge_kw) -> dict:
    """Score the group pairwise (the existing measurement), then merge it.

    Scoring runs without opening its own pairwise issues: the merge opens one per disputed cluster with a
    reason that names every voter, and two issues about the same box from two passes would be noise.
    """
    scored = await score_replica_group(db, replica_group, iou_thresh=merge_kw.get("iou_thresh", 0.5),
                                       raise_issues=False)
    merged = await merge_replica_group(db, replica_group, **merge_kw)
    merged["agreement"] = scored["counts"]
    return merged


async def consensus_board(db: AsyncSession, task_id: str, *, iou_thresh: float = 0.5, quorum: int = 2,
                          include_model: bool = True, require_human: bool = True,
                          max_groups: int = 60) -> dict:
    """Per replica group: its jobs, whether it can be merged, and per-class agreement, as a heatmap.

    The per-class cells come from a dry-run merge with the given parameters, so the heatmap answers "if I
    merged now with these settings, how much of each class would be accepted" and moves when the settings
    do. A group with a replica still open shows as waiting rather than as zero agreement.
    """
    tid = uuid.UUID(str(task_id))
    task = await db.get(LabelTask, tid)
    if task is None:
        raise AgreementError("task not found")
    jobs = list((await db.execute(
        select(LabelJob).where(LabelJob.task_id == tid, LabelJob.replica_group.isnot(None))
        .order_by(LabelJob.created_at, LabelJob.replica_index))).scalars().all())
    groups: dict[uuid.UUID, list[LabelJob]] = {}
    for j in jobs:
        groups.setdefault(j.replica_group, []).append(j)

    agreement_rows = list((await db.execute(
        select(JobAgreement).where(JobAgreement.task_id == tid))).scalars().all())
    agree_by_group: dict[uuid.UUID, list[JobAgreement]] = {}
    for r in agreement_rows:
        agree_by_group.setdefault(r.replica_group, []).append(r)

    out_groups = []
    classes: set[str] = set()
    for gid, gjobs in list(groups.items())[:max_groups]:
        ready = len(gjobs) >= 2 and all(j.state in SUBMITTED_STATES for j in gjobs)
        rows = agree_by_group.get(gid, [])
        det = [float((r.metrics or {}).get("detection_agreement", 0.0)) for r in rows]
        n_cons = int((await db.execute(
            select(func.count(Object.object_id)).where(
                Object.source == CONSENSUS_SOURCE,
                Object.provenance["consensus"]["replica_group"].astext == str(gid)))).scalar() or 0)
        g = {"replica_group": str(gid), "ready": ready,
             "jobs": [{"job_id": str(j.job_id), "replica_index": j.replica_index, "state": j.state,
                       "assignee_id": str(j.assignee_id) if j.assignee_id else None,
                       "n_frames": len(j.frame_ids or [])} for j in gjobs],
             "scored_frames": len(rows),
             "detection_agreement": round(sum(det) / len(det), 4) if det else None,
             "consensus_objects": n_cons, "per_class": {}, "counts": None}
        if ready:
            preview = await merge_replica_group(db, str(gid), iou_thresh=iou_thresh, quorum=quorum,
                                                include_model=include_model, require_human=require_human,
                                                dry_run=True)
            g["per_class"] = preview["per_class"]
            g["counts"] = preview["counts"]
            classes.update(preview["per_class"])
        out_groups.append(g)

    ordered = sorted(classes)
    cells = []
    for cname in ordered:
        row = []
        for g in out_groups:
            pc = g["per_class"].get(cname)
            if not pc:
                row.append(None)
                continue
            decided = pc["accepted"] + pc["disputed"]
            row.append({**pc, "rate": round(pc["accepted"] / decided, 4) if decided else None})
        cells.append(row)
    return {"task_id": str(tid), "name": task.name, "replicas": task.replicas,
            "params": {"iou_thresh": iou_thresh, "quorum": quorum, "include_model": include_model,
                       "require_human": require_human},
            "groups": out_groups, "n_groups": len(groups),
            "heatmap": {"classes": ordered, "groups": [g["replica_group"] for g in out_groups], "cells": cells}}


async def task_disagreements(db: AsyncSession, task_id: str, *, status: str = "open",
                             limit: int = 200) -> list[dict]:
    """The disagreement issues on a task's frames, oldest frame first: the queue a reviewer works."""
    from db.models import Issue, IssueComment
    from services.autolabel.ontology import get_ontology

    tid = uuid.UUID(str(task_id))
    jobs = list((await db.execute(select(LabelJob).where(LabelJob.task_id == tid))).scalars().all())
    frame_ids = {uuid.UUID(str(f)) for j in jobs for f in (j.frame_ids or [])}
    if not frame_ids:
        return []
    onto = get_ontology()
    rows = (await db.execute(
        select(Issue, Object, Frame.ts_ns)
        .join(Frame, Frame.frame_id == Issue.frame_id)
        .outerjoin(Object, Object.object_id == Issue.object_id)
        .where(Issue.kind == DISAGREEMENT_KIND, Issue.status == status, Issue.frame_id.in_(list(frame_ids)))
        .order_by(Frame.ts_ns, Issue.created_at).limit(limit))).all()
    ids = [r[0].issue_id for r in rows]
    first_comment: dict[uuid.UUID, str] = {}
    if ids:
        for iid, body in (await db.execute(
                select(IssueComment.issue_id, IssueComment.body).where(IssueComment.issue_id.in_(ids))
                .order_by(IssueComment.created_at))).all():
            first_comment.setdefault(iid, body)
    job_index = {j.job_id: j.replica_index for j in jobs}
    out = []
    for issue, obj, _ts in rows:
        cname = None
        if obj is not None:
            try:
                cname = onto.by_id(int(obj.class_id)).name
            except Exception:  # noqa: BLE001
                cname = str(obj.class_id)
        out.append({"issue_id": str(issue.issue_id), "frame_id": str(issue.frame_id),
                    "object_id": str(issue.object_id) if issue.object_id else None,
                    "job_id": str(issue.job_id) if issue.job_id else None,
                    "replica_index": job_index.get(issue.job_id) if issue.job_id else None,
                    "class_name": cname, "bbox": list(obj.bbox) if obj is not None else issue.region,
                    "source": obj.source if obj is not None else None,
                    "reason": first_comment.get(issue.issue_id, ""),
                    "created_at": issue.created_at.isoformat() if issue.created_at else None})
    return out
