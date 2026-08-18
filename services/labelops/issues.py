"""Issue threads pinned to annotations.

A reviewer rejecting a job teaches nobody anything. An issue anchored to the exact object (or to a box region
on a frame, when the complaint is that something is MISSING and so has no object to point at) turns review
into specific, resolvable feedback that survives the conversation.

Issues are deliberately not a workflow state on the job: a job can carry several open issues at once, and one
issue can outlive the job it was raised in, which a per-job status field could not express.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Issue, IssueComment, User

log = get_logger("labelops_issues")

# `disagreement` is opened by the agreement pass rather than by a person: two annotators labelled the
# same frame independently and did not produce the same answer. It carries the same lifecycle as the
# rest, because settling it is the same act.
KINDS = ("comment", "wrong_class", "bad_geometry", "missing", "duplicate", "unclear", "disagreement")


class IssueError(RuntimeError):
    pass


async def create_issue(db: AsyncSession, *, kind: str = "comment", body: str | None = None,
                       object_id: str | None = None, frame_id: str | None = None,
                       job_id: str | None = None, region: list | None = None,
                       created_by: str | None = None) -> dict:
    """Open an issue. It must be anchored to something: an object, or a frame (optionally with a region)."""
    if kind not in KINDS:
        raise IssueError(f"kind must be one of {KINDS}")
    if not object_id and not frame_id:
        raise IssueError("an issue must be anchored to an object or a frame")

    issue = Issue(kind=kind, status="open",
                  object_id=UUID(object_id) if object_id else None,
                  frame_id=UUID(frame_id) if frame_id else None,
                  job_id=UUID(job_id) if job_id else None,
                  region=region,
                  created_by=UUID(created_by) if created_by else None)
    db.add(issue)
    await db.flush()
    if body and body.strip():
        db.add(IssueComment(issue_id=issue.issue_id, body=body.strip(),
                            author_id=UUID(created_by) if created_by else None))
    await db.commit()
    log.info("labelops.issue_opened", issue=str(issue.issue_id), kind=kind,
             object=object_id, frame=frame_id)
    from services.integrations.webhooks import emit

    await emit("issue.opened", {"issue_id": str(issue.issue_id), "kind": kind,
                                "object_id": object_id, "frame_id": frame_id})

    # An issue nobody is told about is a note to self. Addressed to the reviewer role rather than a person,
    # because who picks it up is a duty rota, not a property of the issue.
    from services.notify import notify

    href = f"/frame/{frame_id}" if frame_id else f"/object/{object_id}"
    await notify(db, kind="issue_opened", severity="warn" if kind != "comment" else "info",
                 title=f"{kind.replace('_', ' ')} raised",
                 body=(body or "").strip()[:280] or None,
                 href=href,
                 subject_type="issue", subject_id=str(issue.issue_id), supersede=False)

    # And the person whose work it is about, which the rota notification does not reach.
    #
    # An issue is feedback on a specific label. Telling only the reviewer role means the annotator who drew
    # it is the one participant never informed - they find out when the job comes back, if at all, which is
    # the slowest possible form of the correction loop and the reason issues read as write-only from their
    # side. Addressed to the person rather than the annotator role, because "your box is wrong" is not a
    # duty rota.
    owner = await _work_owner(db, object_id=object_id, job_id=job_id)
    if owner is not None and str(owner) != str(created_by or ""):
        await notify(db, kind="issue_opened", severity="warn" if kind != "comment" else "info",
                     user_id=owner,
                     title=f"{kind.replace('_', ' ')} on your label",
                     body=(body or "").strip()[:280] or None,
                     href=href,
                     subject_type="issue", subject_id=str(issue.issue_id), supersede=False)
    return await get_issue(db, str(issue.issue_id))


async def _work_owner(db: AsyncSession, *, object_id: str | None, job_id: str | None):
    """Whose work an issue is about: the annotator on the object, else the job's assignee.

    The object is preferred because it is the more specific claim - an issue pinned to a box is about
    whoever drew that box, even if the job has since been reassigned. Returns None when neither is
    recorded, which is an ordinary state for a machine-labelled object nobody has touched.
    """
    from db.models import LabelJob, Object

    if object_id:
        obj = await db.get(Object, UUID(object_id))
        if obj is not None and obj.annotator_id is not None:
            return obj.annotator_id
    if job_id:
        job = await db.get(LabelJob, UUID(job_id))
        if job is not None and job.assignee_id is not None:
            return job.assignee_id
    return None


async def comment(db: AsyncSession, issue_id: str, body: str, author_id: str | None = None) -> dict:
    if not body.strip():
        raise IssueError("comment body required")
    issue = await db.get(Issue, UUID(issue_id))
    if issue is None:
        raise IssueError("issue not found")
    db.add(IssueComment(issue_id=issue.issue_id, body=body.strip(),
                        author_id=UUID(author_id) if author_id else None))
    await db.commit()
    return await get_issue(db, issue_id)


async def resolve_issue(db: AsyncSession, issue_id: str, resolved_by: str | None = None,
                        reopen: bool = False) -> dict:
    issue = await db.get(Issue, UUID(issue_id))
    if issue is None:
        raise IssueError("issue not found")
    if reopen:
        issue.status, issue.resolved_at, issue.resolved_by = "open", None, None
    else:
        issue.status = "resolved"
        issue.resolved_at = datetime.now(UTC)
        issue.resolved_by = UUID(resolved_by) if resolved_by else None
    await db.commit()
    log.info("labelops.issue_status", issue=issue_id, status=issue.status)
    if issue.status == "resolved":
        from services.integrations.webhooks import emit

        await emit("issue.resolved", {"issue_id": issue_id})
    return await get_issue(db, issue_id)


async def get_issue(db: AsyncSession, issue_id: str) -> dict:
    issue = await db.get(Issue, UUID(issue_id))
    if issue is None:
        raise IssueError("issue not found")
    rows = (await db.execute(
        select(IssueComment, User.name)
        .join(User, User.user_id == IssueComment.author_id, isouter=True)
        .where(IssueComment.issue_id == issue.issue_id)
        .order_by(IssueComment.created_at))).all()
    return {**_issue_dict(issue),
            "comments": [{"comment_id": str(c.comment_id), "body": c.body, "author": name,
                          "created_at": c.created_at.isoformat() if c.created_at else None}
                         for c, name in rows]}


def _issue_dict(i: Issue) -> dict:
    return {"issue_id": str(i.issue_id), "kind": i.kind, "status": i.status,
            "object_id": str(i.object_id) if i.object_id else None,
            "frame_id": str(i.frame_id) if i.frame_id else None,
            "job_id": str(i.job_id) if i.job_id else None,
            "region": i.region,
            "created_at": i.created_at.isoformat() if i.created_at else None,
            "resolved_at": i.resolved_at.isoformat() if i.resolved_at else None}


async def list_issues(db: AsyncSession, *, frame_id: str | None = None, job_id: str | None = None,
                      object_id: str | None = None, status: str | None = None,
                      about_user: str | None = None, limit: int = 200) -> list[dict]:
    """Issues, filtered.

    `about_user` is the dimension that was missing: issues could be listed by frame, job, object or status,
    which are all ways of asking "what is wrong with this thing" and none of them ways of asking "what has
    been said about my work". Without it an annotator had no query that would find the feedback on their
    own labels, which is why issues read as write-only from the side they are addressed to.
    """
    from db.models import LabelJob, Object

    stmt = select(Issue)
    if about_user:
        uid = UUID(about_user)
        # Either anchor counts: the object carries who drew it, and the job carries who was working it.
        # An issue pinned to a box on someone else's job is still about the person who drew the box.
        mine_objects = select(Object.object_id).where(Object.annotator_id == uid)
        mine_jobs = select(LabelJob.job_id).where(LabelJob.assignee_id == uid)
        stmt = stmt.where(or_(Issue.object_id.in_(mine_objects), Issue.job_id.in_(mine_jobs)))
    if frame_id:
        stmt = stmt.where(Issue.frame_id == UUID(frame_id))
    if job_id:
        stmt = stmt.where(Issue.job_id == UUID(job_id))
    if object_id:
        stmt = stmt.where(Issue.object_id == UUID(object_id))
    if status:
        stmt = stmt.where(Issue.status == status)
    rows = (await db.execute(stmt.order_by(Issue.created_at.desc()).limit(limit))).scalars().all()
    if not rows:
        return []
    counts = dict((await db.execute(
        select(IssueComment.issue_id, func.count())
        .where(IssueComment.issue_id.in_([i.issue_id for i in rows]))
        .group_by(IssueComment.issue_id))).all())
    return [{**_issue_dict(i), "n_comments": int(counts.get(i.issue_id, 0))} for i in rows]
