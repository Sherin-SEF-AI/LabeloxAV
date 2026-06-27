"""Multi-user collaboration over lakeFS (M4.3). Users and roles already exist (app_user: annotator,
reviewer, admin). Each annotator or experiment works on an isolated branch; an assignment ties an item to
a user and a branch; submitting opens a merge request; a reviewer approves and merges to main with
attribution; a bad merge can be reverted. Per-object isolation is provided by the existing Redis locks;
branch isolation lets annotators work concurrently without colliding. An export pins a lakeFS commit so
provenance and exports continue to reference a specific dataset version.
"""

from __future__ import annotations

import json
import uuid
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Assignment, MergeRequest, User
from versioning import lakefs_store as L

log = get_logger("collaborate")

_CAN_MERGE = ("reviewer", "admin")


async def _role(db: AsyncSession, user_id: str | None) -> str | None:
    if not user_id:
        return None
    u = await db.get(User, UUID(user_id))
    return u.role if u else None


def annotator_branch(user_name: str) -> str:
    return f"annot-{user_name}-{uuid.uuid4().hex[:6]}"


async def create_assignment(db: AsyncSession, item_id: str, user_id: str, branch: str | None = None) -> dict:
    u = await db.get(User, UUID(user_id))
    if u is None:
        return {"error": "user not found"}
    branch = branch or annotator_branch(u.name)
    L.ensure_branch(branch, source=L.default_branch())
    a = Assignment(item_id=item_id, user_id=UUID(user_id), branch=branch, status="assigned")
    db.add(a)
    await db.flush()
    aid = str(a.assignment_id)
    await db.commit()
    log.info("collab.assign", assignment_id=aid, user=u.name, branch=branch)
    return {"assignment_id": aid, "item_id": item_id, "user": u.name, "branch": branch, "status": "assigned"}


async def commit_assignment_work(db: AsyncSession, assignment_id: str, labels: dict[str, dict],
                                 message: str = "annotator work") -> dict:
    """Write an annotator's label edits to their isolated branch and commit (their work, in isolation)."""
    a = await db.get(Assignment, UUID(assignment_id))
    if a is None:
        return {"error": "assignment not found"}
    for oid, label in labels.items():
        L.put_label(a.branch, oid, label)
    commit_id = L.commit(a.branch, message, {"assignment": assignment_id})
    a.status = "in_progress"
    await db.commit()
    return {"assignment_id": assignment_id, "branch": a.branch, "commit": commit_id, "n_labels": len(labels)}


async def open_merge_request(db: AsyncSession, title: str, source_branch: str, author_id: str | None = None,
                             target_branch: str = "main", notes: str | None = None) -> dict:
    mr = MergeRequest(title=title, source_branch=source_branch, target_branch=target_branch,
                      author_id=UUID(author_id) if author_id else None, status="open", notes=notes)
    db.add(mr)
    await db.flush()
    mr_id = str(mr.mr_id)
    # mark the matching assignment submitted
    a = (await db.execute(select(Assignment).where(Assignment.branch == source_branch))).scalars().first()
    if a is not None:
        a.status = "submitted"
    await db.commit()
    log.info("collab.mr_open", mr_id=mr_id, source=source_branch)
    return {"mr_id": mr_id, "title": title, "source_branch": source_branch, "status": "open"}


async def approve_merge_request(db: AsyncSession, mr_id: str, reviewer_id: str) -> dict:
    role = await _role(db, reviewer_id)
    if role not in _CAN_MERGE:
        return {"error": f"role '{role}' cannot approve; needs reviewer or admin"}
    mr = await db.get(MergeRequest, UUID(mr_id))
    if mr is None:
        return {"error": "merge request not found"}
    mr.reviewer_id = UUID(reviewer_id)
    mr.status = "approved"
    await db.commit()
    return {"mr_id": mr_id, "status": "approved", "reviewer": reviewer_id}


async def merge_request(db: AsyncSession, mr_id: str, reviewer_id: str | None = None) -> dict:
    mr = await db.get(MergeRequest, UUID(mr_id))
    if mr is None:
        return {"error": "merge request not found"}
    actor = reviewer_id or (str(mr.reviewer_id) if mr.reviewer_id else None)
    if (await _role(db, actor)) not in _CAN_MERGE:
        return {"error": "merge requires a reviewer or admin"}
    if mr.status not in ("approved", "open"):
        return {"error": f"merge request is {mr.status}"}
    merge_commit = L.merge(mr.source_branch, into=mr.target_branch, message=f"Merge MR: {mr.title}")
    mr.merge_commit = merge_commit
    mr.status = "merged"
    # mark assignments on the branch done
    for a in (await db.execute(select(Assignment).where(Assignment.branch == mr.source_branch))).scalars():
        a.status = "done"
    await db.commit()
    log.info("collab.mr_merged", mr_id=mr_id, merge_commit=merge_commit[:12])
    return {"mr_id": mr_id, "status": "merged", "merge_commit": merge_commit, "into": mr.target_branch}


async def revert_merge_request(db: AsyncSession, mr_id: str, reviewer_id: str | None = None) -> dict:
    mr = await db.get(MergeRequest, UUID(mr_id))
    if mr is None or mr.status != "merged" or not mr.merge_commit:
        return {"error": "merge request is not in a merged state"}
    if reviewer_id and (await _role(db, reviewer_id)) not in _CAN_MERGE:
        return {"error": "revert requires a reviewer or admin"}
    L.revert(mr.target_branch, mr.merge_commit)
    mr.status = "reverted"
    await db.commit()
    log.info("collab.mr_reverted", mr_id=mr_id)
    return {"mr_id": mr_id, "status": "reverted"}


async def pin_export(commit_label: str, manifest: dict) -> dict:
    """Pin an export to a lakeFS commit on main so it references a specific dataset version."""
    L.ensure_branch(L.default_branch())
    L.get_repo().branch(L.default_branch()).object(f"exports/{commit_label}.json").upload(
        data=json.dumps(manifest, sort_keys=True).encode(), content_type="application/json")
    commit_id = L.commit(L.default_branch(), f"export {commit_label}", {"export": commit_label})
    return {"export": commit_label, "lakefs_commit": commit_id}


async def list_assignments(db: AsyncSession, user_id: str | None = None) -> list[dict]:
    q = select(Assignment, User.name).join(User, User.user_id == Assignment.user_id)
    if user_id:
        q = q.where(Assignment.user_id == UUID(user_id))
    rows = (await db.execute(q.order_by(Assignment.created_at.desc()).limit(100))).all()
    return [{"assignment_id": str(a.assignment_id), "item_id": a.item_id, "user": name, "branch": a.branch,
             "status": a.status} for a, name in rows]


async def list_merge_requests(db: AsyncSession) -> list[dict]:
    rows = (await db.execute(select(MergeRequest).order_by(MergeRequest.created_at.desc()).limit(100))).scalars().all()
    return [{"mr_id": str(m.mr_id), "title": m.title, "source_branch": m.source_branch,
             "target_branch": m.target_branch, "status": m.status, "merge_commit": m.merge_commit,
             "created_at": m.created_at.isoformat() if m.created_at else None} for m in rows]
