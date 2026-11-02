"""Projects as a way to decide who sees what: list them, set who may see one, manage members, move sessions.

The rule itself is in services/project_access.py. Changing it is for administrators; reading the list is for
anyone signed in, who sees the projects they may see. Every change is recorded in the activity log, because
"who could see this data, and since when" is a question somebody eventually asks.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import LabelProject, ProjectMember, User
from db.models import Session as DbSession
from services.activity import record_activity
from services.api.deps import current_user, db_session, require_role
from services.api.project_guard import caller_hidden_projects
from services.project_access import VISIBILITIES, as_hidden

router = APIRouter()


def _uuid(raw: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(400, f"invalid {what} id") from None


async def _project(db: AsyncSession, project_id: str) -> LabelProject:
    p = await db.get(LabelProject, _uuid(project_id, "project"))
    if p is None:
        raise HTTPException(404, "project not found")
    return p


@router.get("/projects", dependencies=[Depends(require_role("annotator"))])
async def list_projects(db: AsyncSession = Depends(db_session), user=Depends(current_user),
                        hidden=Depends(caller_hidden_projects)):
    """The projects the caller may see, with how many sessions and members each has, and whether the caller
    is a member."""
    hidden = as_hidden(hidden)
    sessions: dict = {pid: n for pid, n in (await db.execute(
        select(DbSession.project_id, func.count()).group_by(DbSession.project_id))).all()}
    members: dict = {pid: n for pid, n in (await db.execute(
        select(ProjectMember.project_id, func.count()).group_by(ProjectMember.project_id))).all()}
    mine: set = set()
    if user is not None:
        mine = set((await db.execute(select(ProjectMember.project_id)
                                     .where(ProjectMember.user_id == user.user_id))).scalars().all())
    rows = (await db.execute(select(LabelProject).order_by(LabelProject.created_at))).scalars().all()
    return {"projects": [
        {"project_id": str(p.project_id), "name": p.name, "description": p.description,
         "visibility": p.visibility or "team", "sessions": int(sessions.get(p.project_id, 0)),
         "members": int(members.get(p.project_id, 0)), "is_member": p.project_id in mine,
         "created_at": p.created_at.isoformat() if p.created_at else None}
        for p in rows if p.project_id not in hidden]}


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    visibility: str = "team"


@router.post("/projects")
async def create_project(body: ProjectIn, admin=Depends(require_role("admin")),
                         db: AsyncSession = Depends(db_session)):
    if body.visibility not in VISIBILITIES:
        raise HTTPException(400, f"visibility must be one of {list(VISIBILITIES)}")
    name = body.name.strip()
    if (await db.execute(select(LabelProject).where(LabelProject.name == name))).scalar_one_or_none():
        raise HTTPException(409, f"a project named '{name}' already exists")
    p = LabelProject(name=name, description=body.description, visibility=body.visibility,
                     created_by=admin.user_id if admin else None)
    db.add(p)
    await db.commit()
    await record_activity(db, user=admin, verb="created_project", subject_type="project",
                          subject_id=str(p.project_id), summary=f"created project {name} ({body.visibility})")
    return {"project_id": str(p.project_id), "name": p.name, "visibility": p.visibility}


class ProjectPatch(BaseModel):
    visibility: str | None = None
    description: str | None = Field(default=None, max_length=2000)


@router.patch("/projects/{project_id}")
async def change_project(project_id: str, body: ProjectPatch, admin=Depends(require_role("admin")),
                         db: AsyncSession = Depends(db_session)):
    p = await _project(db, project_id)
    if body.visibility is not None:
        if body.visibility not in VISIBILITIES:
            raise HTTPException(400, f"visibility must be one of {list(VISIBILITIES)}")
        before, p.visibility = p.visibility, body.visibility
        if before != body.visibility:
            await record_activity(db, user=admin, verb="changed_visibility", subject_type="project",
                                  subject_id=str(p.project_id), commit=False,
                                  summary=f"{p.name}: visible to {'members only' if body.visibility == 'members' else 'the whole team'}")
    if body.description is not None:
        p.description = body.description
    await db.commit()
    return {"project_id": str(p.project_id), "name": p.name, "visibility": p.visibility,
            "description": p.description}


@router.get("/projects/{project_id}/members", dependencies=[Depends(require_role("admin"))])
async def members(project_id: str, db: AsyncSession = Depends(db_session)):
    p = await _project(db, project_id)
    rows = (await db.execute(select(User.user_id, User.name, User.role, ProjectMember.created_at)
                             .join(ProjectMember, ProjectMember.user_id == User.user_id)
                             .where(ProjectMember.project_id == p.project_id).order_by(User.name))).all()
    return {"project_id": str(p.project_id), "members": [
        {"user_id": str(uid), "name": name, "role": role, "added_at": at.isoformat() if at else None}
        for uid, name, role, at in rows]}


@router.put("/projects/{project_id}/members/{user_id}")
async def add_member(project_id: str, user_id: str, admin=Depends(require_role("admin")),
                     db: AsyncSession = Depends(db_session)):
    p = await _project(db, project_id)
    u = await db.get(User, _uuid(user_id, "user"))
    if u is None:
        raise HTTPException(404, "user not found")
    if await db.get(ProjectMember, (p.project_id, u.user_id)) is None:
        db.add(ProjectMember(project_id=p.project_id, user_id=u.user_id, added_by=admin.user_id if admin else None))
        await record_activity(db, user=admin, verb="added_member", subject_type="project", commit=False,
                              subject_id=str(p.project_id), summary=f"added {u.name} to {p.name}")
        await db.commit()
    return {"project_id": str(p.project_id), "user_id": str(u.user_id), "member": True}


@router.delete("/projects/{project_id}/members/{user_id}")
async def remove_member(project_id: str, user_id: str, admin=Depends(require_role("admin")),
                        db: AsyncSession = Depends(db_session)):
    p = await _project(db, project_id)
    uid = _uuid(user_id, "user")
    res = await db.execute(delete(ProjectMember).where(ProjectMember.project_id == p.project_id,
                                                       ProjectMember.user_id == uid))
    if getattr(res, "rowcount", 0):
        u = await db.get(User, uid)
        await record_activity(db, user=admin, verb="removed_member", subject_type="project", commit=False,
                              subject_id=str(p.project_id),
                              summary=f"removed {u.name if u else uid} from {p.name}")
    await db.commit()
    return {"project_id": str(p.project_id), "user_id": str(uid), "member": False}


class MoveIn(BaseModel):
    session_ids: list[str] = Field(min_length=1, max_length=1000)


@router.post("/projects/{project_id}/sessions")
async def move_sessions(project_id: str, body: MoveIn, admin=Depends(require_role("admin")),
                        db: AsyncSession = Depends(db_session)):
    """Put sessions in this project. A session is in exactly one project, so this moves it out of its old one,
    and with it who may see it."""
    p = await _project(db, project_id)
    ids = [_uuid(s, "session") for s in body.session_ids]
    res = await db.execute(update(DbSession).where(DbSession.session_id.in_(ids)).values(project_id=p.project_id))
    await record_activity(db, user=admin, verb="moved_sessions", subject_type="project", commit=False,
                          subject_id=str(p.project_id),
                          summary=f"moved {getattr(res, 'rowcount', 0)} session(s) into {p.name}",
                          meta={"session_ids": [str(i) for i in ids]})
    await db.commit()
    return {"project_id": str(p.project_id), "moved": int(getattr(res, "rowcount", 0) or 0)}
