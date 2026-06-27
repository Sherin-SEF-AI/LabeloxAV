"""Lightweight multi-user: list/create users (no password). The web client picks the current user and
sends it as the X-Lbx-User-Id header; mutations record that user for attribution + the QA workflow."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Review, User
from services.api.deps import UserCreateIn, db_session

router = APIRouter()

_ROLES = {"admin", "reviewer", "annotator"}


async def _with_counts(db: AsyncSession, users: list[User]) -> list[dict]:
    counts = dict((await db.execute(
        select(Review.user_id, func.count()).where(Review.user_id.isnot(None)).group_by(Review.user_id)
    )).all())
    return [{"user_id": str(u.user_id), "name": u.name, "role": u.role, "reviews": int(counts.get(u.user_id, 0))}
            for u in users]


@router.get("/users")
async def list_users(db: AsyncSession = Depends(db_session)):
    users = (await db.execute(select(User).order_by(User.created_at))).scalars().all()
    return await _with_counts(db, users)


@router.post("/users")
async def create_user(payload: UserCreateIn, db: AsyncSession = Depends(db_session)):
    if payload.role not in _ROLES:
        raise HTTPException(400, f"role must be one of {sorted(_ROLES)}")
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "name required")
    if (await db.execute(select(User).where(User.name == name))).scalar_one_or_none():
        raise HTTPException(409, f"user '{name}' already exists")
    u = User(name=name, role=payload.role)
    db.add(u)
    await db.commit()
    return {"user_id": str(u.user_id), "name": u.name, "role": u.role, "reviews": 0}
