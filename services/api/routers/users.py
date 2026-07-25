"""Multi-user: list/create users and issue signed API tokens. Creating a user returns that user's Bearer
token once (the only time the plaintext is shown); the web client stores it and sends it as
'Authorization: Bearer <token>'. Tokens are unforgeable (HMAC over the server signing key), and creating a
user is itself role-gated, so a token can only originate from an already-authorized admin (C1)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.models import Review, User
from services.api.auth_token import mint_token
from services.api.deps import UserCreateIn, db_session, require_role

router = APIRouter()

_ROLES = {"admin", "reviewer", "annotator"}


def _token_for(user: User) -> str:
    return mint_token(user.user_id, get_settings().auth.signing_key)


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


@router.get("/users/me")
async def whoami(user=Depends(require_role("annotator")), db: AsyncSession = Depends(db_session)):
    """Resolve the caller's own identity from their Bearer token. Lets the web client turn a pasted token into
    a name/role for display without trusting anything client-side."""
    rows = await _with_counts(db, [user])
    return rows[0]


@router.post("/users")
async def create_user(payload: UserCreateIn, db: AsyncSession = Depends(db_session)):
    if payload.role not in _ROLES:
        raise HTTPException(400, f"role must be one of {sorted(_ROLES)}")
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "name required")
    if (await db.execute(select(User).where(User.name == name))).scalar_one_or_none():
        raise HTTPException(409, f"user '{name}' already exists")
    # Bootstrap: the very first user (the auth middleware lets it through when the table is empty) is
    # forced to admin so the system has a manager. After that, role gating governs who can create users.
    n = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    role = "admin" if n == 0 else payload.role
    u = User(name=name, role=role)
    db.add(u)
    await db.commit()
    # Return the signed token once: this is the new user's credential, handed off by the creating admin (or
    # kept by the bootstrap admin who created the first account).
    return {"user_id": str(u.user_id), "name": u.name, "role": u.role, "reviews": 0, "token": _token_for(u)}


@router.post("/users/{user_id}/token")
async def reissue_token(user_id: str, _admin=Depends(require_role("admin")),
                        db: AsyncSession = Depends(db_session)):
    """Re-issue a user's Bearer token (e.g. after a lost credential). Admin-only. Because tokens are stateless,
    this does not revoke the old one; rotate the server signing key to invalidate all outstanding tokens."""
    from uuid import UUID

    try:
        u = await db.get(User, UUID(user_id))
    except ValueError:
        raise HTTPException(400, "invalid user id") from None
    if u is None:
        raise HTTPException(404, "user not found")
    return {"user_id": str(u.user_id), "name": u.name, "role": u.role, "token": _token_for(u)}
