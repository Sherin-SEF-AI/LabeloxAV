"""Multi-user: list/create users and issue signed API tokens. Creating a user returns that user's Bearer
token once (the only time the plaintext is shown); the web client stores it and sends it as
'Authorization: Bearer <token>'. Tokens are unforgeable (HMAC over the server signing key), and creating a
user is itself role-gated, so a token can only originate from an already-authorized admin (C1)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.models import Review, User
from services.activity import record_activity
from services.api.auth_token import mint_token
from services.api.deps import UserCreateIn, db_session, require_role

router = APIRouter()

_ROLES = {"admin", "reviewer", "annotator"}


def _token_for(user: User) -> str:
    s = get_settings()
    return mint_token(user.user_id, s.auth.signing_key, token_version=user.token_version,
                      ttl_seconds=s.auth.token_ttl_seconds)


async def _with_counts(db: AsyncSession, users: list[User]) -> list[dict]:
    """Attach each user's review count.

    The aggregation is scoped to the users being returned. It used to GROUP BY over the entire review table,
    which on a corpus with hundreds of thousands of reviews is a full scan run on every page mount of the web
    shell, to produce counts for at most a screenful of users.
    """
    if not users:
        return []
    ids = [u.user_id for u in users]
    counts = dict((await db.execute(
        select(Review.user_id, func.count()).where(Review.user_id.in_(ids)).group_by(Review.user_id)
    )).all())
    return [{"user_id": str(u.user_id), "name": u.name, "role": u.role, "reviews": int(counts.get(u.user_id, 0))}
            for u in users]


@router.get("/users")
async def list_users(limit: int = Query(200, ge=1, le=500), offset: int = Query(0, ge=0),
                     db: AsyncSession = Depends(db_session)):
    """Paginated, and bounded by a ceiling the caller cannot raise. The list was previously unbounded, so it
    grew without limit as the team did and a caller could not page through it.

    Newest first: on a paginated admin list the recently added users are the ones an operator is looking for,
    and oldest-first would bury a just-created account on the last page.
    """
    users = (await db.execute(
        select(User).order_by(User.created_at.desc()).offset(offset).limit(limit))).scalars().all()
    total = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    rows = await _with_counts(db, users)
    return {"total": int(total), "offset": offset, "limit": limit, "users": rows}


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
    """Re-issue a user's Bearer token (e.g. after a lost credential). Admin-only. To also invalidate the old
    token, call POST /users/{id}/revoke-tokens first, which bumps the user's token_version."""
    from uuid import UUID

    try:
        u = await db.get(User, UUID(user_id))
    except ValueError:
        raise HTTPException(400, "invalid user id") from None
    if u is None:
        raise HTTPException(404, "user not found")
    # Minting a working credential for another account is the most impersonation-shaped action in the API
    # and it left no trace at all - the sign-in path records `signed_in`, this recorded nothing, so a token
    # issued for someone else was indistinguishable from that person working. Records the admin as actor
    # and the recipient as subject.
    await record_activity(db, user=_admin, verb="reissued_token", subject_type="user",
                          subject_id=str(u.user_id),
                          summary=f"{getattr(_admin, 'name', 'an admin')} re-issued a token for {u.name}")
    return {"user_id": str(u.user_id), "name": u.name, "role": u.role, "token": _token_for(u)}


@router.post("/users/{user_id}/revoke-tokens")
async def revoke_tokens(user_id: str, _admin=Depends(require_role("admin")),
                        db: AsyncSession = Depends(db_session)) -> dict:
    """Revoke every outstanding token for a user by incrementing its token_version. The next request carrying
    an old token fails verification and the user must obtain a fresh one. Per-user, no session store."""
    from uuid import UUID

    try:
        u = await db.get(User, UUID(user_id))
    except ValueError:
        raise HTTPException(400, "invalid user id") from None
    if u is None:
        raise HTTPException(404, "user not found")
    u.token_version = (u.token_version or 1) + 1
    await record_activity(db, user=_admin, verb="revoked_tokens", subject_type="user",
                          subject_id=str(u.user_id),
                          summary=f"{getattr(_admin, 'name', 'an admin')} revoked every token for {u.name}",
                          commit=False)
    await db.commit()
    return {"user_id": str(u.user_id), "token_version": u.token_version, "revoked": True}
