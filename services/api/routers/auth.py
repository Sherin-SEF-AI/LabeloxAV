"""Auth helpers for the web app. Currently just dev-login.

Deny-by-default auth (services/api/main.py) means a fresh browser with an empty localStorage cannot call the
gated read routes the shell loads on mount (/api/users, /api/datasets), so the app opens to a wall of 401s
with no way to sign in: the UserPicker itself reads /api/users. dev-login breaks that bootstrap deadlock by
handing the browser a real signed admin token.

It is hard-gated to `env == "local"`. That is the same boundary `_require_prod_secrets` uses to refuse the
weak default signing key, so a real deployment (env != local) both rejects this route AND would have a strong
key that makes a self-minted token impossible anyway. The `auth.dev_login` flag only lets you turn it off
locally to exercise the real login path; it cannot turn it on anywhere it is not already allowed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import User
from services.api.auth_token import mint_token
from services.api.deps import db_session

router = APIRouter()
log = get_logger("auth")


@router.post("/auth/dev-login")
async def dev_login(db: AsyncSession = Depends(db_session)) -> dict:
    """Return a signed admin token for local development. 404 anywhere it is not permitted, so its existence
    is not even revealed off a dev box."""
    settings = get_settings()
    if settings.env != "local" or not settings.auth.dev_login:
        # 404, not 403: an endpoint that mints admin tokens should be invisible where it is not allowed.
        raise HTTPException(404, "not found")

    admin = (await db.execute(
        select(User).where(User.role == "admin").order_by(User.created_at).limit(1)
    )).scalars().first()
    if admin is None:
        # A brand-new local database has no users yet; seed the first admin so dev-login always resolves.
        admin = User(name="dev-admin", role="admin")
        db.add(admin)
        await db.commit()
        await db.refresh(admin)
        log.info("auth.dev_login_seeded_admin", user=str(admin.user_id))

    return {"user_id": str(admin.user_id), "name": admin.name, "role": admin.role,
            "token": mint_token(admin.user_id, settings.auth.signing_key)}
