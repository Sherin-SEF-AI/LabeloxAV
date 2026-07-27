"""Shared auth helper for the HTTP tests, so the suite exercises the production posture (auth on) by default.

When auth is enabled, seed a user of the requested role and return a signed Bearer header a TestClient can
carry on every request. When auth is disabled it returns an empty dict, so the same call site works under
either posture. The token layer is stateless (services/api/auth_token) and the seeded user's token_version
defaults to 1, matching a freshly minted token, so revocation checks pass."""
from __future__ import annotations

import asyncio
import uuid


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def seed_user(role: str = "admin") -> str:
    """Create a user of the given role and return its id. Runs in its own loop and clears the engine cache
    around it, matching the run_async pattern the HTTP tests use at loop boundaries."""

    async def _seed() -> str:
        from db.models import User
        from db.session import get_sessionmaker

        u = User(name=f"http-{role}-{uuid.uuid4().hex[:8]}", role=role)
        async with get_sessionmaker()() as db:
            db.add(u)
            await db.commit()
            return str(u.user_id)

    _clear_db_cache()
    try:
        return asyncio.run(_seed())
    finally:
        _clear_db_cache()


def token_for(user_id: str) -> str:
    from core.config import get_settings
    from services.api.auth_token import mint_token

    return mint_token(user_id, get_settings().auth.signing_key)


def auth_headers(role: str = "admin") -> dict[str, str]:
    """A Bearer header for a freshly seeded user of `role`, or {} when auth is disabled."""
    from core.config import get_settings

    if not get_settings().auth.enabled:
        return {}
    return {"Authorization": f"Bearer {token_for(seed_user(role))}"}


def headers_for(user_id: str) -> dict[str, str]:
    """A Bearer header for a specific existing user (attribution tests that assert who acted), or the legacy
    plaintext id header when auth is disabled so those tests still resolve an identity."""
    from core.config import get_settings

    if not get_settings().auth.enabled:
        return {"X-Lbx-User-Id": user_id}
    return {"Authorization": f"Bearer {token_for(user_id)}"}
