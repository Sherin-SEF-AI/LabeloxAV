"""Minting, verifying and revoking machine credentials.

The key looks like `lbxk_<prefix>_<secret>`. The prefix is public, indexed, and what the lookup finds the row
by; the secret is shown exactly once at creation and stored only as a sha256, so a database someone walks off
with contains no usable credential.

Two decisions that are easy to get wrong.

**The hash is not a password hash.** bcrypt or argon2 would be right for something a human chose, because a
human password is guessable and slow hashing is what makes guessing expensive. This secret is 256 bits from
`secrets.token_urlsafe`, so there is nothing to guess and the only thing a slow hash would buy is a slow
request on the hot path of every machine call. What does matter is comparing in constant time, which sha256
plus `compare_digest` gives.

**Verification is a database read, deliberately.** A signed token would avoid the round trip, and it would
also mean a leaked key stays valid until it expires. A credential that lives in a CI config for a year has to
die the moment somebody revokes it, so the read is the point rather than a cost to engineer away.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ServiceAccount, User

log = get_logger("service_accounts")

KEY_SCHEME = "lbxk"
# Hex, not token_urlsafe, and the reason is the delimiter. `token_urlsafe` emits `_`, so a prefix could
# contain one, `lbxk_<prefix>_<secret>` would split in the wrong place, and the lookup would miss a perfectly
# valid key. It failed roughly one mint in three, at random, which is the worst way for an auth path to
# break. The secret can stay urlsafe because it is the last field and absorbs any delimiter it contains.
PREFIX_BYTES = 6
SECRET_BYTES = 32

# How stale `last_used_at` may get before another write. Stamping it on every request would put a write on
# the hot path of every machine call to answer a question ("is this key still in use?") that nobody asks to
# the second.
LAST_USED_THROTTLE = timedelta(minutes=5)


class ServiceAccountError(ValueError):
    """A credential operation that would be unsafe or meaningless."""


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def split_key(key: str) -> tuple[str, str] | None:
    """`lbxk_<prefix>_<secret>` into its two halves, or None if it is not one of ours.

    Returning None rather than raising keeps the auth path free of exception handling for the ordinary case
    of a request carrying an ordinary bearer token.
    """
    if not key or not key.startswith(f"{KEY_SCHEME}_"):
        return None
    parts = key.split("_", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


async def mint(db: AsyncSession, *, name: str, role: str = "annotator",
               description: str | None = None, created_by: uuid.UUID | None = None,
               expires_in_days: int | None = None, scopes: list[str] | None = None) -> dict:
    """Create a service account and return its key, once.

    The account owns its own `app_user` row rather than borrowing somebody's. That is what makes a machine's
    writes attributable: every `created_by` and every audit entry already keys on a user id, and this way a
    row written by CI says CI rather than naming whoever happened to generate the key.
    """
    name = (name or "").strip()
    if not name:
        raise ServiceAccountError("a service account needs a name")
    if role not in ("annotator", "reviewer", "admin"):
        raise ServiceAccountError(f"unknown role {role!r}")
    if (await db.execute(select(ServiceAccount).where(ServiceAccount.name == name))).scalar_one_or_none():
        raise ServiceAccountError(f"a service account named {name!r} already exists")

    # Distinguishable from a person in every list and audit view, without needing a separate column on User.
    user_name = f"svc:{name}"
    if (await db.execute(select(User).where(User.name == user_name))).scalar_one_or_none():
        raise ServiceAccountError(f"the identity {user_name!r} is taken")

    user = User(user_id=uuid.uuid4(), name=user_name, role=role)
    db.add(user)
    await db.flush()

    prefix = secrets.token_hex(PREFIX_BYTES)
    secret = secrets.token_urlsafe(SECRET_BYTES)
    acct = ServiceAccount(
        service_account_id=uuid.uuid4(), name=name, user_id=user.user_id,
        key_prefix=prefix, key_hash=_hash(secret), scopes=list(scopes or []),
        description=description, created_by=created_by,
        expires_at=(_now() + timedelta(days=expires_in_days)) if expires_in_days else None)
    db.add(acct)
    await db.commit()

    log.info("service_account.minted", name=name, role=role, prefix=prefix,
             expires_in_days=expires_in_days)
    return {
        "service_account_id": str(acct.service_account_id),
        "name": name, "role": role, "user_id": str(user.user_id), "key_prefix": prefix,
        # The only time this is ever returned. There is no path that can show it again.
        "api_key": f"{KEY_SCHEME}_{prefix}_{secret}",
        "expires_at": acct.expires_at.isoformat() if acct.expires_at else None,
    }


async def verify(db: AsyncSession, key: str) -> User | None:
    """The user a key acts as, or None if the key is unknown, revoked, or expired.

    None for every failure rather than a reason: an auth path that explains which half of a credential was
    wrong is a probe someone can use to enumerate valid prefixes.
    """
    split = split_key(key)
    if split is None:
        return None
    prefix, secret = split

    acct = (await db.execute(
        select(ServiceAccount).where(ServiceAccount.key_prefix == prefix))).scalar_one_or_none()
    if acct is None:
        return None
    if not hmac.compare_digest(acct.key_hash, _hash(secret)):
        return None
    if acct.revoked_at is not None:
        return None
    if acct.expires_at is not None and _expired(acct.expires_at):
        return None

    user = await db.get(User, acct.user_id)
    if user is None:
        # The identity was deleted underneath the key. Refuse rather than fall back to anything.
        return None

    last = acct.last_used_at
    if last is None or (_now() - _aware(last)) > LAST_USED_THROTTLE:
        acct.last_used_at = _now()
        await db.commit()
    return user


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _expired(dt: datetime) -> bool:
    return _now() > _aware(dt)


async def revoke(db: AsyncSession, service_account_id: uuid.UUID) -> dict | None:
    """Kill a key now. Idempotent: revoking twice keeps the first timestamp, which is the one that matters."""
    acct = await db.get(ServiceAccount, service_account_id)
    if acct is None:
        return None
    if acct.revoked_at is None:
        acct.revoked_at = _now()
        await db.commit()
        log.warning("service_account.revoked", name=acct.name, prefix=acct.key_prefix)
    return _row(acct)


async def rotate(db: AsyncSession, service_account_id: uuid.UUID) -> dict | None:
    """Issue a new secret for the same account, invalidating the old one immediately.

    Same row and same identity, so everything the account has written stays attributed to it. A leaked key is
    replaced without having to re-point whatever it authorises at a new account.
    """
    acct = await db.get(ServiceAccount, service_account_id)
    if acct is None:
        return None
    if acct.revoked_at is not None:
        raise ServiceAccountError("this account is revoked; mint a new one rather than rotating it")
    prefix = secrets.token_hex(PREFIX_BYTES)
    secret = secrets.token_urlsafe(SECRET_BYTES)
    acct.key_prefix, acct.key_hash = prefix, _hash(secret)
    acct.last_used_at = None
    await db.commit()
    log.warning("service_account.rotated", name=acct.name, prefix=prefix)
    return {**_row(acct), "api_key": f"{KEY_SCHEME}_{prefix}_{secret}"}


async def listing(db: AsyncSession, *, include_revoked: bool = False) -> list[dict]:
    q = select(ServiceAccount).order_by(ServiceAccount.created_at.desc())
    if not include_revoked:
        q = q.where(ServiceAccount.revoked_at.is_(None))
    rows = (await db.execute(q)).scalars().all()
    out = []
    for a in rows:
        user = await db.get(User, a.user_id)
        out.append({**_row(a), "role": user.role if user else None})
    return out


def _row(a: ServiceAccount) -> dict:
    return {
        "service_account_id": str(a.service_account_id), "name": a.name,
        "key_prefix": a.key_prefix, "description": a.description,
        "user_id": str(a.user_id), "scopes": list(a.scopes or []),
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "last_used_at": a.last_used_at.isoformat() if a.last_used_at else None,
        "expires_at": a.expires_at.isoformat() if a.expires_at else None,
        "revoked_at": a.revoked_at.isoformat() if a.revoked_at else None,
        "active": a.revoked_at is None and not (a.expires_at and _expired(a.expires_at)),
    }
