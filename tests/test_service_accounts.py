"""Machine credentials, and the properties that make one safe to leave in a CI config for a year.

Every authentication path in this system was human, so an integration had to be issued a person's
twelve-hour bearer token, and every write a machine made was recorded as that person doing it.

A service account owns its own `app_user` row on purpose. That is what keeps role floors, audit entries and
`created_by` provenance working unchanged, and what makes a row written by CI say CI.

The tests that matter most are the refusals: revoked, expired, wrong secret, and a key whose identity was
deleted underneath it. Each one is a case where returning a user would hand a caller privileges nobody meant
them to have.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from db.models import ServiceAccount, User
from db.session import get_sessionmaker
from services.identity.service_accounts import (
    ServiceAccountError,
    listing,
    mint,
    revoke,
    rotate,
    split_key,
    verify,
)

pytestmark = pytest.mark.db


def _name() -> str:
    return f"test-svc-{uuid.uuid4().hex[:10]}"


# ------------------------------------------------------------------------------- minting

async def test_minting_returns_the_key_once_and_creates_an_identity():
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name(), role="reviewer", description="ci")
        assert out["api_key"].startswith("lbxk_")
        user = await db.get(User, uuid.UUID(out["user_id"]))
        assert user is not None and user.role == "reviewer"
        assert user.name.startswith("svc:"), "a machine identity should be distinguishable in any user list"


async def test_the_secret_is_never_stored():
    """A database someone walks off with must contain no usable credential."""
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        secret = out["api_key"].split("_", 2)[2]
        acct = (await db.execute(select(ServiceAccount).where(
            ServiceAccount.key_prefix == out["key_prefix"]))).scalar_one()
        assert secret not in acct.key_hash
        assert acct.key_hash != secret
        assert len(acct.key_hash) == 64


async def test_the_key_is_not_recoverable_from_the_listing():
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        rows = await listing(db)
        row = next(r for r in rows if r["key_prefix"] == out["key_prefix"])
        assert "api_key" not in row and "key_hash" not in row


async def test_two_accounts_cannot_share_a_name():
    async with get_sessionmaker()() as db:
        n = _name()
        await mint(db, name=n)
        with pytest.raises(ServiceAccountError, match="already exists"):
            await mint(db, name=n)


async def test_an_unknown_role_is_refused():
    async with get_sessionmaker()() as db:
        with pytest.raises(ServiceAccountError, match="unknown role"):
            await mint(db, name=_name(), role="superuser")


# ------------------------------------------------------------------------------- verifying

async def test_a_valid_key_resolves_to_its_identity():
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name(), role="admin")
        user = await verify(db, out["api_key"])
        assert user is not None
        assert str(user.user_id) == out["user_id"] and user.role == "admin"


async def test_a_wrong_secret_with_a_real_prefix_is_refused():
    """The prefix is public. Only the secret half may be the thing that authorises."""
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        assert await verify(db, f"lbxk_{out['key_prefix']}_not-the-secret") is None


@pytest.mark.parametrize("bad", [
    "", "lbxk_", "lbxk_only-one-part", "not-a-key", "lbx2.abc.def", "lbxk__missing-prefix",
])
async def test_a_malformed_key_is_refused_without_raising(bad):
    """This runs on the auth path of every request, so it has to be total."""
    async with get_sessionmaker()() as db:
        assert await verify(db, bad) is None


async def test_a_revoked_key_stops_working_immediately():
    """The reason revocation is a timestamp and not a token version: a credential in a CI config has to die
    the moment somebody presses the button, not when it happens to expire."""
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        assert await verify(db, out["api_key"]) is not None
        await revoke(db, uuid.UUID(out["service_account_id"]))
        assert await verify(db, out["api_key"]) is None


async def test_revoking_twice_keeps_the_first_timestamp():
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        first = await revoke(db, uuid.UUID(out["service_account_id"]))
        second = await revoke(db, uuid.UUID(out["service_account_id"]))
        assert first["revoked_at"] == second["revoked_at"]


async def test_an_expired_key_is_refused():
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name(), expires_in_days=1)
        acct = await db.get(ServiceAccount, uuid.UUID(out["service_account_id"]))
        acct.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await db.commit()
        assert await verify(db, out["api_key"]) is None


async def test_a_key_whose_identity_was_deleted_is_refused():
    """Refuse rather than fall back to anything: the key authorises acting as a user that no longer exists."""
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        user = await db.get(User, uuid.UUID(out["user_id"]))
        await db.delete(user)
        await db.commit()
        assert await verify(db, out["api_key"]) is None


async def test_verifying_records_that_the_key_is_in_use():
    """So "which of these integrations is still live?" is answerable before revoking one."""
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        acct_id = uuid.UUID(out["service_account_id"])
        assert (await db.get(ServiceAccount, acct_id)).last_used_at is None
        await verify(db, out["api_key"])
        await db.refresh(await db.get(ServiceAccount, acct_id))
        assert (await db.get(ServiceAccount, acct_id)).last_used_at is not None


# ------------------------------------------------------------------------------- rotation

async def test_rotating_invalidates_the_old_key_and_keeps_the_identity():
    """A leaked key is replaced without re-pointing whatever it authorises at a different account, and
    everything the account has already written stays attributed to it."""
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        rotated = await rotate(db, uuid.UUID(out["service_account_id"]))
        assert await verify(db, out["api_key"]) is None
        user = await verify(db, rotated["api_key"])
        assert user is not None and str(user.user_id) == out["user_id"]


async def test_a_revoked_account_cannot_be_rotated():
    """Rotating it back to life would make revocation reversible by anyone who can rotate."""
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        await revoke(db, uuid.UUID(out["service_account_id"]))
        with pytest.raises(ServiceAccountError, match="revoked"):
            await rotate(db, uuid.UUID(out["service_account_id"]))


# ------------------------------------------------------------------------------- listing + parsing

async def test_the_listing_hides_revoked_accounts_by_default_but_can_show_them():
    async with get_sessionmaker()() as db:
        out = await mint(db, name=_name())
        await revoke(db, uuid.UUID(out["service_account_id"]))
        assert out["key_prefix"] not in {r["key_prefix"] for r in await listing(db)}
        assert out["key_prefix"] in {r["key_prefix"]
                                     for r in await listing(db, include_revoked=True)}


async def test_a_prefix_never_contains_the_delimiter():
    """The bug this caught. `token_urlsafe` emits `_`, so a prefix could contain one, the key would split in
    the wrong place, and a perfectly valid credential would be refused: roughly one mint in three, at random,
    which is the worst way for an auth path to fail. Prefixes are hex now, and every one must round-trip."""
    async with get_sessionmaker()() as db:
        for _ in range(12):
            out = await mint(db, name=_name())
            assert "_" not in out["key_prefix"] and "-" not in out["key_prefix"]
            assert split_key(out["api_key"])[0] == out["key_prefix"]
            assert await verify(db, out["api_key"]) is not None


def test_split_key_only_accepts_our_scheme():
    assert split_key("lbxk_abc_def") == ("abc", "def")
    for other in ("lbx2.a.b", "Bearer abc", "", None):
        assert split_key(other) is None


def test_a_secret_carries_real_entropy():
    """The hash is fast on purpose, so the secret has to be the thing that is unguessable."""
    import secrets as s

    from services.identity.service_accounts import SECRET_BYTES

    assert SECRET_BYTES >= 32
    assert len(s.token_urlsafe(SECRET_BYTES)) >= 40
