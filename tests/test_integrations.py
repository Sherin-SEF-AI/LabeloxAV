"""Webhooks, storage sources, and the SDK/CLI surface.

The signature scheme is the part worth testing hardest: an unsigned or wrongly-signed delivery is an
unauthenticated write into whatever the receiver triggers.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.integrations.webhooks import EVENTS, sign, verify

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


# ---- signing --------------------------------------------------------------------------------------------

def test_signature_round_trip_and_tamper_detection():
    body = b'{"event":"job.submitted","data":{"job_id":"x"}}'
    sig = sign("s3cret", body)
    assert sig.startswith("sha256=")
    assert verify("s3cret", body, sig)

    # any tamper with the body or the key must fail
    assert not verify("s3cret", body + b" ", sig)
    assert not verify("other", body, sig)
    assert not verify("s3cret", body, "sha256=deadbeef")
    # an absent signature must never pass
    assert not verify("s3cret", body, "")
    assert not verify("s3cret", body, None)  # type: ignore[arg-type]


# ---- webhooks -------------------------------------------------------------------------------------------

@requires_infra
def test_webhook_create_validates_and_returns_secret_once():
    from db.session import get_sessionmaker
    from services.integrations.webhooks import create_webhook, delete_webhook, list_webhooks

    async def run():
        wid = None
        try:
            async with get_sessionmaker()() as db:
                with pytest.raises(ValueError):
                    await create_webhook(db, url="ftp://nope.example")     # not http(s)
                with pytest.raises(ValueError):
                    await create_webhook(db, url="https://x.example", events=["not.an.event"])

                wh = await create_webhook(db, url="https://x.example/hook", events=["job.submitted"])
                wid = wh["webhook_id"]
                # the receiver needs the secret to verify deliveries, so it is returned at creation
                assert wh["secret"] and len(wh["secret"]) >= 32
                assert wh["active"] is True and wh["failure_count"] == 0

                listed = await list_webhooks(db)
                mine = [w for w in listed if w["webhook_id"] == wid]
                assert mine, "the webhook is listed"
                # ...and the secret is NOT echoed back on the listing
                assert "secret" not in mine[0]
        finally:
            if wid:
                async with get_sessionmaker()() as db:
                    await delete_webhook(db, wid)

    run_async(run())


@requires_infra
def test_emit_only_matches_subscribed_events():
    """A subscription with an explicit event list must not receive everything else."""
    from db.session import get_sessionmaker
    from services.integrations.webhooks import create_webhook, delete_webhook, emit

    async def run():
        ids = []
        try:
            async with get_sessionmaker()() as db:
                # url is unroutable on purpose: emit must not block or raise on a dead endpoint
                a = await create_webhook(db, url="http://127.0.0.1:9/hook", events=["job.submitted"])
                ids.append(a["webhook_id"])

            n = await emit("job.submitted", {"job_id": "x"})
            assert n >= 1, "a matching subscription is delivered to"

            n2 = await emit("issue.opened", {"issue_id": "y"})
            # our subscription only wants job.submitted, so it must not be counted here
            assert n2 == 0 or n2 < n

            assert await emit("not.a.real.event", {}) == 0
            # let the detached delivery tasks finish so they do not leak into another test
            await asyncio.sleep(0.2)
        finally:
            async with get_sessionmaker()() as db:
                for wid in ids:
                    await delete_webhook(db, wid)

    run_async(run())


def test_every_documented_event_is_known():
    """The router advertises this list; a typo here would let a caller subscribe to something never emitted."""
    assert "job.submitted" in EVENTS and "drift.breached" in EVENTS
    assert len(set(EVENTS)) == len(EVENTS), "no duplicate event names"
    for e in EVENTS:
        assert "." in e, f"{e} should be namespaced like noun.verb"


# ---- storage sources ------------------------------------------------------------------------------------

@requires_infra
def test_storage_source_registers_without_credentials():
    from db.session import get_sessionmaker
    from services.integrations.storage_sources import (
        SourceError,
        delete_source,
        list_sources,
        register_source,
    )

    async def run():
        sid = None
        try:
            async with get_sessionmaker()() as db:
                with pytest.raises(SourceError):
                    await register_source(db, name="x", provider="dropbox", bucket="b")

                s = await register_source(db, name=f"src-{uuid.uuid4().hex[:6]}", provider="s3",
                                          bucket="my-bucket", prefix="drives/2026/",
                                          credential_profile="fleet-reader")
                sid = s["source_id"]
                assert s["uri"] == "s3://my-bucket/drives/2026/"
                # the row must never carry keys, only the name of a server-side profile
                assert s["credential_profile"] == "fleet-reader"
                assert "secret" not in s and "access_key" not in s

                listed = await list_sources(db)
                assert any(x["source_id"] == sid for x in listed)
        finally:
            if sid:
                async with get_sessionmaker()() as db:
                    await delete_source(db, sid)

    run_async(run())


# ---- sdk / cli ------------------------------------------------------------------------------------------

def test_sdk_builds_auth_headers_correctly():
    from sdk.labelox_client import Labelox

    with Labelox("http://x.example", token="lbx1.abc") as c:
        assert c._headers["Authorization"] == "Bearer lbx1.abc"
        assert "X-Lbx-User-Id" not in c._headers

    with Labelox("http://x.example/", user_id="u-1") as c:
        # legacy header only, and the trailing slash is normalized so paths do not double up
        assert c._headers["X-Lbx-User-Id"] == "u-1"
        assert "Authorization" not in c._headers
        assert c.base == "http://x.example"


def test_cli_exposes_the_documented_commands():
    from sdk.cli import cli

    names = set(cli.commands)
    for expected in ("health", "projects", "facets", "tag", "import", "export", "scorecards"):
        assert expected in names, f"cli is missing {expected} (has {sorted(names)})"
