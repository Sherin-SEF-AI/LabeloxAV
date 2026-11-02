"""A database that cannot hand out a connection is a 503, not a 401.

The identity lookup used to swallow every exception as "no such user", and the web client answers a 401 by
signing the person out. So a database at its connection limit signed out everyone with a perfectly good
token; it surfaced as two team-invite tests failing only in the full suite, where the limit is reached.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from db.session import database_unavailable

pytestmark = pytest.mark.db


def test_connection_failures_are_told_apart_from_a_bad_token():
    import asyncpg

    assert database_unavailable(OperationalError("select", {}, ConnectionError("too many clients")))
    assert database_unavailable(asyncpg.exceptions.TooManyConnectionsError("sorry, too many clients already"))
    assert database_unavailable(ConnectionRefusedError())
    assert not database_unavailable(ValueError("badly formed hexadecimal UUID string"))
    assert not database_unavailable(KeyError("uid"))


def test_a_good_token_during_an_outage_is_asked_to_retry_not_signed_out(monkeypatch):
    from _authutil import auth_headers
    from sqlalchemy.ext.asyncio import AsyncSession

    from services.api.main import app

    headers = auth_headers("admin")

    async def _down(self, *a, **k):
        raise OperationalError("select", {}, ConnectionError("sorry, too many clients already"))

    with TestClient(app) as c:
        monkeypatch.setattr(AsyncSession, "get", _down)
        r = c.get("/api/users", headers=headers)
        monkeypatch.undo()
    assert r.status_code == 503, r.text
    assert r.headers.get("retry-after") == "5"


def test_a_token_for_nobody_is_still_a_401():
    from _authutil import token_for

    from services.api.main import app

    with TestClient(app) as c:
        nobody = token_for("00000000-0000-0000-0000-000000000000")
        r = c.get("/api/users", headers={"Authorization": f"Bearer {nobody}"})
    assert r.status_code == 401
