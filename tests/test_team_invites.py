"""Adding a teammate: an administrator invites them, they open a link once, choose a password and sign in.

It replaces handing a new person a raw API token. The link works once, expires, and leaves them with an
ordinary password sign-in of their own; issuing one is recorded with the administrator as actor.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from db.models import ActivityEvent
from db.session import get_sessionmaker

pytestmark = pytest.mark.db

PASSWORD = "correct-horse-battery-staple-91"


def _client():
    from services.api.main import app

    return TestClient(app)


def _name() -> str:
    return f"teammate-{uuid.uuid4().hex[:8]}"


def test_an_invited_person_chooses_a_password_and_signs_in():
    from _authutil import auth_headers

    name = _name()
    with _client() as c:
        r = c.post("/api/users/invite", json={"name": name, "role": "reviewer"}, headers=auth_headers("admin"))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["role"] == "reviewer" and body["expires_in_hours"] == 72 and body["invite_token"]

        # The link: choose a password, which signs them in.
        r = c.post("/api/auth/password/reset", json={"token": body["invite_token"], "password": PASSWORD})
        assert r.status_code == 200, r.text
        assert r.json().get("token")

        # It works once.
        again = c.post("/api/auth/password/reset", json={"token": body["invite_token"], "password": PASSWORD + "x"})
        assert again.status_code == 400

        # From now on, an ordinary password sign-in.
        r = c.post("/api/auth/login", json={"name": name, "password": PASSWORD})
        assert r.status_code == 200, r.text


def test_a_lost_password_gets_a_fresh_link():
    from _authutil import auth_headers

    name = _name()
    with _client() as c:
        uid = c.post("/api/users/invite", json={"name": name, "role": "annotator"},
                     headers=auth_headers("admin")).json()["user_id"]
        r = c.post(f"/api/users/{uid}/sign-in-link", headers=auth_headers("admin"))
        assert r.status_code == 200 and r.json()["invite_token"]
        assert c.post("/api/auth/password/reset",
                      json={"token": r.json()["invite_token"], "password": PASSWORD}).status_code == 200


def test_issuing_a_link_is_recorded():
    import asyncio

    from _authutil import auth_headers

    from db.session import get_engine

    name = _name()
    with _client() as c:
        uid = c.post("/api/users/invite", json={"name": name, "role": "annotator"},
                     headers=auth_headers("admin")).json()["user_id"]

    async def verbs():
        async with get_sessionmaker()() as db:
            return (await db.execute(select(ActivityEvent.verb)
                                     .where(ActivityEvent.subject_id == uid))).scalars().all()

    # The client ran on its own loop; a fresh engine for this one.
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    assert "invited_user" in asyncio.run(verbs())


def test_only_an_administrator_can_invite_or_issue_links():
    from _authutil import auth_headers

    with _client() as c:
        assert c.post("/api/users/invite", json={"name": _name(), "role": "annotator"},
                      headers=auth_headers("reviewer")).status_code == 403
        assert c.post(f"/api/users/{uuid.uuid4()}/sign-in-link",
                      headers=auth_headers("reviewer")).status_code == 403


def test_a_name_already_on_the_team_is_refused():
    from _authutil import auth_headers

    name = _name()
    with _client() as c:
        assert c.post("/api/users/invite", json={"name": name, "role": "annotator"},
                      headers=auth_headers("admin")).status_code == 200
        assert c.post("/api/users/invite", json={"name": name, "role": "annotator"},
                      headers=auth_headers("admin")).status_code == 409


def test_an_unknown_person_has_no_link():
    from _authutil import auth_headers

    with _client() as c:
        assert c.post(f"/api/users/{uuid.uuid4()}/sign-in-link", headers=auth_headers("admin")).status_code == 404


def test_the_server_side_link_lets_a_locked_out_admin_back_in(capsys):
    """How the first administrator gets in after install, and how a locked-out one gets back."""
    import argparse
    import asyncio

    from db.session import get_engine
    from scripts.sign_in_link import _run

    name = _name()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    rc = asyncio.run(_run(argparse.Namespace(name=name, create=True, role="admin", token_only=False,
                                             url="http://lab.example:3000/")))
    assert rc == 0
    # The installer picks the line carrying the link; logs may share stdout inside the test process.
    link = next(ln for ln in capsys.readouterr().out.splitlines() if "/login?invite=" in ln)
    assert link.startswith("http://lab.example:3000/login?invite=")

    from urllib.parse import unquote

    token = unquote(link.split("invite=", 1)[1])
    with _client() as c:
        r = c.post("/api/auth/password/reset", json={"token": token, "password": PASSWORD})
        assert r.status_code == 200 and r.json()["role"] == "admin"


def test_a_link_for_nobody_is_refused(capsys):
    import argparse
    import asyncio

    from db.session import get_engine
    from scripts.sign_in_link import _run

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    rc = asyncio.run(_run(argparse.Namespace(name=_name(), create=False, role="admin", token_only=False,
                                             url="http://x")))
    assert rc == 1
