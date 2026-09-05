"""Image subresources authenticate with a media cookie, and that cookie can buy nothing else.

An <img src> cannot set an Authorization header. When reads went deny-by-default, every frame image, object
crop, and mask overlay in the app started returning 401, so the editor canvas, the filmstrip, the search grid
and the review thumbnails rendered blank while the JSON around them loaded fine. The fix is a second,
deliberately weak credential the browser attaches by itself.

The whole value of that fix rests on the credential being weak, so most of what is tested here is what the
media token cannot do: it is refused as a Bearer, refused on a data route, refused on a write, and it expires
in minutes. A media cookie that could act as the user would be strictly worse than the bug it fixes, because
a cookie is sent automatically to URLs that reach proxies and access logs.
"""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from services.api.auth_token import (
    MEDIA_SCOPE,
    bearer_payload,
    mint_media_token,
    mint_token,
    verify_token,
)
from services.api.media import MEDIA_COOKIE, is_media_read

KEY = "test-signing-key-not-a-real-one"


# ---------------------------------------------------------------- the token itself

def test_a_media_token_verifies_and_carries_its_scope():
    uid = str(uuid.uuid4())
    p = verify_token(mint_media_token(uid, KEY, token_version=3), KEY)
    assert p is not None
    assert p.uid == uid
    assert p.scope == MEDIA_SCOPE
    assert p.token_version == 3


def test_a_session_token_has_no_scope():
    """The absent claim must read as an ordinary session token, not as a broader one. Every token minted
    before media tokens existed looks like this."""
    p = verify_token(mint_token(str(uuid.uuid4()), KEY), KEY)
    assert p is not None and p.scope is None


def test_a_media_token_is_refused_as_a_bearer():
    """The restriction has to hold at the header too. If presenting the cookie as a Bearer worked, anything
    that could read the cookie could act as the user in full, and the scoping would buy nothing."""
    tok = mint_media_token(str(uuid.uuid4()), KEY)
    assert bearer_payload(f"Bearer {tok}", KEY) is None
    # ... while the ordinary token still works through the same path.
    assert bearer_payload(f"Bearer {mint_token(str(uuid.uuid4()), KEY)}", KEY) is not None


def test_a_media_token_expires():
    tok = mint_media_token(str(uuid.uuid4()), KEY, ttl_seconds=60, now=int(time.time()) - 3600)
    assert verify_token(tok, KEY) is None


def test_a_media_token_signed_with_another_key_is_refused():
    assert verify_token(mint_media_token(str(uuid.uuid4()), KEY), "a-different-key") is None


# ---------------------------------------------------------------- which paths it covers

def test_the_media_path_set_is_the_image_routes_and_nothing_adjacent():
    fid, oid = uuid.uuid4(), uuid.uuid4()
    assert is_media_read(f"/api/frames/{fid}/image")
    assert is_media_read(f"/api/objects/{oid}/crop")
    assert is_media_read(f"/api/predictions/{oid}/crop")
    assert is_media_read(f"/api/frames/{fid}/segment/labelids.png")
    assert is_media_read(f"/api/frames/{fid}/segment/overlay")
    # An asset's bytes: the document and audio editors render an <img>/<audio> at this, and neither can
    # set a header. Added after that endpoint shipped without it and answered 401 to every editor open.
    assert is_media_read(f"/api/assets/{uuid.uuid4()}/media")

    # Matched end to end, so a data route sharing the prefix does not come along with it.
    assert not is_media_read(f"/api/frames/{fid}/objects")
    assert not is_media_read(f"/api/frames/{fid}/image/../objects")
    assert not is_media_read(f"/api/objects/{oid}")
    assert not is_media_read("/api/sessions")
    assert not is_media_read(f"/api/frames/{fid}/segment")
    # The asset routes that are data, not media, must not come with it: annotations are writable and the
    # media cookie is a credential the browser attaches by itself.
    assert not is_media_read(f"/api/assets/{oid}/annotations")
    assert not is_media_read(f"/api/assets/{oid}")
    assert not is_media_read(f"/api/assets/{oid}/media/../annotations")


# ---------------------------------------------------------------- over HTTP
# Only this section needs a database. The token and path-matching tests above are pure units and must stay
# runnable in the no-Postgres tier.

def _client() -> TestClient:
    from _authutil import _clear_db_cache

    from services.api.main import app

    _clear_db_cache()
    return TestClient(app)


def _media_cookie_for(role: str = "annotator") -> tuple[str, str]:
    """A real media cookie, obtained the way the browser does: sign in, then downgrade."""
    from _authutil import auth_headers

    h = auth_headers(role)
    with _client() as c:
        r = c.post("/api/auth/media-token", headers=h)
        assert r.status_code == 200, r.text
        return r.json()["token"], h["Authorization"]


@pytest.mark.db
def test_minting_a_media_token_needs_a_session_token():
    """It downgrades access the caller already holds. It is not a way to obtain access."""
    with _client() as c:
        assert c.post("/api/auth/media-token").status_code == 401


@pytest.mark.db
def test_an_image_request_with_no_credential_is_refused():
    with _client() as c:
        r = c.get(f"/api/frames/{uuid.uuid4()}/image")
    assert r.status_code == 401


@pytest.mark.db
def test_the_media_cookie_authenticates_an_image_request():
    """404 rather than 401 is the pass condition: the request cleared the auth gate and reached the handler,
    which then could not find that frame. A 401 would mean the cookie was not honoured."""
    tok, _ = _media_cookie_for()
    with _client() as c:
        c.cookies.set(MEDIA_COOKIE, tok)
        r = c.get(f"/api/frames/{uuid.uuid4()}/image")
    assert r.status_code == 404


@pytest.mark.db
def test_the_media_cookie_reaches_a_route_behind_its_own_role_floor():
    """/predictions is a reviewer-gated router, so the cookie has to satisfy the route dependency as well as
    the middleware. It was the case that broke when only the middleware knew about the cookie."""
    tok, _ = _media_cookie_for("reviewer")
    with _client() as c:
        c.cookies.set(MEDIA_COOKIE, tok)
        r = c.get(f"/api/predictions/{uuid.uuid4()}/crop")
    assert r.status_code == 404


@pytest.mark.db
def test_the_media_cookie_does_not_open_a_data_route():
    """The point of the whole design. The browser sends this cookie on every /api request it makes, so if it
    worked anywhere but the image routes it would be a session token with extra steps."""
    tok, _ = _media_cookie_for()
    with _client() as c:
        c.cookies.set(MEDIA_COOKIE, tok)
        assert c.get("/api/sessions").status_code == 401
        assert c.get(f"/api/frames/{uuid.uuid4()}/objects").status_code == 401
        assert c.get("/api/users").status_code == 401


@pytest.mark.db
def test_the_media_cookie_does_not_authorise_a_write():
    tok, _ = _media_cookie_for("admin")
    with _client() as c:
        c.cookies.set(MEDIA_COOKIE, tok)
        r = c.post(f"/api/frames/{uuid.uuid4()}/objects",
                   json={"class_name": "pedestrian", "bbox": [0, 0, 1, 1]})
    assert r.status_code == 401


@pytest.mark.db
def test_the_media_token_is_refused_as_a_bearer_over_http():
    tok, _ = _media_cookie_for("admin")
    with _client() as c:
        assert c.get("/api/sessions", headers={"Authorization": f"Bearer {tok}"}).status_code == 401


@pytest.mark.db
def test_the_session_token_still_works_on_an_image_route():
    """The cookie is an addition, not a replacement: a scripted client with a Bearer must keep working."""
    _, authz = _media_cookie_for()
    with _client() as c:
        r = c.get(f"/api/frames/{uuid.uuid4()}/image", headers={"Authorization": authz})
    assert r.status_code == 404


@pytest.mark.db
def test_a_garbage_cookie_is_refused_rather_than_crashing():
    with _client() as c:
        c.cookies.set(MEDIA_COOKIE, "not-a-token")
        assert c.get(f"/api/frames/{uuid.uuid4()}/image").status_code == 401
        c.cookies.set(MEDIA_COOKIE, "lbx2.aaa.bbb")
        assert c.get(f"/api/frames/{uuid.uuid4()}/image").status_code == 401
