"""Serving an asset's bytes without turning an asset id into an arbitrary file read.

The document and audio editors pointed straight at `Asset.uri`. Every asset in this corpus carries a
`file://` uri, and a browser refuses `file://` from an http page, so the whole surface displayed nothing
and said so in the console:

    Not allowed to load local resource: file:///home/jo/.local/share/drivelab/frames/1/1.jpg

Frames have had a server-side reader since they existed. Assets had none, so the page had nothing to point
at but the raw path.

The reason this is not simply "open the uri" is that `Asset.uri` is caller-supplied: `services/assets/
store.py` takes whatever an importer puts there and checks only that one of uri, text or frame_id is
present. An endpoint that opened any path in that column would be an arbitrary file read addressed by an
asset id, through an authenticated route that looks entirely ordinary.

So the tests here are mostly about refusals, and the symlink one is the important one: checking the string
before following the link is the standard way this guard is defeated.
"""

import os
import tempfile
from pathlib import Path

import pytest

from services.assets.media import MediaError, read_asset_bytes, resolve_local


@pytest.fixture
def media_root(monkeypatch):
    """A directory the server is allowed to read, plus one it is not."""
    from core.config import get_settings

    with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as secret:
        (Path(allowed) / "page.jpg").write_bytes(b"\xff\xd8\xff-not-really-a-jpeg")
        (Path(secret) / "passwd").write_bytes(b"root:x:0:0")
        s = get_settings()
        monkeypatch.setattr(s.paths, "media_roots", [allowed], raising=False)
        get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
        monkeypatch.setattr("services.assets.media.allowed_roots", lambda: [Path(allowed).resolve()])
        yield Path(allowed), Path(secret)


def test_a_file_inside_an_allowed_root_is_served(media_root):
    allowed, _secret = media_root
    data, ct = read_asset_bytes(f"file://{allowed / 'page.jpg'}")
    assert data.startswith(b"\xff\xd8\xff")
    assert ct == "image/jpeg", "the content type comes from the name, so a browser renders it"


def test_a_file_outside_every_allowed_root_is_refused(media_root):
    """The whole point of the allowlist. An importer can put any path in Asset.uri."""
    _allowed, secret = media_root
    with pytest.raises(MediaError) as exc:
        read_asset_bytes(f"file://{secret / 'passwd'}")
    assert exc.value.status == 403
    assert "outside every configured media root" in str(exc.value)


def test_a_traversal_out_of_an_allowed_root_is_refused(media_root):
    """`/allowed/../secret/passwd` is a string that starts with the allowed root and is not inside it.
    Resolving before comparing is what catches it."""
    allowed, secret = media_root
    with pytest.raises(MediaError) as exc:
        read_asset_bytes(f"file://{allowed}/../{secret.name}/passwd")
    assert exc.value.status in (403, 404)


def test_a_symlink_out_of_an_allowed_root_is_refused(media_root):
    """The one that matters, and the way a prefix check is normally defeated.

    A symlink sitting inside the allowed directory and pointing outside it passes any test done on the
    string. The link has to be followed BEFORE the containment check, not after.
    """
    allowed, secret = media_root
    link = allowed / "innocent.jpg"
    try:
        os.symlink(secret / "passwd", link)
    except OSError:
        pytest.skip("this filesystem does not support symlinks")
    with pytest.raises(MediaError) as exc:
        read_asset_bytes(f"file://{link}")
    assert exc.value.status == 403, "a symlink escaping the root must be refused, not followed"


def test_with_no_roots_configured_nothing_local_is_served(monkeypatch):
    """The default. A deployment whose media lives in object storage should never read a local file, and
    the refusal says what to set rather than only that it failed."""
    monkeypatch.setattr("services.assets.media.allowed_roots", lambda: [])
    with pytest.raises(MediaError) as exc:
        read_asset_bytes("file:///anything/at/all.jpg")
    assert exc.value.status == 501
    assert "LBX_PATHS__MEDIA_ROOTS" in str(exc.value)


def test_a_missing_file_inside_an_allowed_root_is_a_404_not_a_403(media_root):
    """The two are different problems and lead somewhere different: one is a configuration question and
    the other is a missing import."""
    allowed, _ = media_root
    with pytest.raises(MediaError) as exc:
        resolve_local(f"file://{allowed / 'gone.jpg'}")
    assert exc.value.status == 404


def test_an_http_uri_is_refused_rather_than_fetched():
    """Fetching a caller-supplied URL from inside the API is a request-forgery primitive, and an asset uri
    is caller-supplied."""
    with pytest.raises(MediaError) as exc:
        read_asset_bytes("http://169.254.169.254/latest/meta-data/")
    assert exc.value.status == 400
    assert "not a scheme this server will fetch" in str(exc.value)


def test_an_empty_uri_says_so():
    with pytest.raises(MediaError) as exc:
        read_asset_bytes("")
    assert exc.value.status == 404


def test_a_relative_path_is_refused(media_root):
    """A relative path has no meaning without a working directory, and guessing one would make the guard
    depend on where the server happened to be started."""
    with pytest.raises(MediaError) as exc:
        resolve_local("file:relative/page.jpg")
    assert exc.value.status == 400


def test_a_file_on_another_host_is_refused_rather_than_read_locally(media_root):
    """`file://somehost/page.jpg` names a file on `somehost`. Dropping the host and reading the local path
    of the same name answers a question nobody asked, and would let a uri that looks remote reach local
    disk. This test exists because an earlier version did exactly that."""
    allowed, _ = media_root
    with pytest.raises(MediaError) as exc:
        resolve_local(f"file://somehost{allowed / 'page.jpg'}")
    assert exc.value.status == 400
    assert "another host" in str(exc.value)
