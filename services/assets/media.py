"""Serving an asset's bytes, without turning an asset id into an arbitrary file read.

The asset annotation surface passed `Asset.uri` straight to an `<img src>`. Every asset in this corpus
carries a `file://` uri, and a browser will never load `file://` from an http page, so the document and
audio editors could not display a single one of their 217 assets. The console said so plainly:

    Not allowed to load local resource: file:///home/jo/.local/share/drivelab/frames/1/1.jpg

Frames already solved this: `GET /api/frames/{id}/image` reads the bytes server-side and streams them.
Assets had no equivalent, so the page had nothing to point at but the raw path.

**The reason this is not simply "open the uri".** `Asset.uri` is caller-supplied. `services/assets/store.py`
takes whatever an importer puts in it and validates only that one of uri, text or frame_id is present. An
endpoint that opened any path in that column would let anyone who can create an asset read any file the
API process can, addressed by an asset id, through an authenticated route that looks entirely ordinary.

So a local path is served only from a directory an operator has explicitly allowed, the resolved path is
checked to be inside it after following symlinks, and the default allowlist is empty. A deployment whose
media lives in object storage never serves a local file at all.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from core.config import get_settings
from core.logging import get_logger

log = get_logger("assets.media")


class MediaError(Exception):
    """The bytes cannot be served, with a reason fit to show a caller."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def allowed_roots() -> list[Path]:
    """The directories local media may be read from, resolved and existing."""
    out = []
    for raw in get_settings().paths.media_roots:
        if not raw:
            continue
        try:
            p = Path(raw).expanduser().resolve(strict=False)
        except Exception:  # noqa: BLE001 - a malformed root is a configuration fault, not a request fault
            log.warning("assets.media.bad_root", root=raw)
            continue
        out.append(p)
    return out


def resolve_local(uri: str) -> Path:
    """The file a `file://` uri names, if an operator has allowed the directory it sits in.

    Resolved with symlinks followed BEFORE the containment check. Checking the string first and opening
    the link afterwards is the classic way this kind of guard is defeated: a symlink inside an allowed
    directory pointing at /etc is a path that passes a prefix test and reads something else entirely.
    """
    parsed = urlparse(uri)
    # `file://somehost/etc/passwd` names a file on `somehost`. Dropping the host and reading the local
    # path of the same name would answer a question nobody asked, so a host other than localhost is
    # refused outright. An empty host is the ordinary `file:///path` spelling.
    if parsed.netloc not in ("", "localhost"):
        raise MediaError(f"this asset names a file on another host ({parsed.netloc}), which this server "
                         "will not fetch", 400)
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise MediaError("this asset's uri is not an absolute path", 400)

    roots = allowed_roots()
    if not roots:
        raise MediaError(
            "this asset points at a local file and no media roots are configured, so the server will not "
            "read it. Set LBX_PATHS__MEDIA_ROOTS to the directories that may be served, or re-import the "
            "asset into object storage.", 501)

    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if not resolved.is_file():
            raise MediaError("the file this asset points at is missing", 404)
        return resolved
    raise MediaError("this asset's file is outside every configured media root", 403)


def read_asset_bytes(uri: str) -> tuple[bytes, str]:
    """The asset's bytes and a content type, from object storage or from an allowed local directory."""
    if not uri:
        raise MediaError("this asset has no uri", 404)
    scheme = urlparse(uri).scheme

    if scheme in ("s3", ""):
        from core.storage import get_object_store

        try:
            data = get_object_store().get_bytes(uri)
        except Exception as exc:  # noqa: BLE001
            raise MediaError(f"could not read this asset from object storage: {str(exc)[:160]}", 404) from exc
        return data, _content_type(uri)

    if scheme == "file":
        path = resolve_local(uri)
        try:
            return path.read_bytes(), _content_type(str(path))
        except OSError as exc:
            raise MediaError(f"could not read {path.name}: {exc.strerror}", 404) from exc

    # http(s) is refused deliberately rather than proxied: fetching a caller-supplied URL from inside the
    # API is a request-forgery primitive, and an asset uri is caller-supplied.
    raise MediaError(f"'{scheme}' is not a scheme this server will fetch", 400)


def _content_type(name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"
