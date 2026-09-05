"""Which paths are image subresources, and the cookie that authenticates them.

An <img src> cannot set an Authorization header. Under the fail-closed read gate that left every frame
image, object crop, and mask overlay in the app returning 401, so the canvas, the filmstrip, the search grid,
and the review thumbnails all rendered blank while the JSON around them loaded fine.

The fix is a second, deliberately weak credential: a short-lived media-scoped token in a cookie, which the
browser attaches by itself. Because it is sent automatically it reaches proxies and access logs far more
readily than a header does, so the set of things it can buy is kept as small as possible: GET on the paths
below, and nothing else. It lives here rather than in main.py because the auth middleware and the per-route
dependency both have to agree on that set, and two copies of a security boundary drift.

Matched end to end rather than by prefix, so /api/frames/{id}/objects does not fall in alongside
/api/frames/{id}/image.
"""

from __future__ import annotations

import re

MEDIA_COOKIE = "lbx_media"

_MEDIA_READ_PATTERNS = (
    re.compile(r"^/api/frames/[^/]+/image$"),
    re.compile(r"^/api/frames/[^/]+/segment/(labelids|panoptic)\.png$"),
    re.compile(r"^/api/frames/[^/]+/segment/overlay$"),
    re.compile(r"^/api/objects/[^/]+/crop$"),
    re.compile(r"^/api/predictions/[^/]+/crop$"),
    # An asset's bytes, for the document and audio editors. Same reason as the frame image above: the
    # editors render an <img>/<audio> pointing at this, and neither can set a header.
    re.compile(r"^/api/assets/[^/]+/media$"),
)


def is_media_read(path: str) -> bool:
    """True for an image subresource, which may authenticate with the media cookie instead of a header."""
    return any(p.match(path) for p in _MEDIA_READ_PATTERNS)
