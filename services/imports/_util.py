"""Shared helpers for import adapters: locate the manifest file and resolve image references."""

from __future__ import annotations

import json
from pathlib import Path


def find_file(root: Path, *globs: str) -> Path | None:
    """First file under root matching any glob (in order). Case-insensitive on the final name."""
    for g in globs:
        hits = sorted(root.rglob(g))
        if hits:
            return hits[0]
    return None


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def resolve_image(root: Path, ref: str) -> str | Path | None:
    """Resolve an image reference, CONFINED to root. s3:// uris pass through (run.py fetches them from the
    store). A local ref (from untrusted manifest data) is matched only within root: exact relative path, then
    by basename anywhere under root. Absolute paths and any ref that escapes root via ".." are rejected, so a
    hostile ref like "/etc/passwd" or "../../secret" cannot read arbitrary files."""
    if ref.startswith("s3://"):
        return ref
    root_r = root.resolve()

    def _within(p: Path) -> Path | None:
        try:
            rp = p.resolve()
        except OSError:
            return None
        return rp if (rp == root_r or root_r in rp.parents) and rp.exists() else None

    if not Path(ref).is_absolute():
        cand = _within(root / ref)               # exact relative path, confined
        if cand is not None:
            return cand
    base = Path(ref).name                          # basename search, confined to root
    for hit in sorted(root.rglob(base)):
        w = _within(hit)
        if w is not None:
            return w
    return None
