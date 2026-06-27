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
    """Resolve an image reference. s3:// uris pass through (run.py fetches them from the store).
    Local refs are matched against root: exact relative path, then by basename anywhere under root."""
    if ref.startswith("s3://"):
        return ref
    p = Path(ref)
    if p.is_absolute() and p.exists():
        return p
    cand = root / ref
    if cand.exists():
        return cand
    base = p.name
    hits = sorted(root.rglob(base))
    return hits[0] if hits else None
