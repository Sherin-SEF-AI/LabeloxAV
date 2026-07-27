"""Plate watchlist matching.

A security deployment watches for specific registration marks (a stolen-vehicle list, an access allow-list).
Matching is on the normalised plate string, so "KA 01 AB 1234", "ka-01-ab-1234" and "KA01AB1234" all match the
same watchlist entry regardless of how either side was written. The watchlist is supplied by the caller (it is
deployment state, not baked in); this module only decides membership.
"""

from __future__ import annotations

from collections.abc import Iterable

from services.anpr.india_format import normalize_plate
from services.anpr.recognize import PlateRead


def normalize_watchlist(entries: Iterable[str]) -> set[str]:
    """Normalise a watchlist once (uppercase, strip separators) for repeated membership tests."""
    return {n for n in (normalize_plate(e) for e in entries) if n}


def match(read: PlateRead, watchlist: Iterable[str] | set[str]) -> str | None:
    """The normalised plate if it is on the watchlist, else None. Accepts a raw iterable or a set already
    normalised by normalize_watchlist."""
    norm = read.parse.normalized
    if not norm:
        return None
    wl = watchlist if isinstance(watchlist, set) else normalize_watchlist(watchlist)
    return norm if norm in wl else None
