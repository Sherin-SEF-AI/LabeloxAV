"""Where a drive happened, resolved from what the corpus actually recorded.

The corpus records places the way the capture rig spelled them, and the strings are not a taxonomy. This one
says `BLR` on 372 sessions and `Bengaluru` on one: a single city that looks like two strata to anything
counting regional coverage, and a stratification built on the raw strings would report the corpus as covering
two places when it covers one.

Nothing is stored. `Session.city` is already on disk and the mapping is a deterministic lookup, so a resolved
copy in a second column would only be a thing that could drift from the table it was copied out of. The one
case that would need storage - deriving a place from GNSS - applies to 1 session and 3 frames of 41,752, and
is handled by reading the coordinates when they are there.

`road_class` is the other half of the brief's stratification and it is empty: `Frame.road_class` is NULL on
all 41,752 frames. It resolves `unresolved`, and the datasheet prints that rather than a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.domain import active_pack

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalise(place: str | None) -> str:
    """Lowercase, punctuation-free, single-spaced. The form the pack's alias table is keyed on."""
    if not place:
        return ""
    return _PUNCT.sub("", place.strip().lower()).strip()


@dataclass(frozen=True)
class Region:
    """A resolved place. `status` is the part a report must not throw away."""

    city: str | None
    state: str | None
    urban_class: str | None
    # resolved | outside | unknown | absent
    #
    # Four, not three. `absent` is a session with no city recorded at all and `unknown` is one whose city
    # string the pack does not model: the first is a gap in capture and the second is a gap in the region
    # table, and a report that merges them tells whoever reads it to go and fix the wrong thing.
    status: str
    # For `outside`, the country or "mixed". Non-Indian imports are not an Indian urban stratum, and giving
    # a KITTI drive through Karlsruhe a class_1 label puts German motorway footage in an Indian bucket.
    outside: str | None = None
    raw: str | None = None

    def stratum(self) -> str:
        """One label for grouping. Never None, so a group-by cannot silently drop rows."""
        if self.status == "resolved":
            return f"{self.state}/{self.city}"
        if self.status == "outside":
            return f"outside:{self.outside}"
        return self.status


def resolve_region(city: str | None, pack_id: str | None = None) -> Region:
    """Resolve a recorded place string against the active pack's region model."""
    raw = city or None
    spec = active_pack(pack_id).region
    if spec is None:
        # A domain with no regional structure. Saying so beats inventing one.
        return Region(None, None, None, "unknown", raw=raw)
    key = normalise(city)
    if not key:
        return Region(None, None, None, "absent", raw=raw)
    hit = spec.resolve(key)
    if hit is not None:
        return Region(hit[0], hit[1], hit[2], "resolved", raw=raw)
    where = spec.outside(key)
    if where is not None:
        return Region(None, None, None, "outside", outside=where, raw=raw)
    return Region(None, None, None, "unknown", raw=raw)


def city_strings_for(city_or_state: str, pack_id: str | None = None) -> set[str]:
    """Every recorded string that resolves to this city or state, for filtering without a stored column.

    Inverting the alias table rather than adding a `region` column keeps one source of truth. A filter asking
    for Bengaluru has to match `BLR` too, and it does so by asking the same table the resolver asks.
    """
    spec = active_pack(pack_id).region
    if spec is None:
        return set()
    target = city_or_state.strip().lower()
    out: set[str] = set()
    for key in spec.keys():
        hit = spec.resolve(key)
        if hit and (hit[0].lower() == target or hit[1].lower() == target):
            out.add(key)
    return out


def road_class_of(frame) -> str:
    """The frame's road class, or `unresolved`.

    `Frame.road_class` exists and is NULL on all 41,752 frames. It resolves to a real value the moment
    anything populates it; until then this returns the honest answer rather than a default that would make
    the corpus look stratified by road type when it is not.
    """
    return getattr(frame, "road_class", None) or "unresolved"
