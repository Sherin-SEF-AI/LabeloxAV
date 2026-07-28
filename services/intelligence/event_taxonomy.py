"""The event vocabulary, loaded once and enforced everywhere.

Three consumers need to agree about what an event kind is: the derivers that propose them, the API that
accepts them, and the editor that draws them. Left to themselves each would carry its own list, and the lists
would drift, which is how a corpus ends up with `lane_change`, `laneChange` and `lane-change` all meaning the
same thing and none of them queryable together.

So the taxonomy is data, and this module is the only thing that reads it. A kind that is not in the file
cannot be written, which is a deliberate constraint rather than an inconvenience: an event vocabulary that
anyone can extend at write time is not a vocabulary, and the mining queries that filter on `severity` would
silently miss every kind nobody remembered to classify.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.logging import get_logger

log = get_logger("event_taxonomy")

_PATH = Path(__file__).resolve().parents[2] / "configs" / "event_taxonomy.yaml"

SEVERITIES = ("info", "notable", "violation")
SHAPES = ("interval", "point")
ANCHORS = ("track", "frame", "session")


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    """The parsed taxonomy. Cached: it is read once per process and does not change under a running server."""
    import yaml

    try:
        raw = yaml.safe_load(_PATH.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        # Unlike the class priors, a missing taxonomy is not something to degrade past. Every write goes
        # through validate(), so an empty taxonomy rejects everything, which is the safe direction: no events
        # is recoverable, a corpus of unclassifiable events is not.
        log.error("event_taxonomy.unavailable", path=str(_PATH), error=str(exc))
        return {"version": "unknown", "kinds": {}}
    return raw


def kinds() -> dict[str, dict]:
    return dict(load_taxonomy().get("kinds") or {})


def kind_spec(kind: str) -> dict | None:
    return kinds().get(kind)


def derived_kinds() -> list[str]:
    """The kinds a deriver may propose. Everything else is human-only by declaration."""
    return sorted(k for k, spec in kinds().items() if spec.get("derived"))


def human_kinds() -> list[str]:
    return sorted(k for k, spec in kinds().items() if not spec.get("derived"))


def severity_of(kind: str) -> str:
    return str((kind_spec(kind) or {}).get("severity") or "info")


def signal_phase_graph() -> dict[str, list[str]]:
    return {str(k): [str(v) for v in (vs or [])]
            for k, vs in (load_taxonomy().get("signal_phase_graph") or {}).items()}


def signal_min_phase_ns() -> int:
    return int(load_taxonomy().get("signal_min_phase_ns") or 0)


def lane_params() -> dict:
    return dict(load_taxonomy().get("lane") or {})


class TaxonomyError(ValueError):
    """A rejected event. Carries a message meant to be shown to whoever tried to write it."""


def validate(kind: str, *, t_start_ns: int, t_end_ns: int | None,
             track_id=None, frame_id=None, source: str = "human") -> dict:
    """Check one event against its declared shape and return its spec.

    Raises TaxonomyError rather than returning a flag, because every caller's correct response to an invalid
    event is to refuse to write it, and a returned flag is the kind of thing a caller forgets to read.
    """
    spec = kind_spec(kind)
    if spec is None:
        raise TaxonomyError(
            f"unknown event kind '{kind}'. Known kinds: {', '.join(sorted(kinds()))}")

    if source != "human" and not spec.get("derived"):
        # A deriver claiming a human-only kind means somebody taught geometry to recognise intent. It cannot,
        # and a machine-proposed near_miss would put an unfalsifiable claim into the training set.
        raise TaxonomyError(f"'{kind}' is human-only and cannot be proposed by a deriver")

    shape = str(spec.get("shape") or "interval")
    if shape == "point" and t_end_ns is not None and t_end_ns != t_start_ns:
        raise TaxonomyError(f"'{kind}' is a point event and cannot have a distinct t_end_ns")
    if shape == "interval" and t_end_ns is None:
        raise TaxonomyError(f"'{kind}' is an interval event and needs a t_end_ns")
    if t_end_ns is not None and t_end_ns < t_start_ns:
        raise TaxonomyError("t_end_ns is before t_start_ns")

    anchor = str(spec.get("anchor") or "session")
    if anchor == "track" and track_id is None:
        raise TaxonomyError(f"'{kind}' is about an actor and needs a track_id")
    if anchor == "frame" and frame_id is None:
        raise TaxonomyError(f"'{kind}' is about one frame and needs a frame_id")

    return spec


def describe() -> dict:
    """The taxonomy as the editor and the API docs want it: flat, with the shape fields promoted."""
    out = []
    for kind, spec in sorted(kinds().items()):
        out.append({
            "kind": kind,
            "modality": spec.get("modality", "driving"),
            "shape": spec.get("shape", "interval"),
            "anchor": spec.get("anchor", "session"),
            "severity": spec.get("severity", "info"),
            "derived": bool(spec.get("derived")),
            "description": spec.get("description", ""),
            "payload": list(spec.get("payload") or []),
        })
    return {"version": load_taxonomy().get("version", "unknown"), "kinds": out,
            "severities": list(SEVERITIES),
            "signal_phase_graph": signal_phase_graph()}
