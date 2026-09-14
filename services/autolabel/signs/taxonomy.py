"""Indian RTO sign taxonomy loader (M2.3): categories + types + per-type text_bearing flag + SigLIP 2
zero-shot prompt. Cached like the main ontology."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from core.config import get_settings


@functools.lru_cache(maxsize=1)
def get_sign_taxonomy() -> dict:
    """The sign types, and the negatives that let a classifier decline to type something.

    `negatives` carries prompts for things that are not signs. Without them a zero-shot classifier over the
    types alone has no way to answer "none of these", because a softmax over mutually exclusive prompts
    always elects a winner however unlike a sign the crop is.
    """
    data = yaml.safe_load(Path(get_settings().models.sign.taxonomy_path).read_text())
    types = data["types"]
    # IRC:67 is the standard these signs are actually erected under, and its code is what a road authority
    # calls a sign. Carrying it means a class here can be talked about with somebody outside this system,
    # and it gives the hierarchical evaluation a real middle level: a stop sign read as a give way is a
    # mandatory sign read as a mandatory sign, which is a smaller error than reading it as a hospital.
    return {"version": data["version"], "categories": data["categories"], "types": types,
            "negatives": data.get("negatives") or [],
            "by_name": {t["name"]: t for t in types},
            "by_irc_code": {t["irc_code"]: t for t in types if t.get("irc_code")},
            "groups": sorted({t["irc_group"] for t in types if t.get("irc_group")})}


def irc_group_of(sign_type: str | None) -> str | None:
    """The IRC:67 group a sign type belongs to, or None when the type is unknown or ungrouped.

    None rather than a default group: a sign whose type nobody recognised is not an informatory sign, and
    folding it into one would make the group-level metric look better than it is.
    """
    if not sign_type:
        return None
    t = get_sign_taxonomy()["by_name"].get(sign_type)
    return (t or {}).get("irc_group")


@functools.lru_cache(maxsize=1)
def text_bearing_types() -> frozenset[str]:
    """Sign types that carry readable text, so OCR can be aimed rather than run over everything.

    Four of the twenty-one. The flag has been in the taxonomy from the start and nothing read it, so every
    no-horn roundel in a session was sent to a VLM to be told it has no text.
    """
    return frozenset(t["name"] for t in get_sign_taxonomy()["types"] if t.get("text_bearing"))
