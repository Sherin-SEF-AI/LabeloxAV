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
    return {"version": data["version"], "categories": data["categories"], "types": types,
            "negatives": data.get("negatives") or [],
            "by_name": {t["name"]: t for t in types}}


@functools.lru_cache(maxsize=1)
def text_bearing_types() -> frozenset[str]:
    """Sign types that carry readable text, so OCR can be aimed rather than run over everything.

    Four of the twenty-one. The flag has been in the taxonomy from the start and nothing read it, so every
    no-horn roundel in a session was sent to a VLM to be told it has no text.
    """
    return frozenset(t["name"] for t in get_sign_taxonomy()["types"] if t.get("text_bearing"))
