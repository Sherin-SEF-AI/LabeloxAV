"""Indian RTO sign taxonomy loader (M2.3): categories + types + per-type text_bearing flag + SigLIP 2
zero-shot prompt. Cached like the main ontology."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from core.config import get_settings


@functools.lru_cache(maxsize=1)
def get_sign_taxonomy() -> dict:
    data = yaml.safe_load(Path(get_settings().models.sign.taxonomy_path).read_text())
    types = data["types"]
    return {"version": data["version"], "categories": data["categories"], "types": types,
            "by_name": {t["name"]: t for t in types}}
