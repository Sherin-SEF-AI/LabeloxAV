"""Pack discovery and access.

This is the one sanctioned bridge between the engine and concrete packs. It discovers packs by their
`pack.yaml`, dynamically imports each pack module (importlib, so no static engine->concrete-pack edge exists
for import-linter to flag), reads its `PACK`, and validates it structurally against the DomainPack contract.

The engine calls `get_pack(pack_id)` / `get_ontology(pack_id)`; it never imports `packs.av` directly.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

import yaml

from core.logging import get_logger
from packs.base import DomainPack

log = get_logger("packs.registry")

PACKS_DIR = Path(__file__).resolve().parent


def _pack_dirs() -> list[Path]:
    """Every immediate subdirectory of packs/ that carries a pack.yaml manifest."""
    return sorted(p.parent for p in PACKS_DIR.glob("*/pack.yaml"))


def _load_one(pack_dir: Path) -> DomainPack:
    pack_id = pack_dir.name
    module_name = f"packs.{pack_id}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - surface a broken pack with its id, do not mask it
        raise RuntimeError(f"pack '{pack_id}' failed to import ({module_name}): {exc}") from exc

    pack = getattr(module, "PACK", None)
    if pack is None:
        raise RuntimeError(f"pack '{pack_id}' exposes no PACK object in {module_name}")
    if not isinstance(pack, DomainPack):
        missing = [f for f in DomainPack.__annotations__ if not hasattr(pack, f)]
        raise RuntimeError(f"pack '{pack_id}' is not a DomainPack; missing surfaces: {missing}")
    if pack.manifest.id != pack_id:
        raise RuntimeError(
            f"pack directory '{pack_id}' declares manifest id '{pack.manifest.id}'; they must match")

    # The pack.yaml is the human-facing manifest; keep it and the code manifest honest about each other.
    meta = yaml.safe_load((pack_dir / "pack.yaml").read_text()) or {}
    if meta.get("id") != pack.manifest.id:
        raise RuntimeError(
            f"pack '{pack_id}' pack.yaml id '{meta.get('id')}' != code manifest id '{pack.manifest.id}'")
    return pack


@lru_cache(maxsize=1)
def load_packs() -> dict[str, DomainPack]:
    """Discover, import, and validate every pack under packs/. Cached for the process."""
    packs: dict[str, DomainPack] = {}
    for d in _pack_dirs():
        pack = _load_one(d)
        packs[pack.manifest.id] = pack
    log.info("packs.loaded", packs=sorted(packs))
    return packs


def get_pack(pack_id: str) -> DomainPack:
    """The pack with this id, or KeyError naming the ids that do exist."""
    packs = load_packs()
    if pack_id not in packs:
        raise KeyError(f"no domain pack '{pack_id}'; known packs: {sorted(packs)}")
    return packs[pack_id]


def pack_ids() -> list[str]:
    return sorted(load_packs())


def default_pack_id() -> str:
    """The active pack for engine paths not yet routed per-session (governance, curation). Config-driven,
    defaults to 'av'."""
    from core.config import get_settings

    return get_settings().packs.default_pack
