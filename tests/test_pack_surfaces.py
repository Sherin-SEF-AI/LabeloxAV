"""Zone rules and live-stream sampling are reached through the pack contract, not imported from the pack.

Three `from packs.sec import ...` statements sat in engine code and broke the import-linter contract, which
means CI could never go green. The contract is not bureaucracy: those imports made the security pack a hard
dependency of the engine, so an AV-only deployment still pulled RTSP handling and OpenCV, and a third pack
could not supply its own spatial rules without editing engine code.

The linter catches the direct import. It cannot catch the thing that makes the seam real, which is that both
surfaces are genuinely optional: a pack without a fixed camera has no zones to police, and the engine has to
say so rather than assume every pack is the security one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.config import REPO_ROOT
from packs.base import DomainPack, StreamSource, ZonePolicy
from packs.registry import get_pack, pack_ids


def test_the_contract_declares_both_surfaces():
    for surface in ("zone_policy", "stream_source"):
        assert surface in DomainPack.__annotations__, f"{surface} must be part of the contract"


def test_the_security_pack_fills_them():
    pack = get_pack("sec")
    assert isinstance(pack.zone_policy, ZonePolicy)
    assert isinstance(pack.stream_source, StreamSource)


def test_the_av_pack_leaves_them_empty_and_is_still_a_valid_pack():
    """Absent is a real answer, not an unfinished one. A dashcam has no permanent polygon to police, so the
    AV pack declining both surfaces must not make it fail the registry's structural check."""
    pack = get_pack("av")
    assert pack.zone_policy is None
    assert pack.stream_source is None
    assert isinstance(pack, DomainPack)


@pytest.mark.parametrize("pack_id", pack_ids())
def test_every_pack_satisfies_the_extended_contract(pack_id):
    assert isinstance(get_pack(pack_id), DomainPack)


def test_the_unavailable_error_is_part_of_the_surface():
    """The engine answers 502 for an unreachable camera and 500 for its own failure. It can only tell them
    apart if the pack publishes the exception type, because catching the pack's own class is the import this
    change removed."""
    src = get_pack("sec").stream_source
    assert isinstance(src.unavailable_error, type)
    assert issubclass(src.unavailable_error, Exception)


def test_a_policy_override_that_was_never_set_does_not_erase_a_pack_default():
    """The engine forwards optional request fields; a None must not overwrite the pack's own default and
    silently reconfigure sampling."""
    src = get_pack("sec").stream_source
    default = src.sampling_policy()
    assert src.sampling_policy(motion_threshold=None).motion_threshold == default.motion_threshold
    assert src.sampling_policy(motion_threshold=99.0).motion_threshold == 99.0


def test_the_crossing_shape_is_the_contracts_not_the_packs():
    """The engine reads every field of a Crossing to build an incident. Two definitions of it would let a
    pack add a field the engine silently drops."""
    from packs.base import Crossing as ContractCrossing
    from packs.sec.zones import Crossing as PackCrossing

    assert PackCrossing is ContractCrossing


# Engine packages that must never name a concrete pack. Mirrors the import-linter contract so the failure
# also surfaces in pytest, where it is seen long before CI.
_ENGINE_DIRS = ("services", "core", "db")
_CONCRETE_PACK_IMPORT = re.compile(r"^\s*(?:from|import)\s+packs\.(?!base|registry)(\w+)", re.MULTILINE)


def test_no_engine_module_imports_a_concrete_pack():
    offenders: list[str] = []
    for d in _ENGINE_DIRS:
        for path in (Path(REPO_ROOT) / d).rglob("*.py"):
            for m in _CONCRETE_PACK_IMPORT.finditer(path.read_text(encoding="utf-8", errors="replace")):
                line = path.read_text(encoding="utf-8", errors="replace")[:m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line} -> packs.{m.group(1)}")
    assert not offenders, "engine code must reach packs through packs.registry:\n  " + "\n  ".join(offenders)
