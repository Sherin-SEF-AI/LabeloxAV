"""The permanent per-pack parity gate.

A frozen JSON snapshot of each pack's every surface. Any change to a pack's behaviour (ontology, safety
definition, auto-label profile, eval strata, quality profile, forge targets, privacy plane, scene model)
changes its digest and fails this test. AV metric parity is a hard gate for the multi-domain refactor, so AV
behaviour may only move by a deliberate, reviewed golden update; the Sec golden guards Sec the same way.

To update after an intentional change:  LBX_REGEN_GOLDEN=1 pytest tests/test_golden_av_pack.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from packs.digest import pack_digest
from packs.registry import pack_ids

GOLDEN_DIR = Path(__file__).parent / "golden"


def _first_difference(golden: dict, current: dict) -> str:
    keys = sorted(set(golden) | set(current))
    for k in keys:
        g, c = golden.get(k, "<missing>"), current.get(k, "<missing>")
        if g != c:
            return (f"section '{k}' drifted:\n  golden : {json.dumps(g, sort_keys=True)}\n"
                    f"  current: {json.dumps(c, sort_keys=True)}")
    return "digests differ but no top-level section does (nested ordering?)"


@pytest.mark.parametrize("pack_id", pack_ids())
def test_pack_golden(pack_id):
    current = pack_digest(pack_id)
    golden_path = GOLDEN_DIR / f"{pack_id}_pack.json"

    if os.getenv("LBX_REGEN_GOLDEN"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        return

    assert golden_path.exists(), f"missing golden {golden_path}; regenerate with LBX_REGEN_GOLDEN=1"
    golden = json.loads(golden_path.read_text())
    assert current == golden, (
        f"{pack_id} pack digest drifted from the frozen golden. If this change is intentional, review it and "
        "regenerate with LBX_REGEN_GOLDEN=1.\n" + _first_difference(golden, current)
    )
