"""The git sha of the running tree, resolved once per process.

Stamped onto every InferenceRun so an evaluation result can be tied to the exact code that produced it: the
prediction plane is only reproducible, and the (model, gold, code, params) run key only meaningful, if the code
version is recorded. Falls back to the LBX_CODE_SHA env var (set in the Dockerfile) when the .git directory is
absent, as in a deployed image, and to None when neither is available so callers can record "unknown" rather
than crash.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def code_sha() -> str | None:
    env = os.environ.get("LBX_CODE_SHA")
    if env:
        return env.strip()[:40]
    if not (_REPO_ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:  # noqa: BLE001 - a missing/broken git must degrade to "unknown", never fail a run
        return None
    sha = out.stdout.strip()
    return sha[:40] if out.returncode == 0 and sha else None
