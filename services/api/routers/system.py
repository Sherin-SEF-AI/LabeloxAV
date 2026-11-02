"""Installation health for administrators: the same checks `scripts/doctor.sh` runs, as JSON.

The web app's system status page reads this, so an administrator sees "the GPU worker is not running, start
it with ..." on screen instead of learning it from a job bar that never moves. Admin only: the results name
hosts, ports and model files.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from services.api.deps import require_role

router = APIRouter()


@router.get("/system/features", dependencies=[Depends(require_role("annotator"))])
async def system_features():
    """Which optional parts this installation has, so the web app shows only what exists.

    Any signed-in person may read it: it says which pages to offer, nothing about hosts or files.
    """
    from core.config import get_settings
    from packs.registry import enabled_optional, optional_pack_ids
    from services.doctor import gpu_worker_running

    enabled = enabled_optional()
    return {
        "optional_packs": {pid: pid in enabled for pid in optional_pack_ids()},
        # Whether queued GPU jobs (training, and auto-label in worker mode) will actually run.
        "gpu_worker": await gpu_worker_running(),
        "autolabel_executor": get_settings().jobs.autolabel_executor,
        # Switching identity in the account menu exists only for a dev server with auth off; with auth on,
        # who you are is your sign-in.
        "auth_enabled": get_settings().auth.enabled,
    }


@router.get("/system/checks", dependencies=[Depends(require_role("admin"))])
async def system_checks():
    from services.doctor import run_checks, summarize

    return summarize(await run_checks())
