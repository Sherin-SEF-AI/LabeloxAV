"""Installation health for administrators: the same checks `scripts/doctor.sh` runs, as JSON.

The web app's system status page reads this, so an administrator sees "the GPU worker is not running, start
it with ..." on screen instead of learning it from a job bar that never moves. Admin only: the results name
hosts, ports and model files.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from services.api.deps import require_role

router = APIRouter()


@router.get("/system/checks", dependencies=[Depends(require_role("admin"))])
async def system_checks():
    from services.doctor import run_checks, summarize

    return summarize(await run_checks())
