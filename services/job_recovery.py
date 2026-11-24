"""Continue the imports and exports a restart interrupted.

These run as tasks inside the API process. When the process ends, so do they, and until now the only
thing that happened next was the reaper failing the row ten minutes later and a person starting the
whole job again (services/job_reaper.py). For an import that meant a second session holding the same
frames; for an export, hours of repeated work.

Startup is where this runs, and the job's lock (services/job_lock.py) is what makes it safe: a job is
picked up only if no live process holds its lock, which is true of every job the previous process was
running and false of one another API process is running right now. There is no staleness window to wait
out, so a job is continued at the first startup after it stopped rather than failed.

What is picked up, deliberately narrowly:
  * a job whose row says it may be (`resumable`). Rows written before this existed, rows a test seeded
    and rows another path created carry no such mark and are left to the reaper exactly as before;
  * at most `jobs.max_attempts` times, so a job that takes the process down every time is failed with a
    message saying so instead of taking it down again;
  * imports of annotated formats, which continue into their own session (services/imports/run.py), and
    exports, which are repeated from their stored spec - nothing is sealed until an export finishes, so
    a second run replaces the first rather than adding to it.

Raw media imports (video, MCAP, image folders) go through the ingest pipeline, which has no cursor to
continue from. They are not picked up; the reaper fails them as before.

GPU work is not started here. Auto-label and training have their own durable queue and worker
(services/gpu_worker.py), and an agent sweep is continued by a person from its run page, because starting
model inference at boot is not something a restart should decide.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.observability import spawn
from db.models import ExportJob, ImportJob
from db.session import get_sessionmaker
from services.job_lock import is_held

log = get_logger("job_recovery")

LIVE = ("pending", "running")
GAVE_UP = ("interrupted by a restart {n} times and not continued again; start it again once whatever "
           "is stopping the server is fixed")


async def _resumable_imports() -> list[ImportJob]:
    from services.imports.run import ADAPTERS

    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(ImportJob).where(ImportJob.status.in_(LIVE)))).scalars().all()
    return [j for j in rows if (j.spec or {}).get("resumable") and j.format in ADAPTERS]


async def _resumable_exports() -> list[ExportJob]:
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(ExportJob).where(ExportJob.status.in_(LIVE)))).scalars().all()
    return [j for j in rows if (j.checkpoint or {}).get("resumable") and j.spec]


async def _take_attempt(model, job_id: uuid.UUID, field: str) -> int | None:
    """Count one more pick-up on the row. None once the cap is reached, with the row failed."""
    cap = get_settings().jobs.max_attempts
    async with get_sessionmaker()() as db:
        j = await db.get(model, job_id, with_for_update=True)
        if j is None or j.status not in LIVE:
            return None
        state = dict(getattr(j, field) or {})
        n = int(state.get("attempts") or 0)
        if n >= cap:
            j.status = "error"
            j.error = GAVE_UP.format(n=n)
            await db.commit()
            log.warning("job_recovery.gave_up", job_id=str(job_id), attempts=n)
            return None
        state["attempts"] = n + 1
        setattr(j, field, state)
        await db.commit()
        return n + 1


async def recover_interrupted_jobs() -> dict[str, list[str]]:
    """Pick up every interrupted import and export nobody is running. Returns the ids continued."""
    out: dict[str, list[str]] = {"import": [], "export": []}
    if not get_settings().jobs.auto_resume:
        return out

    from services.imports.records import ImportSpec
    from services.imports.run import run_import_guarded

    for j in await _resumable_imports():
        if await is_held(j.job_id):
            continue
        if await _take_attempt(ImportJob, j.job_id, "spec") is None:
            continue
        spec = j.spec or {}
        ispec = ImportSpec(format=j.format, source_uri=j.source_uri or "", target_vehicle=j.target_vehicle,
                           city=j.city, options=spec.get("options") or {}, project_id=spec.get("project_id"))
        spawn(run_import_guarded(ispec, j.job_id), name="run_import_guarded", job_id=str(j.job_id),
              resumed=True)
        out["import"].append(str(j.job_id))

    from services.export.dataset import SliceSpec

    for e in await _resumable_exports():
        if await is_held(e.job_id):
            continue
        if await _take_attempt(ExportJob, e.job_id, "checkpoint") is None:
            continue
        from services.api.routers.datasets import _run_export

        spawn(_run_export(e.job_id, SliceSpec(**e.spec)), name="_run_export", job_id=str(e.job_id),
              resumed=True)
        out["export"].append(str(e.job_id))

    if out["import"] or out["export"]:
        log.info("job_recovery.continued", imports=len(out["import"]), exports=len(out["export"]))
    return out
