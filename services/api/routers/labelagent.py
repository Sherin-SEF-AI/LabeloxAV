"""API for the label agent (services/labelagent): its benchmark, checking a session, and undoing a fix.

`GET  /labelagent/benchmark`                  the latest benchmark report (scripts/label_benchmark.py --save)
`POST /labelagent/sessions/{session_id}/run`  queue a check of a session; a dry run unless `apply`
`POST /labelagent/objects/{object_id}/undo`   put one object back the way the agent found it

A whole run is reverted through the agents' shared route, `POST /api/agent/runs/{run_id}/revert`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role

router = APIRouter()


def _uuid(v: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(v)
    except ValueError:
        raise HTTPException(400, f"invalid {what} id") from None


@router.get("/labelagent/benchmark", dependencies=[Depends(require_role("annotator"))])
async def benchmark() -> dict:
    """The stored report, or an empty answer that says how to make one. Not computed here: a full run reads
    every ruled object's vectors and runs the text encoder, which is a job, not a request."""
    from services.api.offload import run_blocking
    from services.labelagent.report import load_latest

    report = await run_blocking(load_latest)
    if report is None:
        return {"report": None, "detail": "no benchmark has been saved yet (scripts/label_benchmark.py --save)"}
    return {"report": report}


class RunIn(BaseModel):
    apply: bool = False
    limit: int | None = Field(default=None, ge=1, le=200_000)


@router.post("/labelagent/sessions/{session_id}/run")
async def run(session_id: str, body: RunIn, db: AsyncSession = Depends(db_session),
              user=Depends(require_role("annotator"))) -> dict:
    """Queue the check as a tool job: it embeds new objects on the GPU, so it waits for the GPU worker and
    never starts in office hours. Applying changes labels and needs a reviewer; a dry run does not."""
    from db.models import Session as DbSession
    from services.api.deps import role_rank
    from services.tools.registry import enqueue

    sid = _uuid(session_id, "session")
    if await db.get(DbSession, sid) is None:
        raise HTTPException(404, "session not found")
    if body.apply and role_rank(user.role) < role_rank("reviewer"):
        raise HTTPException(403, "applying the label agent's fixes needs a reviewer")
    job = await enqueue(db, "labelagent_session", {"apply": body.apply, "limit": body.limit}, session_id=sid,
                        user_id=getattr(user, "user_id", None))
    return {"job_id": str(job.job_id), "status": job.status, "apply": body.apply}


@router.post("/labelagent/objects/{object_id}/undo")
async def undo(object_id: str, db: AsyncSession = Depends(db_session),
               user=Depends(require_role("annotator"))) -> dict:
    from services.labelagent.agent import undo_object

    oid = _uuid(object_id, "object")
    try:
        return await undo_object(db, oid)
    except LookupError:
        raise HTTPException(404, "object not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
