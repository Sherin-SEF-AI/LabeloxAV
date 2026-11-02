"""The home page in one call: how big the corpus is, what is waiting for people, what is running, where work
was last happening, and which first-run steps a new team has done.

The home page used to be the triage queue alone. That serves a reviewer coming back to work, and tells a
person on their first day nothing: not whether anything has been imported, not whether a job failed
overnight, not what to do next. Everything here is counted from the database at request time; nothing is
cached or estimated, so the numbers on screen are the numbers in the tables.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    AutolabelJob,
    DatasetCommit,
    Frame,
    ImportJob,
    ModelRegistry,
    Object,
    TrainingJob,
)
from db.models import Session as DbSession
from services.api.deps import db_session, require_role
from services.api.project_guard import caller_hidden_projects
from services.project_access import as_hidden, scope_visible_sessions, visible_session_ids

router = APIRouter()

# Object states that mean a person has settled the label, and the one that means it waits for a person.
SETTLED = ("accepted", "auto_accept", "rejected")
WAITING = "review"
RECENT_SESSIONS = 6
ACTIVE = ("pending", "running")
# A failure older than this is history, not something needing attention: without a window, one bad night stays
# at the top of every person's home page indefinitely.
ATTENTION_DAYS = 14


async def _count(db: AsyncSession, stmt) -> int:
    return int((await db.execute(stmt)).scalar_one() or 0)


@router.get("/home", dependencies=[Depends(require_role("annotator"))])
async def home(db: AsyncSession = Depends(db_session), hidden=Depends(caller_hidden_projects)):
    # Counted over what the caller may see (services/project_access.py). With nothing hidden, the plain counts.
    hidden = as_hidden(hidden)
    sessions = await _count(db, scope_visible_sessions(select(func.count()).select_from(DbSession), hidden))
    frame_q = select(func.count()).select_from(Frame)
    state_q = select(Object.state, func.count()).group_by(Object.state)
    if hidden:
        frame_q = frame_q.where(Frame.session_id.in_(visible_session_ids(hidden)))
        state_q = state_q.join(Frame, Frame.frame_id == Object.frame_id) \
            .where(Frame.session_id.in_(visible_session_ids(hidden)))
    frames = await _count(db, frame_q)
    by_state = {k: int(v) for k, v in (await db.execute(state_q)).all()}
    objects = sum(by_state.values())

    # Jobs: what is running or queued now, and what failed recently, across the three job kinds a team starts
    # from the interface.
    since = datetime.now(UTC) - timedelta(days=ATTENTION_DAYS)
    active: dict[str, int] = {}
    running = 0
    failed: list[dict] = []
    job_models: tuple[tuple[str, Any], ...] = (("autolabel", AutolabelJob), ("import", ImportJob),
                                                ("training", TrainingJob))
    for kind, model in job_models:
        active[kind] = await _count(db, select(func.count()).select_from(model).where(model.status.in_(ACTIVE)))
        running += await _count(db, select(func.count()).select_from(model).where(model.status == "running"))
        for j in (await db.execute(select(model).where(model.status == "error", model.updated_at >= since)
                                   .order_by(model.updated_at.desc()).limit(3))).scalars():
            failed.append({"kind": kind, "job_id": str(j.job_id), "error": (j.error or "")[:240],
                           "at": j.updated_at.isoformat() if j.updated_at else None})
    failed.sort(key=lambda f: f["at"] or "", reverse=True)

    # Recent sessions with how far labelling has got: frames, objects, and how many still wait for a person.
    recent_rows = (await db.execute(scope_visible_sessions(select(DbSession), hidden)
                                    .order_by(DbSession.start_ts_ns.desc()).limit(RECENT_SESSIONS))).scalars().all()
    recent = []
    for s in recent_rows:
        n_frames = await _count(db, select(func.count()).select_from(Frame).where(Frame.session_id == s.session_id))
        obj_q = select(Object.state, func.count()).join(Frame, Frame.frame_id == Object.frame_id) \
            .where(Frame.session_id == s.session_id).group_by(Object.state)
        states = {k: int(v) for k, v in (await db.execute(obj_q)).all()}
        recent.append({"session_id": str(s.session_id), "vehicle_id": s.vehicle_id, "city": s.city,
                       "start_ts_ns": s.start_ts_ns, "frames": n_frames, "objects": sum(states.values()),
                       "waiting": states.get(WAITING, 0)})

    # Every export, from the Datasets page or the API and SDK, leaves a dataset commit; only the page's path
    # records an ExportJob, so counting jobs alone left the step unticked for a team that exports by script.
    exports = await _count(db, select(func.count()).select_from(DatasetCommit))
    models = await _count(db, select(func.count()).select_from(ModelRegistry))
    autolabelled = await _count(db, select(func.count()).select_from(AutolabelJob)
                                .where(AutolabelJob.status == "done"))
    settled = sum(by_state.get(s, 0) for s in SETTLED)

    return {
        "corpus": {"sessions": sessions, "frames": frames, "objects": objects, "by_state": by_state,
                   "waiting": by_state.get(WAITING, 0), "settled": settled},
        "jobs": {"active": active, "active_total": sum(active.values()), "running": running,
                 "failed": failed[:5], "failed_window_days": ATTENTION_DAYS},
        "recent_sessions": recent,
        "waiting_for_people": await _waiting_for_people(db),
        # The first-run path, in order. Each is a fact about the database, not a box somebody ticked.
        "steps": {
            "data": sessions > 0,
            "autolabel": autolabelled > 0,
            "review": settled > 0,
            "export": exports > 0,
            "model": models > 0,
        },
    }


async def _waiting_for_people(db: AsyncSession) -> list[dict]:
    """Measurements that cannot finish without a person: blind audits not yet labelled.

    A blind audit is the only instrument that says whether gold recall is real or a small gold set flattering
    the model, and it waits on someone labelling its frames from scratch. Nothing in the app said so, so the
    first one sat seeded and untouched for months. Each row names the first frame still to label, which the
    project filter also uses to keep a hidden project's audit off the page.
    """
    from db.models import BlindAuditFrame
    from services.verdyx.blind_audit import list_audits

    out = []
    for a in await list_audits(db, limit=20):
        if not a["job_id"] or a["status"] == "scored" or a["n_labeled"] >= a["n_frames"]:
            continue
        first = (await db.execute(
            select(BlindAuditFrame.frame_id).where(BlindAuditFrame.audit_id == uuid.UUID(a["audit_id"]),
                                                   BlindAuditFrame.labeled_at.is_(None))
            .order_by(BlindAuditFrame.audit_frame_id).limit(1))).scalar()
        if first is None:
            continue
        out.append({"kind": "blind_audit", "audit_id": a["audit_id"], "job_id": a["job_id"],
                    "frame_id": str(first), "n_frames": a["n_frames"], "n_labeled": a["n_labeled"],
                    "created_at": a["created_at"]})
    return out
