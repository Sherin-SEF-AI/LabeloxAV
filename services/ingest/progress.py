"""Ingest progress, from the database first and the batch log only as a fallback.

Progress was read by running a regular expression over `.perception_work/ingest_batch.log`. That works
exactly as long as the API process shares a filesystem with the shell script, which stops being true the
moment the API runs in its own container, and it is silently wrong rather than obviously broken: the
endpoint returns `active: false` and looks like a quiet system.

The database is the real source. Every ingest through the API creates an `ImportJob` row carrying its own
status and counts, so a running import is visible wherever the API runs. The log fallback is kept because
`scripts/ingest_dashcam_batch.sh` genuinely writes one and is genuinely used, but it is now clearly second,
and the response says which source answered so an operator can tell a container-boundary problem from a
quiet system.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("ingest_progress")

# A job that has not been touched in this long is stale rather than running. Ten minutes is generous enough
# to cover a slow video decode and short enough that a crashed worker does not look busy forever.
STALE_AFTER_SECONDS = 600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def read_progress(db: AsyncSession) -> dict:
    """Current ingest progress. Prefers live ImportJob rows; falls back to the batch log."""
    from db.models import ImportJob

    running = (await db.execute(
        select(ImportJob).where(ImportJob.status.in_(("running", "queued")))
        .order_by(ImportJob.created_at.desc()).limit(20))).scalars().all()

    if running:
        now = time.time()
        fresh = []
        for j in running:
            updated = getattr(j, "updated_at", None) or j.created_at
            age = now - updated.timestamp() if updated else 0
            if age <= STALE_AFTER_SECONDS:
                fresh.append(j)
        if fresh:
            frames = sum(int((j.counts or {}).get("frames") or 0) for j in fresh)
            progress = sum(float(j.progress or 0.0) for j in fresh) / len(fresh)
            return {
                "source": "import_job",
                "active": True, "finished": False,
                "done": sum(1 for j in running if j.status == "done"),
                "total": len(running),
                "current": fresh[0].source_uri, "frames": frames,
                "progress": round(progress, 4),
                "jobs": [{"job_id": str(j.job_id), "status": j.status,
                          "progress": float(j.progress or 0.0),
                          "source_uri": j.source_uri} for j in fresh],
            }

    return _read_batch_log(db)


def _read_batch_log(db: AsyncSession) -> dict:
    """The shell batch script's own log. Second choice, and the response says so."""
    root = _repo_root()
    logfile = root / ".perception_work" / "ingest_batch.log"
    done_file = root / ".perception_work" / "ingested_videos.txt"
    if not logfile.exists():
        # Not an error. No API job and no log means nothing is ingesting, which is the ordinary state.
        return {"source": "none", "active": False, "finished": False, "done": 0, "total": 0,
                "current": None, "frames": 0, "progress": 0.0, "jobs": []}

    text = logfile.read_text()
    marks = re.findall(r"\[(\d+)/(\d+)\]", text)
    total = int(marks[-1][1]) if marks else 0
    done = 0
    if done_file.exists():
        done = len([ln for ln in done_file.read_text().splitlines() if ln.strip()])
    current = re.findall(r"ingesting (\S+)", text)
    finished = "BATCH DONE" in text
    active = (not finished) and (time.time() - logfile.stat().st_mtime) < 120
    return {
        "source": "batch_log",
        "active": active, "finished": finished, "done": done, "total": total,
        "current": current[-1] if current else None,
        "frames": 0,        # filled by the caller, which owns the session
        "progress": round(done / total, 4) if total else 0.0,
        "jobs": [],
        # Said out loud: this path only works when the API shares a filesystem with the script, and an
        # operator seeing a quiet system needs to be able to tell that apart from a container boundary.
        "note": "read from the batch log; only visible when the API shares a filesystem with the script",
    }


async def with_frame_count(db: AsyncSession, progress: dict, vehicle_id: str = "DASHCAM-01") -> dict:
    """Attach the real landed-frame count, which is the number an operator actually watches."""
    from db.models import Frame
    from db.models import Session as DbSession

    if progress.get("frames"):
        return progress
    try:
        frames = (await db.execute(
            select(func.count(Frame.frame_id))
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .where(DbSession.vehicle_id == vehicle_id))).scalar()
        progress["frames"] = int(frames or 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("ingest_progress.frame_count_failed", error=str(exc))
    return progress
