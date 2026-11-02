"""Read the dashcam telemetry stamp into Frame.gnss and Frame.ego_speed, then build each session's ego poses.

    python -m scripts.backfill_overlay_telemetry                 # every session without GNSS
    python -m scripts.backfill_overlay_telemetry --session <id>  # one session

Sessions with no stamp are probed on a few frames and skipped. Where a stamp is read, the session's ego
trajectory is rebuilt from it (services/intelligence/ego_pose.py), which is what ego-compensated propagation
and the map need. CPU only; nothing else changes.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import click
from sqlalchemy import func, select

from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import Frame
from db.session import get_sessionmaker

log = get_logger("backfill_overlay")


async def run(session: str | None) -> dict:
    from services.ingest.overlay_telemetry import backfill_session
    from services.intelligence.ego_pose import build_ego_pose

    store = get_object_store()
    async with get_sessionmaker()() as db:
        if session:
            sids = [uuid.UUID(session)]
        else:
            q = (select(Frame.session_id).group_by(Frame.session_id)
                 .having(func.count(Frame.gnss) == 0))
            sids = list((await db.execute(q)).scalars().all())
    totals = {"sessions": len(sids), "with_stamp": 0, "frames_written": 0, "poses": 0, "rejected_jumps": 0}
    t0 = time.time()
    for i, sid in enumerate(sids, 1):
        async with get_sessionmaker()() as db:
            res = await backfill_session(db, store, sid)
        if res.get("written"):
            totals["with_stamp"] += 1
            totals["frames_written"] += res["written"]
            totals["rejected_jumps"] += res.get("rejected_jumps", 0)
            pose = await build_ego_pose(sid)
            totals["poses"] += int(pose.get("poses") or 0)
        if i % 10 == 0:
            log.info("overlay.progress", sessions=i, of=len(sids), with_stamp=totals["with_stamp"],
                     frames=totals["frames_written"])
    totals["seconds"] = round(time.time() - t0)
    return totals


@click.command()
@click.option("--session", default=None)
def main(session: str | None) -> None:
    setup_logging()
    print(json.dumps(asyncio.run(run(session)), indent=1))


if __name__ == "__main__":
    main()
