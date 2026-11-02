"""Score every stored track's temporal coherence (services/temporal/tube.py), session by session.

    python -m scripts.backfill_tube_scores            # tracks with no score yet
    python -m scripts.backfill_tube_scores --all      # rescore everything

Tracking and retracking score their own tracks; this covers the ones made before track.tube_score existed.
Reads boxes and writes one float per track; nothing else changes.
"""

from __future__ import annotations

import asyncio
import json
import time

import click
from sqlalchemy import select

from core.logging import get_logger, setup_logging
from db.models import Track
from db.session import get_sessionmaker

log = get_logger("backfill_tube")


async def run(rescore_all: bool) -> dict:
    from services.temporal.tube import score_session_tracks

    async with get_sessionmaker()() as db:
        q = select(Track.session_id).distinct()
        if not rescore_all:
            q = q.where(Track.tube_score.is_(None))
        sessions = (await db.execute(q)).scalars().all()
    totals = {"sessions": len(sessions), "tracks": 0, "measured": 0}
    t0 = time.time()
    for i, sid in enumerate(sessions, 1):
        async with get_sessionmaker()() as db:
            res = await score_session_tracks(db, sid)
            await db.commit()
        totals["tracks"] += res["tracks"]
        totals["measured"] += res["measured"]
        if i % 20 == 0:
            log.info("tube.progress", sessions=i, of=len(sessions), tracks=totals["tracks"],
                     measured=totals["measured"])
    totals["seconds"] = round(time.time() - t0)
    return totals


@click.command()
@click.option("--all", "rescore_all", is_flag=True)
def main(rescore_all: bool) -> None:
    setup_logging()
    print(json.dumps(asyncio.run(run(rescore_all)), indent=1))


if __name__ == "__main__":
    main()
