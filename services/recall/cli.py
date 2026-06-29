"""CLI: run recall recovery over a session.

    python -m services.recall.cli run-recall --session <id>
    python -m services.recall.cli run-recall --session <id> --shortlist-only

Trackgap runs over the full session (free, no model). With --shortlist-only the expensive model
channels (SAM everything + VLM) touch only the active-learning shortlist frames rather than every frame.
"""

from __future__ import annotations

import asyncio

import click

from core.config import get_settings
from core.logging import setup_logging


@click.command("run-recall")
@click.option("--session", "session_id", required=True, help="session id to recover recall over")
@click.option("--shortlist-only", is_flag=True, default=False,
              help="run model channels only on the active-learning shortlist (trackgap stays full-session)")
def main(session_id: str, shortlist_only: bool) -> None:
    setup_logging(get_settings().log_level)
    asyncio.run(_run(session_id, shortlist_only))


async def _run(session_id: str, shortlist_only: bool) -> None:
    from db.session import get_sessionmaker
    from services.recall.recover import run_recall

    frame_ids = None
    if shortlist_only:
        from services.activelearn.selector import score_candidates

        async with get_sessionmaker()() as db:
            items = await score_candidates(db, session_id=session_id)
        frame_ids = sorted({it["frame_id"] for it in items})

    async with get_sessionmaker()() as db:
        result = await run_recall(db, session_id, frame_ids=frame_ids)
    click.echo(result)


if __name__ == "__main__":
    main()
