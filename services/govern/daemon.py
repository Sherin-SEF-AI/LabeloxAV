"""The autonomy controller daemon (M4.4): the driver that makes the loop run on its own. It holds a
Postgres advisory lock so exactly one daemon runs (and ticks are serialized against each other), then
calls controller.tick on a fixed cadence. Each tick reads governance state, scans drift, gates any
registered challenger, and schedules a local retrain in off-hours. The manual POST /govern/controller/tick
endpoint stays for tests and on-demand runs.

    uv run python -m services.govern.daemon
"""

from __future__ import annotations

import asyncio

import click
from sqlalchemy import text

from core.config import get_settings
from core.logging import get_logger, setup_logging
from db.session import get_engine, get_sessionmaker
from services.govern.controller import tick
from services.job_reaper import reap_stale_jobs

log = get_logger("govern_daemon")


async def controller_daemon() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    key = settings.phase4.govern.controller_lock_key
    poll_s = settings.phase4.govern.controller_poll_s

    # One daemon at a time: a second instance refuses to start (mirrors the training worker's GPU mutex).
    async with get_engine().connect() as lock_conn:
        got = (await lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
        if not got:
            log.error("daemon.lock_held", note="another controller daemon is running; exiting")
            return
        log.info("daemon.start", poll_s=poll_s, advisory_lock=key)
        try:
            while True:
                try:
                    async with get_sessionmaker()() as db:
                        res = await tick(db, schedule_bursts=True)
                    log.info("daemon.tick", status=res.get("status"),
                             actions=[a["action"] for a in res.get("actions", [])])
                except Exception as exc:  # noqa: BLE001
                    log.error("daemon.tick_failed", error=str(exc))

                # Sweep dead job rows on the same cadence. This ran only in the API's startup hook, which
                # is the one moment the *previous* process's tasks are known to be dead - but it meant a
                # long-lived API whose in-process autolabel task died without writing a terminal status
                # left a row reading `running` forever, and routers/autolabel.py refuses to start while any
                # row does. Auto-labeling was then disabled for everyone until somebody restarted the API,
                # which is exactly the deadlock job_reaper's own docstring describes. The staleness window
                # still applies, so this cannot kill a job another replica is actively running.
                try:
                    async with get_sessionmaker()() as db:
                        reaped = await reap_stale_jobs(db)
                    if any(reaped.values()):
                        log.info("daemon.reaped", **{k: len(v) for k, v in reaped.items() if v})
                except Exception as exc:  # noqa: BLE001
                    log.error("daemon.reap_failed", error=str(exc))

                await asyncio.sleep(poll_s)
        finally:
            await lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            await lock_conn.commit()


@click.command()
def main() -> None:
    asyncio.run(controller_daemon())


if __name__ == "__main__":
    main()
