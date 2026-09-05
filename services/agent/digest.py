"""The morning digest: one notification that says what the fleet did overnight, and the weekly report.

The nightly agents each record an AgentRun and an audit row, and until now that was the whole story:
the overnight auditor demoted auto-accepts and told nobody, the gold-drift check could roll a champion
back in silence, and finding out what happened required reading the runs table. An autonomous system
that acts at night and says nothing in the morning trains its operators to stop trusting it - or worse,
to stop checking.

One superseding notification per calendar day, sent off-hours once the night's own agents have had
their turn: per agent kind, ran / status / the one number that matters, from the AgentRun rows
themselves rather than a parallel summary that could drift. Links the audit page; the run ids in the
meta are the drill-down.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun

log = get_logger("agent.digest")

KIND = "nightly_digest"
WEEKLY_KIND = "weekly_report"

# The single most informative count each agent's report carries, by kind. A kind not listed still
# appears in the digest with its status; this only picks which number rides in the summary line.
_HEADLINE = {
    "overnight_auditor": "demoted",
    "gold_drift_check": "status",
    "gate_unblock": "target_run",
    "embed_pending": "embedded_objects",
}


def _one_line(run: AgentRun) -> dict:
    report = run.counts or {}
    line = {"kind": run.kind, "status": run.status, "run_id": str(run.run_id)}
    key = _HEADLINE.get(run.kind)
    if key and key in report:
        line[key] = report[key]
    if run.status == "error" and report.get("error"):
        line["error"] = str(report["error"])[:160]
    return line


async def maybe_send_digest(db: AsyncSession) -> dict:
    """Off-hours hook: once per day, after the nightly agents have finished (not merely launched)."""
    from services.agent.runtime.report import ran_since

    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, KIND, day_start):
        return {"ran": False, "reason": "already sent today"}

    runs = (await db.execute(
        select(AgentRun).where(AgentRun.created_at >= day_start, AgentRun.kind != KIND)
        .order_by(AgentRun.created_at))).scalars().all()
    if not runs:
        return {"ran": False, "reason": "nothing ran today yet; the digest waits for the night"}
    if any(r.status == "running" for r in runs):
        # A digest sent while the auditor is mid-flight would report a night that has not happened yet.
        return {"ran": False, "reason": "an agent is still running; the digest goes out when the "
                                        "night is finished"}

    lines = [_one_line(r) for r in runs]
    errors = [ln for ln in lines if ln["status"] == "error"]
    demoted = sum(int(ln.get("demoted") or 0) for ln in lines if ln["kind"] == "overnight_auditor")
    title = f"overnight: {len(runs)} agent runs, {len(errors)} failed"
    if demoted:
        title += f", {demoted} auto-accepts demoted"

    from services.agent.runtime.report import finish_run, launch
    from services.notify import notify

    async def worker(run_id):
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as wdb:
            await notify(wdb, kind=KIND, severity="warn" if errors else "info", title=title,
                         body="; ".join(f"{ln['kind']}: {ln['status']}" for ln in lines),
                         href="/agents", subject_type="digest",
                         subject_id=day_start.date().isoformat(),
                         meta={"runs": lines})
        await finish_run(run_id, status="committed",
                         report={"date": day_start.date().isoformat(), "runs": len(runs),
                                 "errors": len(errors), "title": title})

    return {"ran": True, **(await launch(db, KIND, worker, created_by="scheduler"))}


async def maybe_weekly_report(db: AsyncSession) -> dict:
    """Off-hours hook: the Documentation Agent's weekly report, once every 7 days.

    generate_weekly_report existed behind POST /api/agent/docs/weekly-report; a weekly artifact that
    only exists when somebody asks weekly is a request handler, not a report.
    """
    from services.agent.runtime.report import finish_run, launch, ran_since

    if await ran_since(db, WEEKLY_KIND, datetime.now(UTC) - timedelta(days=7)):
        return {"ran": False, "reason": "a weekly report exists from the last 7 days"}

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.agent.doc_agent import generate_weekly_report

        async with get_sessionmaker()() as wdb:
            res = await generate_weekly_report(wdb)
        await finish_run(run_id, status="committed", report={k: v for k, v in res.items()
                                                            if k != "markdown"})

    return {"ran": True, **(await launch(db, WEEKLY_KIND, worker, created_by="scheduler"))}
