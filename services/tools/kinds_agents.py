"""Background job kinds for the agents annotation tools. Register them with services.tools.registry.register."""

from __future__ import annotations

import uuid

from services.tools.registry import JobContext, register


@register("autopilot_night", gpu=True, autolabel=True, label="Overnight autopilot")
async def autopilot_night(ctx: JobContext) -> dict:
    """One night of the autopilot (services/agent/autopilot.py): auto-label the new sessions, check the proposed
    auto-accepts with the VLM and the error detectors, keep only what the autonomy ladder allows, report.

    GPU and auto-labelling: the GPU worker runs it under the single GPU slot, and never starts it in office hours.
    A cancel (or the page's kill switch) is noticed between steps; the night then sends whatever it had not
    checked to review rather than leaving it accepted.
    """
    from services.agent.autopilot import run_night

    return await run_night(uuid.UUID(str(ctx.params["night_id"])), ctx.progress)


@register("labelagent_session", gpu=True, autolabel=True, label="Label agent")
async def labelagent_session(ctx: JobContext) -> dict:
    """Check a session's machine labels with the label agent (services/labelagent/agent.py).

    Embeds any of the session's objects that have no vectors yet (that is the GPU part), then runs the
    specialists over them. A dry run unless `apply` is set: it reports what it would fix, flag and ask without
    touching a label. Applied, it writes one reversible AgentRun; fixed labels stay in review.
    """
    from db.session import get_sessionmaker
    from services.intelligence.embed.service import embed_objects
    from services.labelagent.agent import run_session

    if ctx.session_id is None:
        raise ValueError("labelagent_session needs a session")
    await ctx.progress(0.05, "embedding new objects")
    emb = await embed_objects(ctx.session_id, only_missing=True)
    await ctx.progress(0.5, "checking labels")
    async with get_sessionmaker()() as db:
        out = await run_session(db, ctx.session_id, apply=bool(ctx.params.get("apply")),
                                limit=ctx.params.get("limit"),
                                created_by=str(ctx.created_by) if ctx.created_by else "labelagent")
    await ctx.progress(1.0, "done")
    return {**out, "embedded_objects": emb.get("embedded_objects", 0)}


@register("labelagent_benchmark", gpu=True, autolabel=True, label="Label agent benchmark")
async def labelagent_benchmark(ctx: JobContext) -> dict:
    """Re-score the label agent and its specialists against the human rulings and save the report; the agent
    may fix classes only while this report says its wrong-fix rate is under the limit
    (services/labelagent/report.py). Runs the SigLIP 2 text encoder, which is why it is a worker job."""
    from core.config import get_settings
    from db.session import get_sessionmaker
    from services.labelagent.agent import _stamp
    from services.labelagent.report import NIGHTLY_STAGES, build_report, gate_from, save_report

    await ctx.progress(0.05, "scoring against human rulings")
    async with get_sessionmaker()() as db:
        report = await build_report(db, ctx.params.get("stages") or NIGHTLY_STAGES)
        report["rulings_at"] = (await _stamp(db))[0]
    uri = save_report(report)
    ok, why = gate_from(report, get_settings().labelagent.max_wrong_fix)
    await ctx.progress(1.0, "done")
    return {"report": uri, "fixes_allowed": ok, "gate": why, "truth_items": report["truth"]["items"]}
