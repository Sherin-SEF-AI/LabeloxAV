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
