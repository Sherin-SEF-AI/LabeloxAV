"""Background job kinds for the video annotation tools. Register them with services.tools.registry.register."""

from __future__ import annotations

from services.tools.registry import JobContext, register


@register("sam2_video", gpu=True, autolabel=False, label="Video mask tracking")
async def sam2_video(ctx: JobContext) -> dict:
    """Track one object's mask through a window of its clip with SAM 2 (services/temporal/sam2_video.py).

    Not an auto-label kind: a person clicked the object and is waiting on the result, so office hours do not
    hold it back. It is still a GPU kind, so it runs on the GPU worker, one job at a time.
    """
    from services.temporal.sam2_video import run_track

    await ctx.progress(0.02, "loading frames")
    return await run_track(ctx.params, job_id=ctx.job_id, user_id=ctx.created_by, progress=ctx.progress)


@register("describe_session", gpu=True, autolabel=True, label="Describe across a session")
async def describe_session(ctx: JobContext) -> dict:
    """Run one phrase over every frame of a session (services/autolabel/describe_session.py)."""
    from services.autolabel.describe_session import run_describe_session

    await ctx.progress(0.01, "loading models")
    return await run_describe_session(ctx.params, session_id=ctx.session_id, job_id=ctx.job_id,
                                      user_id=ctx.created_by, progress=ctx.progress)
