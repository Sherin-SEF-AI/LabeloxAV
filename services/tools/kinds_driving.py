"""Background job kinds for the driving annotation tools (agent D). Both are CPU kinds: they run in the API process,
with the heavy NumPy work off the event loop.

- driving_calib_refine: search the camera-LiDAR extrinsic that best lines the scan up with the image, over a few
  synced frames (services/lidar/calib/align.py). Records a calibration_alignment row; does not save a calibration.
- driving_lane_propagate: carry a frame's lanes onto the following frames through the ego poses, for ranges too
  long to do inside one request (services/hdmap/lane_map.py).
"""

from __future__ import annotations

import uuid

from core.logging import get_logger
from services.tools.registry import JobContext, register

log = get_logger("kinds_driving")


@register("driving_calib_refine", label="Camera-LiDAR auto-alignment")
async def calib_refine(ctx: JobContext) -> dict:
    from services.lidar.calib.align_store import refine_job

    params = dict(ctx.params)
    if ctx.created_by and not params.get("user_id"):
        params["user_id"] = str(ctx.created_by)
    await ctx.progress(0.02, "loading scans")
    return await refine_job(params, progress=ctx.progress, job_id=ctx.job_id)


@register("driving_lane_propagate", label="Lanes along the route")
async def lane_propagate(ctx: JobContext) -> dict:
    from db.session import get_sessionmaker
    from services.hdmap.lane_map import propagate_by_pose

    p = ctx.params
    lane_ids = [uuid.UUID(x) for x in p.get("lane_ids") or []] or None

    async def prog(frac: float) -> None:
        await ctx.progress(0.02 + 0.96 * frac, "carrying lanes")

    async with get_sessionmaker()() as db:
        return await propagate_by_pose(db, uuid.UUID(str(p["frame_id"])), int(p.get("frames", 10)),
                                       lane_ids, progress=prog)
