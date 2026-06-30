"""lidar_aggregate: the A100 burst job for multi-scan registration, loop closure, and accumulation over many
sessions, plus bulk static-element extraction and traversability. Interactive review stays local; this runs
the heavy alignment and aggregation on the burst node, stages through the network volume, writes results
back, and tears the pod down. Locally a cloud-target job parks on the documented seam, never a fake executor,
the same contract as training and pointcloud_build.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from core.logging import get_logger

log = get_logger("lidar_aggregate_job")

_QUEUED_NOTE = (
    "Queued for the cloud A100. Provision the pod with `make cloud-provision`, push the session clouds to "
    "/workspace, run registration + loop closure (GTSAM) + accumulation on the pod, pull the aggregated map "
    "and aggregated_map row back, then stop the pod. The local worker will not run a cloud job."
)


class LidarAggregateSpec(BaseModel):
    session_ids: list[uuid.UUID] = []
    region: str | None = None
    voxel: float = 0.2
    compute_target: str = "local"            # local (5080) | cloud (A100 seam)


async def run_lidar_aggregate(spec: LidarAggregateSpec) -> dict:
    """Dispatch by compute target: local aggregates on the box, cloud parks on the A100 seam."""
    if spec.compute_target == "cloud":
        log.info("lidar_aggregate.queued_for_cloud", region=spec.region, sessions=len(spec.session_ids))
        return {"region": spec.region, "stage": "queued-cloud", "note": _QUEUED_NOTE}
    from services.lidar.aggregate import aggregate_sessions
    return await aggregate_sessions(spec.session_ids, region=spec.region, voxel=spec.voxel)
