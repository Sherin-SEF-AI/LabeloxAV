"""lidar_perception: the A100 burst job for native 3D detection, point cloud segmentation, and bulk 2D-to-3D
lifting over large volumes. Interactive annotation and correction stay local; this job runs the heavy model
work on the burst node and writes results back. Native detection and PTv3 segmentation need OpenPCDet and
Pointcept (CUDA and dense-LiDAR bound), which live on the burst node; locally the job parks on the documented
seam, the same contract as training and pointcloud_build, never a fake executor.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from core.logging import get_logger
from services.lidar.detect3d.native import NativeDetectionUnavailable

log = get_logger("lidar_perception")

_QUEUED_NOTE = (
    "Queued for the cloud A100. Provision the pod with `make cloud-provision`, push the clouds to /workspace "
    "on the network volume, run native detection (OpenPCDet) and PTv3 segmentation (Pointcept) on the pod, "
    "pull object_3d and point_segmentation rows back into the DB and MinIO, then stop the pod. The local "
    "worker will not run a cloud job."
)


class LidarPerceptionSpec(BaseModel):
    cloud_ids: list[uuid.UUID] = []          # clouds to run native detection / segmentation over
    frame_ids: list[uuid.UUID] = []          # frames to bulk-lift (2D-to-3D)
    task: str = "lift"                        # lift | native_detect | segment
    compute_target: str = "local"            # local (5080) | cloud (A100 seam)


async def run_lidar_perception(spec: LidarPerceptionSpec) -> dict:
    """Dispatch by compute target and task. Lifting runs locally (it reuses the 2D stack); native detection
    and segmentation run on the burst node and park on the seam locally."""
    if spec.compute_target == "cloud":
        return await _queue_for_cloud(spec)

    if spec.task == "lift":
        from services.lidar.detect3d.run import lift_frame
        built = []
        for fid in spec.frame_ids:
            built.append(await lift_frame(fid))
        return {"task": "lift", "frames": len(spec.frame_ids), "results": built}

    if spec.task == "native_detect":
        from services.lidar.detect3d.run import detect_native_cloud
        try:
            results = [await detect_native_cloud(cid) for cid in spec.cloud_ids]
            return {"task": "native_detect", "clouds": len(spec.cloud_ids), "results": results}
        except NativeDetectionUnavailable as exc:
            log.info("lidar_perception.native_unavailable", reason=str(exc))
            return {"task": "native_detect", "stage": "queued-cloud", "note": _QUEUED_NOTE, "reason": str(exc)}

    if spec.task == "segment":
        from services.lidar.segment3d.run import segment_cloud
        results = [await segment_cloud(cid) for cid in spec.cloud_ids]
        return {"task": "segment", "clouds": len(spec.cloud_ids), "results": results}

    return {"error": f"unknown task {spec.task}"}


async def _queue_for_cloud(spec: LidarPerceptionSpec) -> dict:
    log.info("lidar_perception.queued_for_cloud", task=spec.task,
             clouds=len(spec.cloud_ids), frames=len(spec.frame_ids))
    return {"task": spec.task, "stage": "queued-cloud", "note": _QUEUED_NOTE}
