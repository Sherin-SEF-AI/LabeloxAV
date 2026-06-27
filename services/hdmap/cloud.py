"""Cloud map-fusion seam (M3.3): route multi-drive HD-map fusion to the RunPod A100 for the heavy
GTSAM trajectory pose-graph alignment. compute_target='cloud' parks the MapFusionJob in a visible
"queued for cloud" state; the local API never runs it. Executing it runs cloud/mapfusion_pod.py over the
georef'd map_elements, mirroring services/autolabel/cloud.py.

Data contract (MinIO <-> pod /workspace):
  1. ensure the pod is up (RunPod API; volume + venv + checkpoints persist across stops).
  2. push the per-drive map_elements (GeoJSON) + trajectories + a manifest to /workspace/in.
  3. ssh: `python cloud/mapfusion_pod.py --manifest in/manifest.json --out out/fused.json`
     (GTSAM pose-graph aligns the drives; lanelet2 refines lane topology).
  4. pull out/fused.json; write fused map_elements + seal the map_commit + export lanelet2/opendrive.
  5. stop the pod to cap billing.
The local averaging-fusion fallback (services/hdmap/run.py::run_map_fusion) keeps M3.3 fully testable.
"""

from __future__ import annotations

import uuid

from core.logging import get_logger
from db.models import MapFusionJob
from db.session import get_sessionmaker

log = get_logger("hdmap_cloud")

_NOTE = (
    "Queued for the cloud A100. Multi-drive fusion runs GTSAM there (cloud/mapfusion_pod.py). Start the "
    "pod (make cloud-provision) and run `make cloud-mapfusion JOB=<id>` to execute it; the local API will "
    "not run a cloud job (single-box GPU discipline). The local averaging-fusion fallback handles the "
    "single-box path."
)


async def mark_queued_for_cloud_fusion(job_id, session_ids, region) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(MapFusionJob, uuid.UUID(str(job_id)))
        if j is None:
            return
        j.status = "pending"
        j.stage = "queued_cloud"
        j.counts = {"compute_target": "cloud", "region": region, "sessions": len(session_ids), "note": _NOTE}
        await db.commit()
    log.info("hdmap.queued_for_cloud", job_id=str(job_id), sessions=len(session_ids))


async def dispatch_cloud_fusion(job_id) -> None:
    raise NotImplementedError(
        "cloud map fusion executes on the live A100 pod via cloud/mapfusion_pod.py (GTSAM). Start the pod "
        "(make cloud-provision), then run the dispatch per the MinIO<->/workspace contract in "
        "services/hdmap/cloud.py. The local averaging-fusion fallback runs on the single box."
    )
