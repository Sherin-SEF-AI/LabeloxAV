"""Cloud compute seam: route heavy training/labeling to the RunPod A100 (the hybrid "cloud" target).

This is the SEAM and the contract, not a fake executor. The local worker never claims cloud jobs
(see worker._claim), so a compute_target='cloud' job parks in a clear "queued for cloud" state until
the pod runs it. The full remote execution is the bounded next directive; its contract is documented
here so it is unambiguous to implement.

Data-movement contract (MinIO <-> pod /workspace), to be realized in the next directive:
  1. ensure the pod is up: `cloud/provision_runpod.sh` (idempotent; restarts the stopped pod or creates one).
  2. export the dataset slice from MinIO (services.export.dataset / the YOLO builder) and push it to
     /workspace/data on the network volume (rclone/s3 or huggingface_hub upload).
  3. run training on the pod (cloud/ entrypoint: ultralytics YOLO26 fine-tune, or the SAM3.1 + Qwen3-VL
     labeling sweep), streaming progress back to the training_job row via the API.
  4. pull the resulting weights from /workspace back into MinIO (store.put_file) and write a model_run
     row (purpose/task_type/job_id) so the cloud-trained model joins the same registry as local ones.
  5. stop the pod (runpodctl pod stop) to cap billing; the volume + venv + checkpoints persist.
"""

from __future__ import annotations

import uuid

from core.logging import get_logger
from db.models import TrainingJob
from db.session import get_sessionmaker

log = get_logger("training_cloud")

_QUEUED_NOTE = (
    "Queued for the cloud A100. Provision the pod with `make cloud-provision`, then run the cloud "
    "dispatch (next directive). The local worker will not run this job."
)


async def mark_queued_for_cloud(job_id) -> None:
    """Park a cloud job in a clear, visible state. The local worker skips it (compute_target filter)."""
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(str(job_id)))
        if j is None:
            return
        j.stage = "queued-cloud"
        j.result = {"note": _QUEUED_NOTE}
        await db.commit()
    log.info("training.queued_for_cloud", job_id=str(job_id))


async def dispatch_cloud_job(job_id) -> None:
    """Entry point for the cloud executor (next directive). Intentionally not wired this turn: it
    documents the contract above and fails loud rather than pretending to train."""
    raise NotImplementedError(
        "cloud dispatch is the next directive. Provision the pod (make cloud-provision) and implement "
        "the MinIO <-> /workspace data sync per the contract in services/training/cloud.py."
    )
