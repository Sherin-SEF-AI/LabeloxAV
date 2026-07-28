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


async def dispatch_cloud_job(job_id, *, dataset_dir=None, entrypoint: str = "train_yolo") -> dict:
    """Publish a job's workspace so a pod can claim it, and mark the job dispatched.

    Steps 2 and 4 of the contract above, which is the half that runs on this side of the network. Steps 1,
    3 and 5 are the pod's, and they need a provisioned GPU that this environment does not have.

    Deliberately not a fake executor. Nothing here claims the job trained: it publishes the dataset and the
    manifest, and the job stays visibly awaiting a pod. When a pod picks it up and pushes results back,
    `collect_cloud_results` brings the weights into the same registry as a local run.
    """
    from pathlib import Path

    from services.cloud.transfer import push_workspace

    job_id = str(job_id)
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
        if j is None:
            raise ValueError(f"training job {job_id} not found")
        spec = dict(j.dataset_spec or {})
        hparams = dict(j.hparams or {})
        task_type = j.task_type

    if dataset_dir is None:
        raise ValueError(
            "dispatch_cloud_job needs the built dataset directory. Build it locally first (the task "
            "plugin's build_dataset), then dispatch: the pod trains, it does not query this database.")

    pushed = push_workspace(
        Path(dataset_dir), job_id=job_id, kind="training", entrypoint=entrypoint,
        args={"task_type": task_type, "hparams": hparams, "dataset_spec": spec})

    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
        if j is not None:
            j.stage = "dispatched-cloud"
            j.result = {"note": "workspace published; awaiting a pod", **pushed}
            await db.commit()
    log.info("training.dispatched_to_cloud", job_id=job_id, **pushed)
    return pushed


async def collect_cloud_results(job_id) -> dict:
    """Bring a finished cloud run's weights home, so it joins the same registry as a local run.

    Step 4 of the contract. Returns without changing the job when nothing has been pushed yet, because a pod
    that has not finished is the normal case rather than an error.
    """
    from services.cloud.transfer import fetch_results, weights_uri

    job_id = str(job_id)
    results = fetch_results(job_id, "training")
    if not results["ready"]:
        return {"ready": False, "job_id": job_id, "detail": "the pod has not pushed results yet"}

    uri = weights_uri(job_id, "training")
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
        if j is None:
            raise ValueError(f"training job {job_id} not found")
        j.stage = "done"
        j.status = "done"
        j.result = {**(j.result or {}), "weights_uri": uri, "artifacts": results["files"]}
        await db.commit()
    log.info("training.cloud_results_collected", job_id=job_id, weights=uri)
    return {"ready": True, "job_id": job_id, "weights_uri": uri,
            "artifacts": len(results["files"])}
