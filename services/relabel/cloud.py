"""Cloud relabel seam (M4.2): route bulk champion-model re-inference to the RunPod A100. compute_target=
'cloud' parks the RelabelJob visibly; the local API never runs it. Executing it runs cloud/relabel_pod.py
over the dataset's frames and returns relabeled.jsonl, which services.relabel.run.ingest_model_relabel
applies through the diff + apply path. Mirrors services/autolabel/cloud.py.

Data contract (MinIO <-> pod /workspace):
  1. ensure the pod is up (RunPod API; volume + venv + checkpoints persist across stops).
  2. push the champion weights + the dataset's frames + a manifest to /workspace/in.
  3. ssh: `python cloud/relabel_pod.py --weights in/champion.pt --manifest in/manifest.jsonl --out out/relabeled.jsonl`
  4. pull out/relabeled.jsonl; ingest_model_relabel(model_version, output) applies safe improvements,
     routes conflicts/regressions to review, lands on a new lakeFS branch.
  5. stop the pod to cap billing.
"""

from __future__ import annotations

import uuid

from core.logging import get_logger
from db.models import RelabelJob
from db.session import get_sessionmaker

log = get_logger("relabel_cloud")

_NOTE = (
    "Queued for the cloud A100. Bulk re-inference runs the champion model there (cloud/relabel_pod.py). "
    "Start the pod (make cloud-provision) and run `make cloud-relabel JOB=<id>` to execute it; the local "
    "API will not run a cloud job. Ontology-promotion relabeling runs locally without the pod."
)


async def mark_queued_for_cloud_relabel(job_id, model_version, session_ids) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(RelabelJob, uuid.UUID(str(job_id)))
        if j is None:
            return
        j.status = "pending"
        j.stage = "queued-cloud"
        j.counts = {"compute_target": "cloud", "model_version": model_version,
                    "sessions": len(session_ids or []), "note": _NOTE}
        await db.commit()
    log.info("relabel.queued_for_cloud", job_id=str(job_id), model_version=model_version)


async def dispatch_cloud_relabel(job_id) -> None:
    raise NotImplementedError(
        "cloud relabel executes on the live A100 pod via cloud/relabel_pod.py. Start the pod "
        "(make cloud-provision), then run the dispatch per the MinIO<->/workspace contract in "
        "services/relabel/cloud.py. Ontology-promotion relabeling runs locally on the single box."
    )
