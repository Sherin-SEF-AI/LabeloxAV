"""Cloud autolabel seam: route a session's heavy autolabel to the RunPod A100. The heavy stack there is
VERIFIED (cloud/smoke_test_result.json: PASS) - YOLO26, SAM 3.1 PCS (text-promptable, Sam3Processor),
and Qwen3-VL all load and infer on the A100. This module wires the dispatch.

compute_target='cloud' parks the AutolabelJob in a clear "queued for cloud" state; the local API never
runs it (start() branches on compute_target). Executing it runs the pod-side entrypoint
`cloud/autolabel_pod.py` (which reuses the proven smoke-test loaders) over the session's frames.

Data contract (MinIO <-> pod /workspace), mirroring services/training/cloud.py:
  1. ensure the pod is up (RunPod API; the volume + venv + 20 GB of checkpoints persist across stops).
  2. push the session's frames + a manifest to /workspace/in.
  3. ssh: `python cloud/autolabel_pod.py --manifest in/manifest.jsonl --out out/labels.jsonl`
     (Path A YOLO26 -> Path B SAM 3.1 masks -> Path C Qwen3-VL verify, the same fuse+gate as local).
  4. pull out/labels.jsonl; ingest objects + masks into Postgres/MinIO via services.autolabel.persist.
  5. stop the pod to cap billing.
"""

from __future__ import annotations

import uuid

from core.logging import get_logger
from db.models import AutolabelJob
from db.session import get_sessionmaker

log = get_logger("autolabel_cloud")

_NOTE = (
    "Queued for the cloud A100. The heavy stack is verified there (SAM 3.1 PCS + Qwen3-VL + YOLO26, "
    "smoke=PASS). Start the pod and run `make cloud-autolabel SESSION=<id>` to execute it; the local "
    "API will not run a cloud job (single-box GPU discipline)."
)


async def mark_queued_for_cloud(job_id, session_id, limit) -> None:
    """Park a cloud autolabel job visibly. It surfaces on the Jobs dashboard as queued-cloud; the local
    API process never runs it (start() only spawns _run_guarded for compute_target='local')."""
    async with get_sessionmaker()() as db:
        j = await db.get(AutolabelJob, uuid.UUID(str(job_id)))
        if j is None:
            return
        j.status = "pending"
        j.counts = {"compute_target": "cloud", "session_id": str(session_id),
                    "limit": limit, "note": _NOTE}
        await db.commit()
    log.info("autolabel.queued_for_cloud", job_id=str(job_id), session_id=str(session_id))


async def dispatch_cloud_autolabel(job_id, session_id, limit=None) -> None:
    """Executor entry for the cloud runner (realized on a live pod). Documents the contract above and
    fails loud rather than pretending to label. Invoked by `make cloud-autolabel` once the pod is up."""
    raise NotImplementedError(
        "cloud autolabel executes on the live A100 pod via cloud/autolabel_pod.py. Start the pod "
        "(make cloud-provision), then run the dispatch per the MinIO<->/workspace contract in "
        "services/autolabel/cloud.py. The heavy stack is already smoke-verified (PASS)."
    )
