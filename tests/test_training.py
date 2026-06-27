"""In-app training platform tests. Pure: task registry + TrainJobSpec. Infra: enqueue + worker claim
+ crash recovery. GPU+opt-in: a 1-epoch lifecycle smoke (slow; needs a GPU and corpus data)."""

from __future__ import annotations

import os
import uuid

import pytest

from core.config import get_settings
from services.training.jobs import TrainJobSpec
from services.training.tasks import get_task, list_tasks


def test_task_registry():
    types = {t["task_type"] for t in list_tasks()}
    assert "detection" in types
    task = get_task("detection")
    assert task.task_type == "detection"
    assert isinstance(task.default_base_weights(), str)
    with pytest.raises(ValueError):
        get_task("does-not-exist")


def test_trainjobspec_defaults():
    spec = TrainJobSpec(purpose="vru-detector", dataset_spec={"include_classes": ["pedestrian"]})
    assert spec.task_type == "detection"
    assert spec.promote is False
    d = spec.model_dump()
    assert d["dataset_spec"]["include_classes"] == ["pedestrian"]


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
@pytest.mark.asyncio
async def test_enqueue_claim_and_recovery():
    from db.models import TrainingJob
    from db.session import get_sessionmaker
    from services.training.jobs import enqueue_job
    from services.training.worker import _claim, _reset_orphans

    job_id = await enqueue_job(TrainJobSpec(purpose="unit-test-line", dataset_spec={"limit": 5}))

    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
        assert j is not None and j.status == "pending" and j.purpose == "unit-test-line"
        assert j.config["dataset_spec"]["limit"] == 5

    # the worker claims the oldest pending and marks it running (atomic)
    claimed = await _claim()
    assert claimed is not None
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(claimed))
        assert j.status == "running"

    # crash recovery resets orphaned running jobs back to pending
    await _reset_orphans()
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(claimed))
        assert j.status == "pending"
        await db.delete(j)  # cleanup any rows we claimed this run
        # also clean our enqueued row if different
        if claimed != job_id:
            other = await db.get(TrainingJob, uuid.UUID(job_id))
            if other:
                await db.delete(other)
        await db.commit()


@requires_infra
@pytest.mark.asyncio
async def test_cloud_job_not_claimed_by_local_worker():
    from db.models import TrainingJob
    from db.session import get_sessionmaker
    from services.training.jobs import enqueue_job
    from services.training.worker import _claim

    # a cloud job parks for the pod and is never claimed by the local worker
    cloud_id = await enqueue_job(TrainJobSpec(purpose="cloud-test-line", compute_target="cloud",
                                              dataset_spec={"limit": 5}))
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(cloud_id))
        assert j.compute_target == "cloud" and j.stage == "queued-cloud"

    # a local job sitting behind it still gets claimed (cloud one is skipped)
    local_id = await enqueue_job(TrainJobSpec(purpose="local-test-line", compute_target="local",
                                              dataset_spec={"limit": 5}))
    claimed = await _claim()
    assert claimed == local_id  # the local worker skips the cloud job, claims the local one

    async with get_sessionmaker()() as db:
        cj = await db.get(TrainingJob, uuid.UUID(cloud_id))
        assert cj.status == "pending"  # cloud job untouched by the local worker
        # cleanup
        for jid in (cloud_id, local_id):
            row = await db.get(TrainingJob, uuid.UUID(jid))
            if row:
                await db.delete(row)
        await db.commit()


def _gpu_train_enabled() -> bool:
    if os.environ.get("LBX_TEST_TRAIN") != "1":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available()) and _infra_up()
    except Exception:
        return False


requires_gpu_train = pytest.mark.skipif(
    not _gpu_train_enabled(), reason="set LBX_TEST_TRAIN=1 + GPU + infra to run the real training smoke"
)


@requires_gpu_train
@pytest.mark.asyncio
async def test_run_job_lifecycle_1epoch():
    from sqlalchemy import func, select

    from db.models import ModelRun, Object, TrainingJob
    from db.session import get_sessionmaker
    from services.training.jobs import enqueue_job, run_job

    async with get_sessionmaker()() as db:
        n = (await db.execute(select(func.count()).select_from(Object).where(Object.state != "rejected"))).scalar_one()
    if n < 20:
        pytest.skip("need corpus objects to train")

    job_id = await enqueue_job(TrainJobSpec(
        purpose="smoke-detector", dataset_spec={"max_per_class": 30}, hparams={"epochs": 1, "imgsz": 640, "batch": 4},
    ))
    result = await run_job(job_id)
    assert "run_id" in result
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
        assert j.status == "done" and j.progress == 1.0
        mr = await db.get(ModelRun, result["run_id"])
        assert mr is not None and mr.purpose == "smoke-detector" and mr.task_type == "detection"
