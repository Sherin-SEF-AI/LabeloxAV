"""Three training paths read fields off TrainingJob that are not columns on it.

TrainJobSpec is stored whole, in the job's `config` JSONB. `dataset_spec`, `hparams` and `notes` are keys
inside it, not attributes of the row, and three call sites read them as attributes:

  * GET /training/sweep/{name} filtered on TrainingJob.notes and read j.hparams, so a sweep could be started
    and its ranking could never be read back.
  * dispatch_cloud_job read j.dataset_spec and j.hparams, so every cloud dispatch died on its first line,
    before a pod was contacted.
  * attach_run read job.hparams and job.dataset_spec, so attaching any finished job to an experiment failed.

Each raised AttributeError at request time rather than at import, which is why they survived: the code
looks right and only a real call finds out.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import TrainingJob
from db.session import get_sessionmaker

SPEC = {
    "purpose": "perception",
    "task_type": "detection",
    "dataset_spec": {"cities": ["BLR"], "min_conf": 0.4},
    "hparams": {"epochs": 3, "imgsz": 960},
    "notes": "sweep 'jobcfg' trial 1/2 {'epochs': 3}",
}


async def _job(**over) -> TrainingJob:
    async with get_sessionmaker()() as db:
        job = TrainingJob(job_id=uuid.uuid4(), status=over.pop("status", "done"), purpose="perception",
                          task_type="detection", compute_target="local",
                          config={**SPEC, **over.pop("config", {})},
                          metrics=over.pop("metrics", {}), counts={}, result={}, progress=1.0)
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job


@pytest.mark.asyncio
async def test_a_sweep_can_be_read_back_after_it_is_started():
    from services.api.routers.training import sweep_status

    job = await _job(metrics={"candidate": {"map50": 0.42}})
    out = await sweep_status("jobcfg", metric="map50")

    assert out["sweep"] == "jobcfg"
    ids = [t["job_id"] for t in out["trials_detail"]]
    assert str(job.job_id) in ids


@pytest.mark.asyncio
async def test_a_sweep_name_with_a_wildcard_does_not_match_every_other_sweep():
    """`like` treats % as "anything", so an unescaped name would rank other people's trials as its own."""
    from services.api.routers.training import sweep_status

    await _job(config={"notes": "sweep 'other' trial 1/1 {}"})
    out = await sweep_status("%", metric="map50")
    assert out["trials_detail"] == []


@pytest.mark.asyncio
async def test_a_cloud_dispatch_gets_past_reading_its_own_spec():
    """The next thing it does is demand a built dataset directory, which is the real precondition. Before
    the fix it never reached that: it raised AttributeError on the line that reads the spec."""
    from services.training.cloud import dispatch_cloud_job

    job = await _job(status="pending")
    with pytest.raises(ValueError, match="dataset directory"):
        await dispatch_cloud_job(job.job_id, dataset_dir=None)


@pytest.mark.asyncio
async def test_attaching_a_job_to_an_experiment_carries_its_hyperparameters():
    """The hyperparameters are the whole point of the record: an experiment run without them cannot be
    compared to anything."""
    from services.training.experiments import attach_run

    job = await _job(metrics={"candidate": {"map50": 0.51}})
    async with get_sessionmaker()() as db:
        out = await attach_run(db, experiment=f"exp-{uuid.uuid4().hex[:8]}", job_id=str(job.job_id))

    assert out["hparams"] == {"epochs": 3, "imgsz": 960}
    assert out["dataset_spec"] == {"cities": ["BLR"], "min_conf": 0.4}
    assert out["metrics"]["map50"] == 0.51
