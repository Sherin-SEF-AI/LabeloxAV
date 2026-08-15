"""Experiment tracking, in the loop rather than beside it.

The per-job metric curve answered "how did this run go". It could not answer "is this line of work getting
better", which is the question actually asked between iterations: comparing two runs meant reading two job
rows and remembering which hyperparameters went with which.

The usual answer is wandb or mlflow. Both would put the loop's own history outside the loop, where the
promotion gate, the flywheel controller and the slice evaluator cannot read it, and would add a network
dependency to the training path. Since every number involved is already in Postgres, the tracker is a table
and a comparison query.

Two decisions worth stating:

- **A run record is denormalised from its job.** A job row is mutable operational state; an experiment run
  is a fixed claim about a finished run. Reading comparisons off mutable rows would let a later job's
  in-place edit silently change what an earlier comparison said.
- **The comparison names what varied.** A diff of every hyperparameter at once is not a finding. The
  comparison reports only the keys that actually differ, so "this run changed lr and nothing else" is
  visible rather than inferred.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Experiment, ExperimentRun, TrainingJob

log = get_logger("experiments")

# The metrics worth ranking on, in the direction that counts as better. Explicit because "higher is better"
# is not universal: a loss and an error rate both improve downward, and ranking them the wrong way would
# quietly promote the worst run in the set.
HIGHER_IS_BETTER = {"map50", "map", "top1", "top5", "precision", "recall", "map50_mask",
                    "map50_pose", "safe_miou", "mota", "idf1", "hota"}
LOWER_IS_BETTER = {"loss", "val_loss", "ece", "id_switches", "fragmentations"}


async def create_experiment(db: AsyncSession, *, name: str, task_type: str = "detection",
                            description: str | None = None, hypothesis: str | None = None,
                            tags: list[str] | None = None, created_by: str | None = None) -> dict:
    existing = (await db.execute(
        select(Experiment).where(Experiment.name == name))).scalar_one_or_none()
    if existing is not None:
        return _exp_dict(existing)
    row = Experiment(name=name, task_type=task_type, description=description,
                     hypothesis=hypothesis, tags=list(tags or []), created_by=created_by)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("experiment.created", name=name, task=task_type)
    return _exp_dict(row)


async def attach_run(db: AsyncSession, *, experiment: str, job_id: str,
                     label: str | None = None, baseline_run_id: str | None = None,
                     notes: str | None = None) -> dict:
    """Record a training job as a run of an experiment, copying the numbers that make it comparable.

    Called after the job finishes so the record is of a finished run. Attaching a running job would store a
    partial curve as though it were the result.
    """
    exp = (await db.execute(
        select(Experiment).where(Experiment.name == experiment))).scalar_one_or_none()
    if exp is None:
        exp_dict = await create_experiment(db, name=experiment)
        exp = await db.get(Experiment, uuid.UUID(exp_dict["experiment_id"]))

    job = await db.get(TrainingJob, uuid.UUID(job_id))
    if job is None:
        raise ValueError(f"training job {job_id} not found")

    metrics = dict((job.metrics or {}).get("candidate") or (job.metrics or {}).get("final") or {})
    curve = list((job.metrics or {}).get("curve") or [])
    row = ExperimentRun(
        experiment_id=exp.experiment_id, job_id=job.job_id,
        label=label or job.purpose or str(job.job_id)[:8],
        # Both come out of config, which holds the whole TrainJobSpec; as attributes they do not
        # exist, and attaching any job to an experiment raised AttributeError.
        hparams=dict((job.config or {}).get("hparams") or {}),
        dataset_spec=dict((job.config or {}).get("dataset_spec") or {}),
        metrics=metrics, curve=curve,
        gold_id=(job.metrics or {}).get("gold_id"),
        baseline_run_id=uuid.UUID(baseline_run_id) if baseline_run_id else None,
        status=job.status, notes=notes,
        finished_at=datetime.now(UTC) if job.status in ("done", "failed", "cancelled") else None,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("experiment.run_attached", experiment=experiment, job=job_id, status=job.status)
    return _run_dict(row)


async def list_experiments(db: AsyncSession, *, task_type: str | None = None,
                           limit: int = 100) -> dict:
    stmt = select(Experiment).order_by(Experiment.created_at.desc()).limit(min(max(limit, 1), 500))
    if task_type:
        stmt = stmt.where(Experiment.task_type == task_type)
    rows = (await db.execute(stmt)).scalars().all()

    counts = {}
    if rows:
        counts = dict((await db.execute(
            select(ExperimentRun.experiment_id, func.count())
            .where(ExperimentRun.experiment_id.in_([r.experiment_id for r in rows]))
            .group_by(ExperimentRun.experiment_id))).all())
    return {"experiments": [{**_exp_dict(r), "runs": int(counts.get(r.experiment_id, 0))}
                            for r in rows]}


async def experiment_detail(db: AsyncSession, name: str, *, metric: str = "map50") -> dict:
    """One experiment's runs, ranked, with the best marked and what varied between runs called out."""
    exp = (await db.execute(
        select(Experiment).where(Experiment.name == name))).scalar_one_or_none()
    if exp is None:
        raise ValueError(f"experiment {name!r} not found")

    runs = (await db.execute(
        select(ExperimentRun).where(ExperimentRun.experiment_id == exp.experiment_id)
        .order_by(ExperimentRun.started_at))).scalars().all()
    dicts = [_run_dict(r) for r in runs]

    scored = [(d, d["metrics"].get(metric)) for d in dicts]
    # A run with no measurement is excluded from the ranking rather than scored zero, which would rank a
    # crashed run above a genuinely poor one.
    measured = [(d, float(v)) for d, v in scored if isinstance(v, int | float)]
    reverse = metric not in LOWER_IS_BETTER
    measured.sort(key=lambda pair: pair[1], reverse=reverse)
    best = measured[0][0]["run_id"] if measured else None

    return {
        **_exp_dict(exp),
        "metric": metric,
        "runs": dicts,
        "ranking": [{"run_id": d["run_id"], "label": d["label"], metric: v} for d, v in measured],
        "unscored_runs": len(dicts) - len(measured),
        "best_run_id": best,
        "varied": _varied_keys([d["hparams"] for d in dicts]),
    }


def _varied_keys(hparam_sets: list[dict]) -> list[str]:
    """Which hyperparameters actually differ across the runs.

    This is what makes a comparison a finding rather than a diff: "lr changed and nothing else" is a
    statement about cause, while a table of forty identical columns is not.
    """
    if len(hparam_sets) < 2:
        return []
    keys = set().union(*(set(h) for h in hparam_sets))
    varied = []
    for k in sorted(keys):
        seen = {repr(h.get(k)) for h in hparam_sets}
        if len(seen) > 1:
            varied.append(k)
    return varied


async def compare_runs(db: AsyncSession, run_a: str, run_b: str) -> dict:
    """Two runs side by side: what differed in the inputs, and what it did to the outputs."""
    a = await db.get(ExperimentRun, uuid.UUID(run_a))
    b = await db.get(ExperimentRun, uuid.UUID(run_b))
    if a is None or b is None:
        raise ValueError("one or both runs not found")

    hp_diff = {}
    for k in sorted(set(a.hparams or {}) | set(b.hparams or {})):
        va, vb = (a.hparams or {}).get(k), (b.hparams or {}).get(k)
        if va != vb:
            hp_diff[k] = {"a": va, "b": vb}

    m_diff = {}
    for k in sorted(set(a.metrics or {}) | set(b.metrics or {})):
        va, vb = (a.metrics or {}).get(k), (b.metrics or {}).get(k)
        entry: dict = {"a": va, "b": vb}
        if isinstance(va, int | float) and isinstance(vb, int | float):
            entry["delta"] = round(float(vb) - float(va), 6)
            # Direction is stated rather than implied by the sign, because for a loss the sign means the
            # opposite of what it means for an mAP.
            entry["better"] = ("b" if (float(vb) > float(va)) == (k not in LOWER_IS_BETTER)
                               else "a") if va != vb else None
        m_diff[k] = entry

    return {"a": _run_dict(a), "b": _run_dict(b),
            "hparam_diff": hp_diff, "metric_diff": m_diff,
            "dataset_changed": (a.dataset_spec or {}) != (b.dataset_spec or {}),
            # A comparison across different gold sets is not a comparison. Flagged rather than blocked,
            # because sometimes that is exactly what is being examined.
            "same_gold": a.gold_id == b.gold_id}


def _exp_dict(e: Experiment) -> dict:
    return {"experiment_id": str(e.experiment_id), "name": e.name, "task_type": e.task_type,
            "description": e.description, "hypothesis": e.hypothesis, "tags": e.tags or [],
            "created_by": e.created_by,
            "created_at": e.created_at.isoformat() if e.created_at else None}


def _run_dict(r: ExperimentRun) -> dict:
    return {"run_id": str(r.run_id), "experiment_id": str(r.experiment_id),
            "job_id": str(r.job_id) if r.job_id else None, "label": r.label,
            "hparams": r.hparams or {}, "dataset_spec": r.dataset_spec or {},
            "metrics": r.metrics or {}, "curve": r.curve or [],
            "gold_id": r.gold_id, "status": r.status, "notes": r.notes,
            "baseline_run_id": str(r.baseline_run_id) if r.baseline_run_id else None,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None}
