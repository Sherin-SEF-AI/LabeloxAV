"""TrainingTask plugin interface + registry. A "purpose" is a dataset_spec + hparams; a new task type
(detection now; segmentation/classification later) is a plugin implementing this protocol. The shared
executor (services/training/jobs.run_job) drives build -> train -> evaluate -> gate for any task.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

# Progress is reported as a dict, e.g. {"stage": "train", "epoch": 3, "total_epochs": 20,
# "metrics": {"map50": 0.41}}. The executor persists it onto the training_job row.
ProgressFn = Callable[[dict], None]


@runtime_checkable
class TrainingTask(Protocol):
    task_type: str

    def default_base_weights(self) -> str: ...

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        """Return a dataset dict with at least: data_yaml, classes, n_train_images, n_val_images."""
        ...

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        """Train (blocking; runs in the worker's executor thread). Return the weights path."""
        ...

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict: ...

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict: ...


TASKS: dict[str, TrainingTask] = {}


def register(task: TrainingTask) -> TrainingTask:
    TASKS[task.task_type] = task
    return task


def get_task(task_type: str) -> TrainingTask:
    if task_type not in TASKS:
        raise ValueError(f"unknown task_type {task_type!r}; registered: {sorted(TASKS)}")
    return TASKS[task_type]


def list_tasks() -> list[dict]:
    return [{"task_type": t.task_type, "default_base_weights": t.default_base_weights()} for t in TASKS.values()]
