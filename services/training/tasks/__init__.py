"""Training task plugins. Importing this package registers the built-in tasks into the TASKS registry.

It also turns off Ultralytics' third-party experiment-tracking integrations, which are on by default and
fire from inside the training loop. The MLflow one raises on this host, because MLflow now refuses a
filesystem tracking backend, and the exception surfaces as a broken pipe from every dataloader worker
rather than as anything naming MLflow. This engine records its own runs in `model_run` and
`training_job`, so none of these integrations has a job to do here; leaving them armed only gives
training a way to die that has nothing to do with training.
"""


from services.training.tasks import (
    classification,  # noqa: F401  (registers ClassificationTask)
    detect3d,  # noqa: F401  (registers Detection3dTask)
    detection,  # noqa: F401  (registers DetectionTask)
    lane,  # noqa: F401  (registers LaneTask)
    pose,  # noqa: F401  (registers PoseTask)
    pretrain,  # noqa: F401  (registers PretrainTask)
    segmentation,  # noqa: F401  (registers SegmentationTask)
    selftrain,  # noqa: F401  (registers SelfTrainTask)
)
from services.training.tasks.base import TASKS, get_task, list_tasks


def _silence_third_party_trackers() -> None:
    try:
        from ultralytics import settings as _ul_settings

        off = {k: False for k in ("mlflow", "clearml", "comet", "dvc", "neptune", "raytune",
                                  "wandb", "hub")
               if _ul_settings.get(k)}
        if off:
            _ul_settings.update(off)
    except Exception:  # noqa: BLE001 - no ultralytics, or a settings file that cannot be written
        pass


_silence_third_party_trackers()

__all__ = ["TASKS", "get_task", "list_tasks"]
