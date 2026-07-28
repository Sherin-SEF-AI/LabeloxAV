"""Training task plugins. Importing this package registers the built-in tasks into the TASKS registry."""

from services.training.tasks import (
    classification,  # noqa: F401  (registers ClassificationTask)
    detect3d,  # noqa: F401  (registers Detection3dTask)
    detection,  # noqa: F401  (registers DetectionTask)
    lane,  # noqa: F401  (registers LaneTask)
    pose,  # noqa: F401  (registers PoseTask)
    segmentation,  # noqa: F401  (registers SegmentationTask)
)
from services.training.tasks.base import TASKS, get_task, list_tasks

__all__ = ["TASKS", "get_task", "list_tasks"]
