"""Training task plugins. Importing this package registers the built-in tasks into the TASKS registry."""

from services.training.tasks import detection  # noqa: F401  (registers DetectionTask)
from services.training.tasks.base import TASKS, get_task, list_tasks

__all__ = ["TASKS", "get_task", "list_tasks"]
