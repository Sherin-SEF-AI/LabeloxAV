"""Shared types for the three label paths. RawDetection is the in-memory proposal (it can carry
a numpy mask, which the Pydantic Detection schema cannot), produced before fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RawDetection:
    path: str                       # path_a_yolo26 | path_b_sam3 | path_c_qwen3vl
    bbox: tuple[float, float, float, float]  # xyxy pixel
    conf: float
    model_version: str
    class_name: str | None = None
    class_id: int | None = None
    mask: np.ndarray | None = None  # bool HxW, in-memory only
    extra: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
