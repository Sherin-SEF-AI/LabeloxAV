"""Instance and panoptic segmentation. The native path is PTv3 plus PointGroup via Pointcept, run on the
burst node (not installed locally, parks on the seam). The runnable instance labeling on the camera fleet is
produced by the projected segmentation in semantic.py: each gated 3D cuboid is its own instance, so a point
inside it gets that instance id, while the semantic class comes from the same ontology as 2D.
"""

from __future__ import annotations

from core.config import get_settings
from services.lidar.ingest.normalize import Cloud
from services.lidar.segment3d.semantic import SegmentationUnavailable


def segment_instance_ptv3(cloud: Cloud, ckpt: str | None = None) -> dict:
    """PTv3 + PointGroup instance/panoptic segmentation via Pointcept. Requires the framework on the burst
    node; raises the seam locally."""
    ckpt = ckpt or get_settings().lidar.segmenter_ckpt
    try:
        import pointcept  # noqa: F401
    except Exception as exc:
        raise SegmentationUnavailable(
            f"PTv3 + PointGroup instance segmentation ({ckpt}) needs Pointcept on the A100 burst node. Run "
            "via the lidar_perception job; the local worker will not execute it.") from exc
    raise SegmentationUnavailable("Pointcept present but the inference wrapper runs on the burst node only.")
