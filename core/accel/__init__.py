"""GPU-accelerated kernels for the perception hot paths, capability-gated with a NumPy reference as both the
correctness oracle and the fallback (runs on the GPU through torch/Triton when a CUDA device is present, else
the identical NumPy math).

Tier 1: fused projection (world -> pixel), all-pairs mask IoU (Triton bit-packed), fused auto-label preprocess.
Tier 2: multi-view epipolar consistency, box IoU + NMS, pseudo-GT cross-path agreement.
Tier 3: calibration residual field, batched image-QA metrics, undistort/rectify LUT (fused into preprocess).
Tier 4: prediction-to-GT matching for mAP, confusion + error-slice aggregation."""

from core.accel.agreement import agreement_matrix, consensus_scores
from core.accel.boxes import box_iou_matrix, nms
from core.accel.geometry_mv import best_epipolar_match, sampson_matrix
from core.accel.imgqa import image_quality_batch, rig_exposure_consistency
from core.accel.mask_iou import mask_iou_matrix
from core.accel.matching import match_detections
from core.accel.preprocess import preprocess_nv12_batch
from core.accel.projection import gpu_available, project_cam_batch, project_world_batch
from core.accel.residual import reprojection_residuals
from core.accel.slices import confusion_matrix, slice_precision, slice_recall
from core.accel.undistort import apply_map_batch, build_fisheye_map

__all__ = [
    "gpu_available",
    "project_world_batch", "project_cam_batch",
    "mask_iou_matrix",
    "preprocess_nv12_batch",
    "sampson_matrix", "best_epipolar_match",
    "box_iou_matrix", "nms",
    "agreement_matrix", "consensus_scores",
    "reprojection_residuals",
    "image_quality_batch", "rig_exposure_consistency",
    "build_fisheye_map", "apply_map_batch",
    "match_detections",
    "confusion_matrix", "slice_recall", "slice_precision",
]
