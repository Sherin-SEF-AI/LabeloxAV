"""GPU-accelerated kernels for the perception hot paths, capability-gated: they run on the GPU through
torch/Triton when a CUDA device is present, and most carry an identical NumPy path used as both the
correctness oracle and the CPU fallback.

Not every kernel has a NumPy fallback. The torch-only kernels raise RuntimeError when torch is unavailable
rather than silently degrading: phash_batch, image_quality_batch, preprocess_nv12_batch, the maskops
morphology (dilate/erode/morph_open/morph_close), warp_masks_by_flow, and apply_map_batch depend on
torch interpolate/conv primitives that have no pure-NumPy equivalent here. Callers on a CPU-only box must
gate on gpu_available() (or catch the RuntimeError) for those.

Tier 1: fused projection (world -> pixel), all-pairs mask IoU (Triton bit-packed), fused auto-label preprocess.
Tier 2: multi-view epipolar consistency, box IoU + NMS, pseudo-GT cross-path agreement.
Tier 3: calibration residual field, batched image-QA metrics, undistort/rectify LUT (fused into preprocess).
Tier 4: prediction-to-GT matching for mAP, confusion + error-slice aggregation.
Active learning: per-detection entropy + margin, ensemble BALD disagreement, temporal flicker (ORACLYX ->
VERDYX triage signal)."""

from core.accel.agreement import agreement_matrix, consensus_scores
from core.accel.boxes import box_iou_matrix, nms
from core.accel.calib_frontend import descriptor_match, refine_corners
from core.accel.curation import (
    cosine_sim_matrix,
    hamming_matrix,
    kcenter_greedy,
    nearest_neighbor_sim,
    phash_batch,
)
from core.accel.geometry_mv import best_epipolar_match, sampson_matrix
from core.accel.imgqa import image_quality_batch, rig_exposure_consistency
from core.accel.mask_iou import mask_iou_matrix
from core.accel.maskops import (
    boundary_tightness,
    dilate,
    erode,
    morph_close,
    morph_open,
    remove_small_components,
    rle_decode,
    rle_encode,
)
from core.accel.matching import match_detections
from core.accel.preprocess import preprocess_nv12_batch
from core.accel.projection import gpu_available, project_cam_batch, project_world_batch
from core.accel.propagate import (
    interp_boxes,
    slerp_batch,
    warp_boxes_by_flow,
    warp_masks_by_flow,
)
from core.accel.residual import reprojection_residuals
from core.accel.sensorqa import equalize_hist_batch, motion_blur_score
from core.accel.slices import confusion_matrix, slice_precision, slice_recall
from core.accel.uncertainty import ensemble_disagreement, entropy_margin, flicker_scores
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
    "entropy_margin", "ensemble_disagreement", "flicker_scores",
    "cosine_sim_matrix", "nearest_neighbor_sim", "phash_batch", "hamming_matrix", "kcenter_greedy",
    "slerp_batch", "interp_boxes", "warp_boxes_by_flow", "warp_masks_by_flow",
    "dilate", "erode", "morph_open", "morph_close", "remove_small_components",
    "rle_encode", "rle_decode", "boundary_tightness",
    "refine_corners", "descriptor_match",
    "equalize_hist_batch", "motion_blur_score",
]
