"""Point Transformer V3 semantic segmentation for outdoor LiDAR (local GPU inference).

Model code: the legacy single-file PTv3 (github.com/Pointcept/PointTransformerV3, branch ``legacy``, MIT),
vendored under ``models/ptv3/src``. Weights: Pointcept's ``nuscenes-semseg-pt-v3m1-0-base`` from
huggingface.co/Pointcept/PointTransformerV3 (model card: MIT). The weights were trained on nuScenes, whose
dataset terms are CC BY-NC-SA 4.0 (non-commercial); check that before any commercial use. Pointcept publishes
no SemanticKITTI PTv3 checkpoint, so this is the nuScenes 16-class head.

``model_state.pth`` is the state_dict extracted from the published ``model_best.pth`` with a restricted
unpickler (the original pickles an LR scheduler and needs ``getattr``), so it loads with ``weights_only=True``.

Preprocessing follows the checkpoint's Pointcept config: GridSample at 0.05 m (one point per voxel),
features = (x, y, z, strength) with strength in 0..1 (nuScenes intensity / 255), voxel predictions mapped back
to every input point. No test-time augmentation. FlashAttention is not used (``enable_flash=False``); the
training patch size of 1024 is kept so attention context matches training. Inference runs in fp32 (see
``infer``). The Block forward is pinned to the Dec-2023 code the checkpoint was trained with (see
``_block_forward_dec2023``); without that the model runs but its predictions are garbage.

Domain note: nuScenes intensity/255 is much darker than KITTI (HDL-64) reflectance (median about 0.05 vs 0.27).
On six KITTI raw scans, ``strength_scale=0.3`` raised the share of ego-corridor ground labelled
driveable_surface from 5-12% to 71-84% on the three scans where it was failing, with little change elsewhere.
That is an unlabelled sanity check, not a measured mIoU.

Heavy imports (torch, spconv, torch_scatter) stay inside functions so this module imports anywhere.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
PTV3_SRC = _REPO / "models" / "ptv3" / "src"
CKPT_DIR = _REPO / "models" / "ptv3" / "nuscenes-semseg-pt-v3m1-0-base"
CHECKPOINT = CKPT_DIR / "model_state.pth"
CHECKPOINT_SOURCE = (
    "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/"
    "nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth"
)

CLASS_NAMES: list[str] = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface",
    "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]

GRID_SIZE = 0.05
PATCH_SIZE = 1024
_PKG = "_ptv3_legacy"

# Backbone exactly as the checkpoint's config (PT-v3m1), with flash disabled.
_BACKBONE_CFG = dict(
    in_channels=4,
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(PATCH_SIZE,) * 5,
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(PATCH_SIZE,) * 4,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=False,   # deterministic at inference
    pre_norm=True,
    enable_rpe=False,
    enable_flash=False,
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
)

_model = None
_model_device: str | None = None
_lock = threading.Lock()


def _import_model_module():
    """Import the vendored legacy ``model.py`` as a package (it uses a relative import)."""
    if f"{_PKG}.model" in sys.modules:
        return sys.modules[f"{_PKG}.model"]
    import importlib

    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(PTV3_SRC)]
        sys.modules[_PKG] = pkg
    return importlib.import_module(f"{_PKG}.model")


def available() -> tuple[bool, str]:
    """Cheap check: dependencies import and the checkpoint is on disk. No GPU work."""
    if not (PTV3_SRC / "model.py").is_file():
        return False, f"PTv3 model code missing at {PTV3_SRC}"
    if not CHECKPOINT.is_file():
        return False, f"checkpoint missing at {CHECKPOINT}"
    for mod in ("torch", "spconv.pytorch", "torch_scatter", "timm", "addict"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            return False, f"cannot import {mod}: {e}"
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"
    except Exception as e:  # noqa: BLE001
        return False, f"torch error: {e}"
    return True, "ok"


def _block_forward_dec2023(self, point):
    """PTv3 Block.forward as it was when this checkpoint was trained (Pointcept 314afb3, 2023-12-31).

    Pointcept commit 74c4fc1 (2024-03-26, "Fix PTv3 structure") changed the last line from
    ``point.sparse_conv_feat.replace_feature(point.feat)`` (a no-op: the result was discarded) to an
    assignment. The nuScenes checkpoint was trained before that fix, so the next block's CPE sparse conv saw
    this block's MLP output (pre-residual) rather than the residual sum. Running the fixed code on these
    weights collapses the predictions (ego-lane ground came out 88% sidewalk on a real nuScenes scan, cars
    0.1% of points, mean confidence 0.54 vs 0.98 after this patch); the original behaviour is reproduced here.
    """
    shortcut = point.feat
    point = self.cpe(point)
    point.feat = shortcut + point.feat
    shortcut = point.feat
    if self.pre_norm:
        point = self.norm1(point)
    point = self.drop_path(self.attn(point))
    point.feat = shortcut + point.feat
    if not self.pre_norm:
        point = self.norm1(point)
    shortcut = point.feat
    if self.pre_norm:
        point = self.norm2(point)
    point = self.drop_path(self.mlp(point))
    point.feat = shortcut + point.feat
    if not self.pre_norm:
        point = self.norm2(point)
    point.sparse_conv_feat.replace_feature(point.feat)  # deliberately not assigned (see docstring)
    return point


def _load(device: str):
    global _model, _model_device
    with _lock:
        if _model is not None and _model_device == device:
            return _model
        import torch
        import torch.nn as nn

        mod = _import_model_module()
        mod.Block.forward = _block_forward_dec2023

        class Segmentor(nn.Module):
            """DefaultSegmentorV2 equivalent: backbone + Linear(64 -> 16)."""

            def __init__(self):
                super().__init__()
                self.backbone = mod.PointTransformerV3(**_BACKBONE_CFG)
                self.seg_head = nn.Linear(64, len(CLASS_NAMES))

            def forward(self, data):
                point = self.backbone(data)
                return self.seg_head(point.feat)

        model = Segmentor()
        ck = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)  # raises on any key/shape mismatch
        del sd, ck
        # SerializedPooling defaults to shuffling its serialization orders even in eval (torch.randperm),
        # which makes repeated runs differ by several percent of points. Fix the order for reproducibility.
        for m in model.modules():
            if hasattr(m, "shuffle_orders"):
                m.shuffle_orders = False
        model.to(device).eval()
        _model, _model_device = model, device
        return model


def release() -> None:
    """Drop the cached model and return its VRAM."""
    global _model, _model_device
    with _lock:
        _model = None
        _model_device = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _grid_sample(xyz: np.ndarray, grid_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointcept GridSample (mode='train'): one point per voxel.

    Returns (keep_idx, grid_coord of kept points, inverse: input point -> kept row).
    """
    scaled = xyz / grid_size
    grid = np.floor(scaled).astype(np.int64)
    grid -= grid.min(0)
    _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    return first, grid[first], inverse.reshape(-1)


def infer(
    xyz: np.ndarray,
    strength: np.ndarray | None = None,
    device: str = "cuda",
    *,
    z_shift: float = 0.0,
    strength_scale: float = 1.0,
) -> dict:
    """Segment one scan.

    xyz: (N,3) float32, LiDAR frame, x forward, y left, z up, metres (sensor at origin, as in KITTI).
    strength: (N,) reflectance 0..1, or None (zeros are used).
    z_shift: added to z before inference. The model uses absolute z as a feature and was trained with the
    sensor at the origin (ground near z = -1.8); for a ground-origin ego frame pass about -1.8.
    strength_scale: multiplies strength before inference; 0.3 helps KITTI-style reflectance (module docstring).
    Points with non-finite coordinates get label -1 and conf 0.
    """
    import torch

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    if n == 0:
        return {"labels": np.zeros(0, np.int32), "conf": np.zeros(0, np.float32),
                "class_names": CLASS_NAMES, "checkpoint": CHECKPOINT_SOURCE}
    if strength is None:
        s = np.zeros((n, 1), np.float32)
    else:
        s = np.clip(np.asarray(strength, dtype=np.float32).reshape(-1, 1), 0.0, 1.0) * np.float32(strength_scale)

    # KITTI-style (x fwd, y left) -> nuScenes LIDAR_TOP (x right, y fwd). Training used full z-rotation
    # augmentation, so this is for fidelity rather than necessity.
    c = np.empty_like(xyz)
    c[:, 0] = -xyz[:, 1]
    c[:, 1] = xyz[:, 0]
    c[:, 2] = xyz[:, 2] + z_shift
    finite = np.isfinite(c).all(1)
    idx_f = np.nonzero(finite)[0]
    c_f, s_f = c[idx_f], s[idx_f]

    keep, grid_coord, inverse = _grid_sample(c_f, GRID_SIZE)
    coord_k = c_f[keep]
    feat = np.concatenate([coord_k, s_f[keep]], axis=1).astype(np.float32)

    model = _load(device)
    with torch.no_grad():
        data = {
            "coord": torch.from_numpy(coord_k).to(device),
            "grid_coord": torch.from_numpy(grid_coord).int().to(device),
            "feat": torch.from_numpy(feat).to(device),
            "offset": torch.tensor([coord_k.shape[0]], device=device, dtype=torch.long),
            "grid_size": GRID_SIZE,
        }
        # fp32 on purpose: under fp16 autocast spconv 2.3.8 receives fp16 features with fp32 weights and
        # its implicit-GEMM tuner finds no kernel ("can't find suitable algorithm for 0") on sm_120.
        logits = model(data)
        prob = torch.softmax(logits.float(), dim=1)
        conf_v, lab_v = prob.max(dim=1)
        lab_v = lab_v.int().cpu().numpy()
        conf_v = conf_v.cpu().numpy().astype(np.float32)
        del data, logits, prob

    labels = np.full(n, -1, np.int32)
    conf = np.zeros(n, np.float32)
    labels[idx_f] = lab_v[inverse]
    conf[idx_f] = conf_v[inverse]
    return {"labels": labels, "conf": conf, "class_names": CLASS_NAMES, "checkpoint": CHECKPOINT_SOURCE}
