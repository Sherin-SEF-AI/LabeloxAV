"""DINOv3 ViT-B/16 visual embeddings via the non-gated timm mirror (vit_base_patch16_dinov3.lvd1689m),
L2-normalized 768-d. Self-supervised, so it powers duplicate detection (M1.1), visual similarity (M1.2),
clustering for rare-scenario discovery (M1.5), and novelty for intelligent extraction (M1.6). The
official facebook/dinov3 repos are HF-gated; the timm mirror carries the same weights without gating.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("dinov3")
_lock = threading.Lock()
_state: dict = {}


def model_tag() -> str:
    return get_settings().intel.embed.dinov3_model


def _model():
    if "model" not in _state:
        with _lock:
            if "model" not in _state:
                import timm
                import torch

                cfg = get_settings().intel.embed
                dev = cfg.device if torch.cuda.is_available() else "cpu"
                model = timm.create_model(cfg.dinov3_model, pretrained=True, num_classes=0).eval().to(dev)
                dcfg = timm.data.resolve_model_data_config(model)
                tf = timm.data.create_transform(**dcfg, is_training=False)
                _state.update(model=model, tf=tf, device=dev, torch=torch)
                log.info("dinov3.loaded", model=cfg.dinov3_model, device=dev, input=dcfg["input_size"])
    return _state


def _prep(image_bgr: np.ndarray):
    from PIL import Image

    s = _model()
    return s["tf"](Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))


def _norm(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


def encode_image(image_bgr: np.ndarray) -> np.ndarray:
    return encode_images([image_bgr])[0]


def encode_images(images_bgr: list[np.ndarray]) -> np.ndarray:
    s = _model()
    batch = s["torch"].stack([_prep(im) for im in images_bgr]).to(s["device"])
    with s["torch"].no_grad():
        v = s["model"](batch).float().cpu().numpy()
    return _norm(v)
