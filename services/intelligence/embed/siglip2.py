"""SigLIP 2 (so400m) image + text embeddings in a shared 1152-d space, L2-normalized. Text-aligned, so
it powers semantic dataset search (M1.4) and zero-shot scene classification (M1.3). Lazy load (FP16 on
GPU, FP32 on CPU for overnight backfill). The checkpoint is recorded on every vector for provenance.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("siglip2")
_lock = threading.Lock()
_state: dict = {}


def model_tag() -> str:
    return get_settings().intel.embed.siglip2_model


def _model():
    if "model" not in _state:
        with _lock:
            if "model" not in _state:
                import torch
                from transformers import AutoModel, AutoProcessor

                cfg = get_settings().intel.embed
                dev = cfg.device if torch.cuda.is_available() else "cpu"
                dt = torch.float16 if dev.startswith("cuda") else torch.float32
                proc = AutoProcessor.from_pretrained(cfg.siglip2_model)
                model = AutoModel.from_pretrained(cfg.siglip2_model, torch_dtype=dt).eval().to(dev)
                _state.update(model=model, proc=proc, device=dev, torch=torch)
                log.info("siglip2.loaded", model=cfg.siglip2_model, device=dev)
    return _state


def _to_pil(image_bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _norm(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


def encode_image(image_bgr: np.ndarray) -> np.ndarray:
    return encode_images([image_bgr])[0]


def encode_images(images_bgr: list[np.ndarray]) -> np.ndarray:
    s = _model()
    inp = s["proc"](images=[_to_pil(im) for im in images_bgr], return_tensors="pt").to(s["device"])
    with s["torch"].no_grad():
        v = s["model"].get_image_features(**inp).float().cpu().numpy()
    return _norm(v)


def encode_text(text: str) -> np.ndarray:
    return encode_texts([text])[0]


def encode_texts(texts: list[str]) -> np.ndarray:
    s = _model()
    inp = s["proc"](text=texts, return_tensors="pt", padding="max_length", max_length=64).to(s["device"])
    with s["torch"].no_grad():
        v = s["model"].get_text_features(**inp).float().cpu().numpy()
    return _norm(v)
