"""Zero-shot object classification for a user-drawn region. When an annotator draws a SAM box or clicks the
wand, we crop that region, score its SigLIP 2 image vector against every ontology class name, and return the
top matches, so the tool auto-detects what the object is instead of the annotator picking the class by hand.
Reuses the same SigLIP 2 zero-shot path as scene (services/intelligence/scene.py) and sign classification.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

log = get_logger("classify_crop")
_cache: dict = {}


def _ontology_text_vecs():
    """Class list + their SigLIP 2 text vectors, computed once (170 classes, cheap matmul thereafter)."""
    if not _cache:
        from services.autolabel.ontology import get_ontology
        from services.intelligence.embed import siglip2
        classes = list(get_ontology().classes)
        prompts = [f"a photo of a {c.name.replace('_', ' ')}" for c in classes]
        _cache["classes"] = classes
        _cache["vecs"] = siglip2.encode_texts(prompts)
    return _cache["classes"], _cache["vecs"]


def classify_crop(crop_bgr: np.ndarray, topk: int = 3) -> list[dict]:
    """Top-k ontology classes for a crop by SigLIP 2 zero-shot similarity, each with a softmax confidence."""
    from services.intelligence.embed import siglip2
    classes, tvecs = _ontology_text_vecs()
    fv = siglip2.encode_image(crop_bgr)
    logits = (np.asarray(tvecs) @ np.asarray(fv)) * 100.0
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    order = np.argsort(p)[::-1][:topk]
    return [{"class_id": classes[i].id, "class_name": classes[i].name, "conf": round(float(p[i]), 4)} for i in order]
