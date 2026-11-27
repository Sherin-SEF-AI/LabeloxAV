"""Zero-shot classification over a family shortlist, with negatives and a margin.

The fix for the bare 200-way softmax in `classify_crop`, in three parts:

- **Restricted.** Only the object's shortlist (families.py) competes, so "sign or hoarding?" is decided
  between sign and hoarding, not won by whichever of 200 names happened to score highest.
- **Negatives.** "Not an object" is represented by prompts for the things a false box usually sits on (road
  surface, wall, sky, trees, a shadow), and it wins when they outscore every class. Without them the
  classifier can only ever name a class.
- **Margin.** It reports the gap between its top answer and the runner-up, and the agent reads that gap, not
  the softmax probability, which is close to 1 for almost anything at a temperature of 100.

Prompts are ensembled over a few templates and the ontology's aliases (an autorickshaw is also a tuk-tuk),
which is cheap because text vectors are computed once per class and cached.

The image side is the SigLIP 2 vector already stored per object (same model, same letterbox preparation
as `classify_crop`), so at decision time this is a dot product, and only the text encoder ever runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.labelagent.truth import NOT_OBJECT

TEMPLATES = (
    "a photo of a {}.",
    "a dashcam photo of a {} on an Indian road.",
    "a cropped photo of a {}.",
)

NEGATIVE_PROMPTS = (
    "a photo of an empty road surface.",
    "a photo of a plain building wall.",
    "a photo of the sky.",
    "a photo of trees and leaves.",
    "a photo of a shadow on the road.",
    "a blurry photo of nothing in particular.",
    "a photo of a car bonnet or dashboard.",
)

SCALE = 100.0

_text_cache: dict[str, np.ndarray] = {}


def _encode(prompts: list[str]) -> np.ndarray:
    missing = [p for p in prompts if p not in _text_cache]
    if missing:
        from services.intelligence.embed import siglip2
        for k in range(0, len(missing), 64):
            vecs = np.asarray(siglip2.encode_texts(missing[k:k + 64]), dtype=np.float32)
            for p, v in zip(missing[k:k + 64], vecs, strict=True):
                _text_cache[p] = v / (np.linalg.norm(v) or 1.0)
    return np.stack([_text_cache[p] for p in prompts])


def class_vector(cd) -> np.ndarray:
    """One text vector per class: the mean over templates and names (the class name and its aliases)."""
    names = [cd.name.replace("_", " "), *[a.replace("_", " ") for a in getattr(cd, "aliases", ())]]
    v = _encode([t.format(n) for n in names for t in TEMPLATES]).mean(axis=0)
    return v / (np.linalg.norm(v) or 1.0)


def negative_vectors() -> np.ndarray:
    return _encode(list(NEGATIVE_PROMPTS))


@dataclass
class ZsAnswer:
    label: int | None
    margin: float
    scores: dict[int, float]


# Subtracted from the "not an object" logit. Left at 0 on purpose, and the reason is worth keeping: the best
# of seven generic negatives outscores a specific class name on most dashcam crops (road, wall and sky are in
# every box), so on its own this calls 11-18% of real objects "not an object", and raising the offset loses
# true catches faster than false alarms (2026-11-27 sweep, 0..10). Zero-shot cannot tell a non-object from
# an object here. That does not matter where it is used: it only vetoes a memory+probe agreement, and on the
# calibration half the veto at 0 kept the most fixes with no breaks. Do not read its "not an object" as a vote.
NEG_BIAS = 0.0


def classify(image_vec: np.ndarray, candidates: list[int], onto, *, neg_bias: float | None = None) -> ZsAnswer:
    """The shortlist's best answer for one stored SigLIP 2 image vector. NOT_OBJECT in `candidates` brings in
    the negative prompts; its score is the best of them, less `neg_bias`."""
    bias = NEG_BIAS if neg_bias is None else neg_bias
    q = np.asarray(image_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    labels, logits = [], []
    for c in candidates:
        if c == NOT_OBJECT:
            labels.append(NOT_OBJECT)
            logits.append(float((negative_vectors() @ q).max()) * SCALE - bias)
        else:
            try:
                labels.append(c)
                logits.append(float(class_vector(onto.by_id(c)) @ q) * SCALE)
            except KeyError:
                labels.pop()
    if not labels:
        return ZsAnswer(None, 0.0, {})
    z = np.array(logits)
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p)
    second = float(p[order[1]]) if len(order) > 1 else 0.0
    return ZsAnswer(label=labels[int(order[0])], margin=round(float(p[order[0]]) - second, 4),
                    scores={labels[n]: round(float(p[n]), 4) for n in range(len(labels))})


def classify_all_classes(image_vec: np.ndarray, onto) -> tuple[int, float]:
    """`classify_crop` as it stands, on a stored vector: one bare prompt per class, softmax over all of
    them, no negatives. Kept as the baseline the family version has to beat."""
    q = np.asarray(image_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    classes = list(onto.classes)
    t = _encode([f"a photo of a {c.name.replace('_', ' ')}" for c in classes])
    z = (t @ q) * SCALE
    p = np.exp(z - z.max())
    p /= p.sum()
    k = int(np.argmax(p))
    return classes[k].id, float(p[k])
