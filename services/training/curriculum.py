"""Ordering an epoch by how much a model still has to learn from each image.

Shuffling treats every training image as equally informative, which they are not: a frame the detector
already agrees with a person about teaches almost nothing, and a frame full of classes the gate says are
short teaches the thing being trained for. Curriculum ordering spends the same epoch differently.

Two weights, and both are measured rather than assumed.

**Class deficit.** An image containing classes the champion gate is short on is worth more than one that
is not, in proportion to how short. That is the same signal gate-directed labelling and the synthetic
generator already act on, so training, labelling and generation are all pulled by one number instead of
three notions of importance.

**Agreement.** An image where the machine's proposals and the human labels agree is one the model already
has. Easy first then the full set, which is the ordering the curriculum-learning literature settles on and
the opposite of what feels intuitive: starting on the hardest examples trains on the noisiest gradients.

Nothing here changes which images are in the epoch, only the order and how often. That matters because a
curriculum that drops images silently changes the dataset a metric was computed on, and two runs would
then be incomparable with nothing saying so.
"""

from __future__ import annotations

import random

# How many times the most valuable image may appear in one epoch. Above this an epoch is mostly one
# frame and the model overfits the thing it was meant to learn.
MAX_REPEAT = 3
# Below this an image is ordinary and appears once, whatever its weight.
REPEAT_THRESHOLD = 0.66


def class_weights(deficits: dict[str, float], *, floor: float = 1.0) -> dict[str, float]:
    """Per-class sample weight from the gate's per-class deficit, normalised so the largest is 2.0.

    Normalised rather than absolute: a deficit is a recall gap between 0 and 1, and using it raw would
    make every weight nearly `floor` and the curriculum a no-op that still changed the run's provenance.
    """
    if not deficits:
        return {}
    worst = max(abs(float(v)) for v in deficits.values()) or 1.0
    return {k: floor + abs(float(v)) / worst for k, v in deficits.items()}


def image_weight(class_names: list[str], weights: dict[str, float], *, agreement: float | None = None,
                 floor: float = 1.0) -> float:
    """One image's weight: its heaviest class, discounted by how well the machine already agrees.

    The heaviest class rather than the sum, because an image with twenty cars is not twenty times more
    valuable than one with a single rider of a starved class, and summing makes crowded frames win
    regardless of what is in them.
    """
    if not class_names:
        return floor
    w = max((weights.get(c, floor) for c in class_names), default=floor)
    if agreement is None:
        return w
    # Full agreement means the model already has this image; the weight falls back toward the floor.
    return floor + (w - floor) * (1.0 - max(0.0, min(1.0, agreement)))


def order_epoch(images: list[str], weights: dict[str, float], *, seed: int = 7,
                max_repeat: int = MAX_REPEAT) -> list[str]:
    """The epoch's image list: easy first, then the whole set, with valuable images repeated.

    Every image appears at least once. A curriculum that dropped images would silently change the dataset
    a metric was computed on, and two runs would be incomparable with nothing saying so.
    """
    if not images:
        return []
    rng = random.Random(seed)
    ranked = sorted(images, key=lambda i: (weights.get(i, 1.0), i))
    out = list(ranked)                       # easy first: the full set in ascending weight
    extra: list[str] = []
    if weights:
        top = max(weights.values())
        for img in ranked:
            w = weights.get(img, 1.0)
            if top > 0 and w / top >= REPEAT_THRESHOLD:
                repeats = min(max_repeat, max(1, int(round(w / top * max_repeat)))) - 1
                extra.extend([img] * repeats)
    rng.shuffle(extra)
    return out + extra
