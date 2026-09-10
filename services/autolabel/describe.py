"""Label an object by describing it, when the ontology has no word for it yet.

An annotator looking at an Indian road sees things the class list does not carry: a cycle rickshaw, a
hand cart, a water tanker, a pile of construction sand in a lane. Today the only options are to force it
into the nearest class, which corrupts that class, or to skip it, which loses the object. Both are worse
than saying what it is.

This runs the same open-vocabulary detector and segmenter the autolabel plane already loads
(`paths/path_b_openvocab.py`), prompted with one phrase instead of the grounded ontology, over one frame
instead of a sweep. The proposals come back as polygons the editor can commit exactly like a SAM result.

**The phrase is evidence, not a class.** A described object is written with `source='described'` and the
phrase on its provenance. It does not invent an ontology entry, because an ontology grown by whatever
somebody typed is how a class list stops meaning anything; what it does is make the object exist and
findable, so a later ontology decision has real examples to look at rather than an argument.

**Grounding is the caller's problem and it is stated, not hidden.** The open-vocabulary detector will
find whatever it is asked for, which is exactly the hallucination that made the ungrounded concept list
dangerous in the sweep (M-Q.0). One frame, prompted by a person looking at it, with the result shown
before anything is written, is a different situation from a corpus sweep: the person is the grounding.
The confidence is returned unmodified so they can see how sure the model was.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("describe")

# The most proposals one phrase may return. A phrase like "vehicle" matches most of a busy frame, and a
# list of eighty boxes is not a proposal, it is a mess to clean up.
MAX_PROPOSALS = 12
# Below this the detector is guessing. Lower than the sweep's floor because a person is about to look at
# every one of these, and a visible weak proposal they reject costs less than a missed object.
MIN_CONF = 0.05
MAX_PHRASE_CHARS = 80

_STATE: dict = {}


@dataclass(frozen=True)
class Proposal:
    bbox: list[float]
    conf: float
    polygons: list[list[float]]


def normalize_phrase(phrase: str) -> str:
    """The prompt actually sent, or an empty string when there is nothing usable in it.

    Collapsed and lowercased because the detector's text encoder is not helped by an annotator's capitals
    or double spaces, and bounded because a paragraph is not a prompt.
    """
    cleaned = " ".join(str(phrase or "").split()).strip().lower()
    return cleaned[:MAX_PHRASE_CHARS]


def _models():
    """The open-vocabulary detector and segmenter, loaded once and kept.

    Shared module state rather than a fresh pair per request, because each call is a person waiting on a
    frame and reloading two checkpoints per keystroke would make the tool unusable.
    """
    if "world" not in _STATE:
        from ultralytics import SAM, YOLOWorld

        cfg = get_settings().models.openvocab
        _STATE["world"] = YOLOWorld(cfg.detector_weights)
        _STATE["sam"] = SAM(cfg.seg_weights)
        log.info("describe.loaded", detector=cfg.detector_weights, segmenter=cfg.seg_weights)
    return _STATE["world"], _STATE["sam"]


def unload() -> None:
    _STATE.clear()


def propose(image_bgr: np.ndarray, phrase: str, *, max_proposals: int = MAX_PROPOSALS,
            min_conf: float = MIN_CONF) -> list[Proposal]:
    """Everything in this frame that matches the phrase, most confident first. Blocking GPU work."""
    from services.autolabel.paths.path_b_openvocab import polygons_from_mask

    text = normalize_phrase(phrase)
    if not text:
        return []
    settings = get_settings()
    world, sam = _models()
    dev = settings.gpu.device

    world.set_classes([text])
    res = world.predict(source=image_bgr, imgsz=settings.models.yolo.imgsz,
                        conf=min_conf, device=dev, verbose=False)
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)[:max_proposals]
    xyxy, confs = xyxy[order], confs[order]

    masks = None
    try:
        sres = sam.predict(source=image_bgr, bboxes=xyxy, device=dev, verbose=False)
        m = sres[0].masks
        if m is not None and m.data is not None:
            masks = m.data.cpu().numpy().astype(bool)
    except Exception as exc:  # noqa: BLE001 - a failed mask is a box proposal, not a failed request
        log.warning("describe.segment_failed", phrase=text, error=str(exc)[:200])

    out: list[Proposal] = []
    for i in range(len(xyxy)):
        polys: list[list[float]] = []
        if masks is not None and i < len(masks):
            polys = polygons_from_mask(masks[i])
        out.append(Proposal(bbox=[float(v) for v in xyxy[i]], conf=float(confs[i]), polygons=polys))
    log.info("describe.proposed", phrase=text, proposals=len(out))
    return out
