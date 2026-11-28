"""Path B: open-vocabulary detect + segment of the long tail, driven by ontology concept phrases.

Spec target is SAM 3.1 Promptable Concept Segmentation (one model, text -> detect+segment+track).
That model is not yet in the installed Ultralytics, so Path B is realized today as the spec's own
D1 fallback shape: YOLO-World (open-vocab text -> boxes over the ontology phrases) followed by SAM
(box -> mask). Same Path B contract: a concept phrase yields boxes plus masks, no retraining.

To swap in real SAM 3.1 PCS later, replace this class's load()/infer() with the PCS predictor and
point models.openvocab at sam3.pt; the fusion contract (RawDetection with class + box + mask) is
unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.base import RawDetection, mask_to_bbox

log = get_logger("path_b")

# Open-vocab synonym expansion: some ontology class names are poor text prompts for YOLO-World (it was
# trained on common nouns). "cattle" barely activates; "cow"/"buffalo"/"bull"/"ox" do. The synonym map now
# lives in the active domain pack (pack.autolabel_profile.openvocab_synonyms), keyed by ontology class name;
# each phrase maps back to its single ontology class.


def sam_model_key(weights: str) -> str:
    """The registry key for a SAM checkpoint, so every caller of the same file shares one instance."""
    return f"sam:{weights}"


def load_sam(weights: str):
    from ultralytics import SAM

    return SAM(weights)


class OpenVocabSamPath:
    # Frozen: this string is the path identity in 834k objects' provenance and in the
    # fusion/gate/explain dispatch. It predates a model substitution, so it no longer names
    # the model this file runs - the filename now does. Renaming it means migrating history.
    name = "path_b_sam3"

    def __init__(self, supported_ids: set[int] | None = None) -> None:
        self.settings = get_settings()
        self.onto = get_ontology()
        self._world = None
        self._sam = None
        # M-Q.0: prompt only with the grounded supported set. An ungrounded class name in the open-vocab
        # concept list is the hallucination source (the model finds water_bottles because it is asked to),
        # so anything outside the set is dropped here and folds to fallback downstream. None means all
        # classes (back-compat for callers that do not pass the grounded set).
        classes = self.onto.classes if supported_ids is None else [c for c in self.onto.classes if c.id in supported_ids]
        # India/rare classes first so the open-vocab budget favors where Path A is weakest.
        self._classes = sorted(classes, key=lambda c: (not c.india, c.id))
        # Expand each class into its prompt phrases (usually just its name; synonyms for weak-prompt classes
        # like cattle). _phrase_classes[i] is the ontology class the i-th phrase resolves back to.
        self._phrases: list[str] = []
        self._phrase_classes: list = []
        from services.domain import active_pack
        synonyms = active_pack().autolabel_profile.openvocab_synonyms
        for c in self._classes:
            for phrase in synonyms.get(c.name, [c.name.replace("_", " ")]):
                self._phrases.append(phrase)
                self._phrase_classes.append(c)
        self.model_version = (
            f"{self.settings.models.openvocab.detector_weights}+{self.settings.models.openvocab.seg_weights}"
        )

    @property
    def world_key(self) -> str:
        # The concept list is part of the key: set_classes() rewrites the detector's text head, so an
        # instance prompted with this ontology's phrases is not interchangeable with one prompted with
        # an annotator's free-text phrase (services/autolabel/describe.py keeps its own).
        import hashlib

        digest = hashlib.sha1("\n".join(self._phrases).encode(), usedforsecurity=False).hexdigest()[:12]
        return f"yoloworld:{self.settings.models.openvocab.detector_weights}:{digest}"

    @property
    def sam_key(self) -> str:
        return sam_model_key(self.settings.models.openvocab.seg_weights)

    def load(self) -> None:
        from core.model_manager import manager

        cfg = self.settings.models.openvocab
        dev = self.settings.gpu.device

        def _world():
            from ultralytics import YOLOWorld

            w = YOLOWorld(cfg.detector_weights)
            w.set_classes(self._phrases)
            return w

        self._world = manager().get(self.world_key, _world, device=dev)
        # The segmenter is one file used by this path, the editor's click-to-segment and the describe
        # tool. It was loaded once by each; they now share the instance.
        self._sam = manager().get(self.sam_key, lambda: load_sam(cfg.seg_weights), device=dev)
        log.info(
            "path_b.loaded",
            detector=cfg.detector_weights,
            segmenter=cfg.seg_weights,
            concepts=len(self._phrases),
        )

    def unload(self) -> None:
        from core.model_manager import manager

        self._world = None
        self._sam = None
        manager().release(self.world_key)
        # Refused, harmlessly, while the editor is mid-click on the shared segmenter.
        manager().release(self.sam_key)

    def infer(self, image_bgr: np.ndarray) -> list[RawDetection]:
        return self.infer_batch([image_bgr])[0]

    def infer_batch(self, images: list[np.ndarray]) -> list[list[RawDetection]]:
        """Open-vocabulary detections for several frames, one list per frame, in order.

        The detector runs once over the batch. Segmentation stays per frame: SAM is prompted with each
        frame's own boxes against that frame's own embedding, which is not a batched operation.
        """
        if self._world is None or self._sam is None:
            raise RuntimeError("OpenVocabSamPath not loaded")
        if not images:
            return []
        from core.model_manager import manager

        cfg = self.settings.models.openvocab
        dev = self.settings.gpu.device
        with manager().infer_lock(self.world_key):
            wres = self._world.predict(
                source=images if len(images) > 1 else images[0],
                imgsz=self.settings.models.yolo.imgsz, half=cfg.half,
                conf=cfg.conf, device=dev, verbose=False,
            )
        if len(wres) != len(images):
            raise RuntimeError(f"open-vocab detector returned {len(wres)} results for {len(images)} frames")
        return [self._frame_detections(img, r, cfg, dev) for img, r in zip(images, wres, strict=True)]

    def _frame_detections(self, image_bgr: np.ndarray, r0, cfg, dev: str) -> list[RawDetection]:
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clsidx = r0.boxes.cls.cpu().numpy().astype(int)

        # Keep the most confident boxes within the per-frame budget.
        order = np.argsort(-confs)[: cfg.max_boxes]
        xyxy, confs, clsidx = xyxy[order], confs[order], clsidx[order]

        masks = self._segment(image_bgr, xyxy, dev)

        dets: list[RawDetection] = []
        for i in range(len(xyxy)):
            ci = clsidx[i]
            spec = self._phrase_classes[ci] if 0 <= ci < len(self._phrase_classes) else self._phrase_classes[-1]
            mask = masks[i] if masks is not None and i < len(masks) else None
            bbox = mask_to_bbox(mask) if mask is not None else tuple(float(v) for v in xyxy[i])
            if bbox is None:
                bbox = tuple(float(v) for v in xyxy[i])
            dets.append(
                RawDetection(
                    path=self.name,
                    bbox=bbox,
                    conf=float(confs[i]),
                    class_name=spec.name,
                    class_id=spec.id,
                    model_version=self.model_version,
                    mask=mask,
                )
            )
        return dets

    def _segment(self, image_bgr: np.ndarray, xyxy: np.ndarray, dev: str) -> np.ndarray | None:
        if len(xyxy) == 0:
            return None
        from core.model_manager import keep_gpu_visible, manager

        with manager().infer_lock(self.sam_key), keep_gpu_visible():
            sres = self._sam.predict(source=image_bgr, bboxes=xyxy, device=dev, verbose=False)
        m = sres[0].masks
        if m is None or m.data is None:
            return None
        return m.data.cpu().numpy().astype(bool)


def polygons_from_mask(mask: np.ndarray, epsilon_frac: float = 0.005, keep_holes: bool = False,
                       epsilon_px: float | None = None) -> list[list[float]]:
    """Compact polygon encoding for persistence, flattened [x,y,...] per ring.

    Default (keep_holes=False) keeps only external contours, which is right for a solid object. With
    keep_holes=True the interior contours are returned too, so a mask with a hole (an erased region, or a
    stuff region cut by an occluding vehicle) round-trips: consumers interpret the ring list with the
    even-odd rule, so a ring nested inside another reads as a hole rather than a second filled blob.

    Simplification tolerance defaults to epsilon_frac * perimeter, which is fine for compact objects but
    coarsens large regions (a huge retaining wall gets a big absolute tolerance, so its boundary drifts
    off the visible edge). Pass epsilon_px to use a fixed pixel tolerance instead, size-independent, so a
    big stuff region follows its edge as tightly as a small one.
    """
    m = (mask.astype(np.uint8)) * 255
    mode = cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        eps = epsilon_px if epsilon_px is not None else epsilon_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        polys.append([float(v) for pt in approx.reshape(-1, 2) for v in pt])
    return polys
