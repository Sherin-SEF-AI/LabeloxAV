"""Model backends for recall recovery: the only file that binds to a model runtime. Every heavy import
(cv2, numpy, ultralytics, torch, the autolabel paths) happens inside a method, so the orchestrator and
its tests load with no GPU stack. The shared YOLO-World, SAM, and VLM are reused from the autolabel
paths, never a second copy.

The only seams that bind to a model runtime are marked # WIRE: and appear nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.logging import get_logger

log = get_logger("recall_backends")


def load_image_bgr(store, img_uri: str):
    """Fetch bytes from the object store and decode to BGR. Keeps cv2 out of recover.py."""
    import cv2
    import numpy as np

    data = store.get_bytes(img_uri)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image at {img_uri}")
    return img


class OpenVocabAdapter:
    """Path B (YOLO-World + SAM) open-vocab detector, reused. Returns named-class boxes the primary
    detector may have missed."""

    def __init__(self, onto, settings) -> None:
        self.onto = onto
        self.settings = settings
        self._path = None

    def _ensure(self):
        if self._path is None:
            from services.autolabel.paths.path_b_sam3 import Sam3Path

            self._path = Sam3Path()
            self._path.load()
            # WIRE: to widen the open-vocab phrase set beyond Path B's ontology phrases (e.g. extra
            # long-tail concepts for a recall sweep), call self._path._world.set_classes([...]) here
            # before inference. Left at Path B's ontology phrases by default.
        return self._path

    def detect(self, image_bgr) -> list[tuple]:
        path = self._ensure()
        return [(tuple(float(v) for v in det.bbox), det.class_name, float(det.conf))
                for det in path.infer(image_bgr)]


class RegionAdapter:
    """Class-agnostic region proposals via SAM everything mode. Noisy on dense Indian frames, so the
    area band drops specks and whole-scene planes before anything reaches the VLM."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self._sam = None

    def _ensure(self):
        if self._sam is None:
            from ultralytics import SAM

            self._sam = SAM(self.settings.phase4.recall.sam_everything_weights)
        return self._sam

    def propose(self, image_bgr) -> list[tuple]:
        return [bb for _, bb in self.propose_masks(image_bgr)]

    def propose_masks(self, image_bgr) -> list[tuple]:
        """Area-filtered SAM-everything regions as (mask_bool, bbox) pairs. Dense semantic/panoptic needs
        the masks (not just boxes), so this is the variant the segment2d service consumes."""
        from services.autolabel.paths.base import mask_to_bbox

        cfg = self.settings.phase4.recall
        sam = self._ensure()
        h, w = image_bgr.shape[:2]
        frame_area = float(h * w)
        # WIRE: SAM everything mode (no prompt) proposes every region in the frame.
        res = sam.predict(source=image_bgr, verbose=False, device=cfg.device)
        masks = res[0].masks if res else None
        if masks is None or masks.data is None:
            return []
        out: list[tuple] = []
        for m in masks.data.cpu().numpy().astype(bool):
            bb = mask_to_bbox(m)
            if bb is None:
                continue
            area = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
            frac = area / frame_area if frame_area > 0 else 0.0
            if frac < cfg.region_min_area_frac or frac > cfg.region_max_area_frac:
                continue
            out.append((m, tuple(float(v) for v in bb)))
        return out


class VlmClassifyAdapter:
    """The duty-cycled VLM, reused, as a region classifier. Returns (class_name, conf), or (None, conf)
    when the VLM reads the region as background or unknown, so a region crop never becomes a class-less
    object."""

    def __init__(self, onto, settings) -> None:
        self.onto = onto
        self.settings = settings
        self._verifier = None

    def _ensure(self):
        if self._verifier is None:
            from services.autolabel.paths.path_c_qwen3vl import VlmVerifier, make_vlm_client

            self._verifier = VlmVerifier(make_vlm_client(self.settings), self.onto, self.settings)
        return self._verifier

    def _shortlist(self) -> list[str]:
        return [c.name for c in self.onto.classes] + ["background"]

    def classify(self, image_bgr, bbox) -> tuple:
        from services.autolabel.paths.path_c_qwen3vl import crop_object

        cfg = self.settings.phase4.recall
        verifier = self._ensure()
        crop = crop_object(image_bgr, bbox, cfg.vlm_crop_margin)
        # WIRE: open-set VLM verify over the full leaf shortlist, so a class-agnostic region can be read
        # as any ontology class (or background).
        res = verifier.client.verify(crop, self._shortlist(), {})
        conf = res.agreement if (res.votes and res.votes > 1) else (0.9 if res.confident else 0.35)
        name = res.class_name
        if name in (None, "background", "unknown") or not self.onto.has_name(name):
            return None, float(conf)
        return name, float(conf)


@dataclass
class RecallBackends:
    openvocab_adapter: OpenVocabAdapter
    region_adapter: RegionAdapter
    vlm_adapter: VlmClassifyAdapter

    def load_image(self, store, img_uri: str):
        return load_image_bgr(store, img_uri)

    def openvocab(self, image_bgr) -> list[tuple]:
        return self.openvocab_adapter.detect(image_bgr)

    def regions(self, image_bgr) -> list[tuple]:
        return self.region_adapter.propose(image_bgr)

    def classify(self, image_bgr, bbox) -> tuple:
        return self.vlm_adapter.classify(image_bgr, bbox)


def build_backends(settings=None) -> RecallBackends:
    from core.config import get_settings
    from services.autolabel.ontology import get_ontology

    settings = settings or get_settings()
    onto = get_ontology()
    return RecallBackends(
        openvocab_adapter=OpenVocabAdapter(onto, settings),
        region_adapter=RegionAdapter(settings),
        vlm_adapter=VlmClassifyAdapter(onto, settings),
    )
