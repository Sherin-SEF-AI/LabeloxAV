"""Path D: grounded open-vocabulary detection with Grounding DINO or OWLv2, a second opinion beside Path B.

Path B is YOLO-World v8s over 34 grounded concepts. It is fast, and it is also the weakest open-vocabulary
detector in common use: most of the machine objects it produces sit in review at confidence 0.4-0.5, and the
long-tail Indian classes (autorickshaw, construction barrier, hoarding) are where it confuses most. Two stronger
open-vocabulary detectors, both Apache-2.0 and both run locally from the Hugging Face cache:

- **Grounding DINO** (`IDEA-Research/grounding-dino-base`): a text-conditioned DETR. Its text encoder takes at
  most 256 tokens, so the concept list is prompted in chunks, one forward pass per chunk.
- **OWLv2** (`google/owlv2-base-patch16-ensemble`): CLIP-style per-box text matching. All concepts go in one
  pass as separate queries. It pads the image to a square, so boxes come back in that padded frame.

Same contract as Path B: concept phrases in (the grounded set plus the pack's synonyms), `RawDetection` out,
each phrase mapping back to exactly one ontology class. Boxes only; Path B's SAM step already makes masks for
the boxes fusion keeps, and running SAM twice would double the most expensive stage for nothing.

Whether either one serves is decided by scripts/detector_benchmark.py on gold labels, not here.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.base import RawDetection

log = get_logger("path_d")

BACKENDS = {
    "grounding_dino": "IDEA-Research/grounding-dino-base",
    "owlv2": "google/owlv2-base-patch16-ensemble",
}

# Grounding DINO's text encoder limit is 256 tokens including the separators; chunks are built to stay well
# under it, since a truncated prompt silently drops whichever concepts fell off the end.
_GDINO_MAX_TOKENS = 200


class GroundedPath:
    name = "path_d_grounded"

    def __init__(self, backend: str = "grounding_dino", supported_ids: set[int] | None = None, *,
                 box_threshold: float = 0.30, text_threshold: float = 0.25, max_boxes: int = 100,
                 nms_iou: float = 0.5) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown Path D backend {backend!r}; expected one of {sorted(BACKENDS)}")
        self.backend = backend
        self.model_id = BACKENDS[backend]
        self.settings = get_settings()
        self.onto = get_ontology()
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.max_boxes, self.nms_iou = max_boxes, nms_iou
        classes = self.onto.classes if supported_ids is None else [c for c in self.onto.classes if c.id in supported_ids]
        from services.domain import active_pack

        synonyms = active_pack().autolabel_profile.openvocab_synonyms
        self._phrases: list[str] = []
        self._phrase_classes: list = []
        for c in sorted(classes, key=lambda c: (not c.india, c.id)):
            for phrase in synonyms.get(c.name, [c.name.replace("_", " ")]):
                if phrase not in self._phrases:
                    self._phrases.append(phrase)
                    self._phrase_classes.append(c)
        self.model_version = f"{self.model_id}:{backend}"
        self._model = None
        self._proc = None
        self._device = "cpu"

    @property
    def key(self) -> str:
        return f"pathd:{self.backend}"

    def load(self) -> None:
        from core.model_manager import manager

        dev = self.settings.gpu.device

        def _load():
            import torch
            from transformers import AutoProcessor

            if self.backend == "grounding_dino":
                from transformers import AutoModelForZeroShotObjectDetection as M
            else:
                from transformers import Owlv2ForObjectDetection as M
            proc = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            model = M.from_pretrained(self.model_id, local_files_only=True).eval()
            if torch.cuda.is_available() and dev.startswith("cuda"):
                model = model.to(dev)
            return {"model": model, "proc": proc}

        s = manager().get(self.key, _load, device=dev)
        self._model, self._proc = s["model"], s["proc"]
        self._device = next(self._model.parameters()).device
        log.info("path_d.loaded", backend=self.backend, concepts=len(self._phrases), device=str(self._device))

    def unload(self) -> None:
        from core.model_manager import manager

        self._model = self._proc = None
        manager().release(self.key)

    # ------------------------------------------------------------------------------------------- inference

    def infer(self, image_bgr: np.ndarray) -> list[RawDetection]:
        return self.infer_batch([image_bgr])[0]

    def infer_batch(self, images: list[np.ndarray]) -> list[list[RawDetection]]:
        if self._model is None:
            raise RuntimeError("GroundedPath not loaded")
        from core.model_manager import manager

        out = []
        with manager().infer_lock(self.key):
            for img in images:
                raw = self._gdino(img) if self.backend == "grounding_dino" else self._owl(img)
                out.append(self._finish(raw))
        return out

    def _chunks(self) -> list[list[int]]:
        """Phrase indices grouped so each Grounding DINO prompt stays under the token limit."""
        tok = self._proc.tokenizer
        chunks, cur, used = [], [], 0
        for i, p in enumerate(self._phrases):
            n = len(tok(p + " . ", add_special_tokens=False)["input_ids"])
            if cur and used + n > _GDINO_MAX_TOKENS:
                chunks.append(cur)
                cur, used = [], 0
            cur.append(i)
            used += n
        if cur:
            chunks.append(cur)
        return chunks

    def _gdino(self, image_bgr: np.ndarray) -> list[tuple[tuple, float, int]]:
        import torch
        from PIL import Image

        h, w = image_bgr.shape[:2]
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        found = []
        for idx in self._chunks():
            text = " . ".join(self._phrases[i] for i in idx) + " ."
            inp = self._proc(images=pil, text=text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inp)
            res = self._proc.post_process_grounded_object_detection(
                outputs, inp.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(h, w)])[0]
            labels = res.get("text_labels", res.get("labels"))
            for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels, strict=False):
                pi = self._match(str(label), idx)
                if pi is not None:
                    found.append((tuple(box), float(score), pi))
        return found

    def _match(self, label: str, idx: list[int]) -> int | None:
        """The phrase a Grounding DINO label came from. It returns the span of prompt tokens that fired,
        which is usually the phrase, sometimes part of one, and occasionally two run together; the longest
        phrase contained in the label wins, then a phrase containing the label."""
        lab = label.strip().lower()
        if not lab:
            return None
        inside = [i for i in idx if self._phrases[i].lower() in lab]
        if inside:
            return max(inside, key=lambda i: len(self._phrases[i]))
        around = [i for i in idx if lab in self._phrases[i].lower()]
        return min(around, key=lambda i: len(self._phrases[i])) if around else None

    def _owl(self, image_bgr: np.ndarray) -> list[tuple[tuple, float, int]]:
        import torch
        from PIL import Image

        h, w = image_bgr.shape[:2]
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        inp = self._proc(text=[self._phrases], images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inp)
        side = max(h, w)   # OWLv2 pads to a square at the top-left, so the padded frame is the side x side box
        res = self._proc.post_process_object_detection(outputs, threshold=self.box_threshold,
                                                       target_sizes=[(side, side)])[0]
        found = []
        for box, score, li in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist(), strict=False):
            x1, y1, x2, y2 = box
            box = (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                found.append((box, float(score), int(li)))
        return found

    def _finish(self, raw: list[tuple[tuple, float, int]]) -> list[RawDetection]:
        """Per-class NMS (chunks and synonyms can find one object twice), then the per-frame budget."""
        if not raw:
            return []
        import torch
        from torchvision.ops import batched_nms

        boxes = torch.tensor([r[0] for r in raw], dtype=torch.float32)
        scores = torch.tensor([r[1] for r in raw], dtype=torch.float32)
        cls = torch.tensor([self._phrase_classes[r[2]].id for r in raw], dtype=torch.int64)
        keep = batched_nms(boxes, scores, cls, self.nms_iou)[: self.max_boxes].tolist()
        return [RawDetection(path=self.name, bbox=tuple(float(v) for v in raw[k][0]), conf=raw[k][1],
                             class_name=self._phrase_classes[raw[k][2]].name,
                             class_id=self._phrase_classes[raw[k][2]].id, model_version=self.model_version,
                             extra={"phrase": self._phrases[raw[k][2]], "backend": self.backend})
                for k in keep]
