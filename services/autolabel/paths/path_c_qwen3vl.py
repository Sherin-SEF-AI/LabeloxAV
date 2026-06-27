"""Path C: the duty-cycled VLM verifier and attribute reader (Principle 08).

Invoked only on the uncertain subset (paths disagree, confidence in the review band, rare/fallback
class, or geometry conflict), never on the full stream. It crops the object with context margin,
asks a tight structured prompt, and parses a strict JSON reply: confirmed class, typed attributes,
short caption.

Spec target is Qwen3-VL-4B at 4-bit. On this box bitsandbytes 4-bit is unusable (no Blackwell
binary) and transformers lacks Qwen3-VL, so the working backend is Ollama serving a Qwen-VL model
(qwen2.5vl), out-of-process. The VlmClient interface is backend-agnostic so the model swaps by
config alone.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import httpx
import numpy as np

from core.config import Settings, get_settings
from core.logging import get_logger
from services.autolabel.ontology import Ontology, get_ontology

log = get_logger("path_c")


@dataclass
class VlmResult:
    class_name: str | None = None
    attrs: dict = field(default_factory=dict)
    caption: str = ""
    confident: bool = False
    votes: int = 1
    agreement: float = 1.0     # fraction of votes that chose class_name
    raw: dict = field(default_factory=dict)


class VlmClient(Protocol):
    def verify(
        self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict, temperature: float = 0.0
    ) -> VlmResult: ...


def crop_object(image_bgr: np.ndarray, bbox: tuple[float, float, float, float], margin: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    cx1 = int(max(0, x1 - mx))
    cy1 = int(max(0, y1 - my))
    cx2 = int(min(w, x2 + mx))
    cy2 = int(min(h, y2 + my))
    if cx2 <= cx1 or cy2 <= cy1:
        return image_bgr
    return image_bgr[cy1:cy2, cx1:cx2]


def _build_prompt(shortlist: list[str], attr_schema: dict) -> str:
    return (
        "You are labeling an object cropped from an Indian road scene for an autonomous-driving "
        "dataset. Identify the object and read its attributes.\n"
        f"Choose exactly one class from this list: {shortlist}.\n"
        f"Attribute schema (return only those that apply): {json.dumps(attr_schema)}.\n"
        'Respond with strict JSON only, no prose: '
        '{"class": "<one of the list>", "attributes": {<name>: <value>}, '
        '"caption": "<short description>", "confident": <true|false>}.'
    )


class OllamaVlmClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.models.vlm

    def verify(
        self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict, temperature: float = 0.0
    ) -> VlmResult:
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return VlmResult()
        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": self.cfg.ollama_tag,
            "messages": [{"role": "user", "content": _build_prompt(shortlist, attr_schema), "images": [b64]}],
            "stream": False,
            "format": "json",
            "options": {"num_ctx": self.cfg.max_context, "temperature": temperature},
        }
        try:
            resp = httpx.post(
                f"{self.cfg.ollama_url}/api/chat", json=payload, timeout=self.cfg.timeout_s
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            data = json.loads(content)
        except Exception as exc:  # noqa: BLE001
            log.warning("vlm.call_failed", error=str(exc))
            return VlmResult()

        return VlmResult(
            class_name=data.get("class"),
            attrs=data.get("attributes", {}) or {},
            caption=data.get("caption", "") or "",
            confident=bool(data.get("confident", False)),
            raw=data,
        )


def make_vlm_client(settings: Settings | None = None) -> VlmClient:
    settings = settings or get_settings()
    backend = settings.models.vlm.backend
    if backend == "ollama":
        return OllamaVlmClient(settings)
    raise NotImplementedError(
        f"vlm backend '{backend}' not wired on this box; use 'ollama' "
        "(transformers 4-bit needs a working bitsandbytes Blackwell binary)."
    )


def apply_vlm(obj, res: VlmResult, onto: Ontology, vlm_tag: str):
    """Merge a VLM verdict into a UnifiedObject: attributes, possible reclassification, confidence
    adjustment, and a path_c provenance proposal. Returns the mutated object."""
    from core.schemas import PathProposal

    if res.attrs:
        obj.attrs.update(res.attrs)

    verdict = "unsure"
    if res.class_name:
        if res.class_name == obj.class_name:
            verdict = "confirm"
            obj.conf = min(1.0, obj.conf + 0.12)
            obj.provenance.agreement = True
        else:
            # A within-superclass refinement (sedan -> suv) is cheap to accept. A cross-superclass
            # jump (dog -> autorickshaw) is a big claim. With multi-vote enabled, require N-vote
            # agreement (robust to an over-confident model); with a single vote, fall back to the
            # model's own confidence flag. Otherwise keep the detector class and record as unsure.
            from core.config import get_settings

            cross_superclass = onto.by_name(obj.class_name).l1 != onto.by_name(res.class_name).l1
            cross_min = get_settings().models.vlm.cross_vote_min
            accept = (not cross_superclass) or (
                res.agreement >= cross_min if res.votes > 1 else res.confident
            )
            if accept:
                verdict = "overruled"
                c = onto.by_name(res.class_name)
                obj.class_id, obj.class_name = c.id, c.name
                obj.conf = max(obj.conf, 0.75)
            else:
                verdict = "unsure"

    obj.provenance.proposals.append(
        PathProposal(path="path_c_qwen3vl", class_name=res.class_name, conf=None, verdict=verdict, model_version=vlm_tag)
    )
    if res.caption:
        obj.provenance.notes.append(f"caption: {res.caption}")
    return obj


class VlmVerifier:
    """Applies a VlmClient to a fused object: builds the shortlist, validates the reply against the
    ontology, and reports the class/attrs to merge back. Pure of DB and GPU so it is unit-testable.
    """

    def __init__(self, client: VlmClient, ontology: Ontology | None = None, settings: Settings | None = None) -> None:
        self.client = client
        self.onto = ontology or get_ontology()
        self.settings = settings or get_settings()

    # Cross-superclass road actors a detection is most likely to actually be (India-weighted). Always
    # offered to the VLM so it can fix gross mislabels across superclasses, e.g. autorickshaw read as
    # sedan, or a person-on-a-scooter read as pedestrian.
    CROSS_ANCHORS = [
        "autorickshaw", "e_auto", "e_rickshaw", "motorcycle", "scooter", "cycle",
        "pedestrian", "rider", "cyclist", "sedan", "suv", "hatchback", "pickup",
        "truck", "lcv", "bus", "tempo", "water_tanker", "cattle", "dog",
        "push_cart", "vendor_handcart", "street_vendor",
    ]

    def _shortlist(self, class_id: int) -> list[str]:
        c = self.onto.by_id(class_id)
        # current class first, then the cross-superclass anchors (guaranteed presence), then L1
        # siblings for fine within-superclass refinement, then the fallback.
        ordered = [c.name]
        ordered += [n for n in self.CROSS_ANCHORS if self.onto.has_name(n)]
        ordered += [k.name for k in self.onto.classes if k.l1 == c.l1]
        ordered.append("object_fallback")
        names = list(dict.fromkeys(ordered))  # dedup, preserve order
        return names[: self.settings.models.vlm.shortlist_size]

    def _attr_schema(self) -> dict:
        return {
            name: {"type": a.type, **({"values": a.values} if a.values else {})}
            for name, a in self.onto.attributes.items()
        }

    def _validate(self, res: VlmResult) -> VlmResult:
        if res.class_name and not self.onto.has_name(res.class_name):
            res.class_name = None
        if res.attrs:
            res.attrs = {
                k: v for k, v in res.attrs.items()
                if k in self.onto.attributes and not self.onto.validate_attrs({k: v})
            }
        return res

    def _vote_plans(self, votes: int) -> list[tuple[float, float]]:
        """(crop_margin, temperature) per vote. Diversity comes from different context windows and
        sampling temperatures, so a genuinely ambiguous object yields disagreeing votes while a clear
        one is unanimous."""
        m = self.settings.models.vlm.crop_margin
        plans = [(m, 0.0), (m * 0.6, 0.5), (m * 1.7, 0.5), (m, 0.7), (m * 0.5, 0.4)]
        while len(plans) < votes:
            plans.append((m, 0.6))
        return plans[:votes]

    def verify_object(self, image_bgr: np.ndarray, bbox: tuple, class_id: int, votes: int | None = None) -> VlmResult:
        from collections import Counter

        votes = votes if votes is not None else self.settings.models.vlm.vote_count
        votes = max(1, votes)
        shortlist = self._shortlist(class_id)
        schema = self._attr_schema()

        results: list[VlmResult] = []
        for margin, temp in self._vote_plans(votes):
            crop = crop_object(image_bgr, bbox, margin)
            results.append(self._validate(self.client.verify(crop, shortlist, schema, temperature=temp)))

        classes = [r.class_name for r in results if r.class_name]
        if not classes:
            return VlmResult(votes=votes, agreement=0.0)

        majority, cnt = Counter(classes).most_common(1)[0]
        winners = [r for r in results if r.class_name == majority]
        merged_attrs: dict = {}
        for r in winners:
            for k, v in r.attrs.items():
                merged_attrs.setdefault(k, v)
        caption = next((r.caption for r in winners if r.caption), "")
        return VlmResult(
            class_name=majority, attrs=merged_attrs, caption=caption,
            confident=any(r.confident for r in winners),
            votes=votes, agreement=round(cnt / votes, 2),
        )
