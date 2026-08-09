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
from services.autolabel.paths.constrain import static_prompt, verdict_schema

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
    provider: str = ""         # which backend served this verdict (ollama | groq | groq:escalate), for provenance


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
    """The prompt for a backend that cannot enforce a schema, so the shortlist has to be asked for in words.

    Anthropic has no response_format, so this is still the only lever there. A constrained backend uses
    `constrain.static_prompt` instead, which omits the shortlist because the schema enforces it and because
    leaving it in would vary the prefix per object and cost the server its cache.
    """
    # The domain preamble comes from the active pack (AV: the Indian-road-scene prompt); the rest of the
    # instruction is generic. Byte-identical for AV.
    from services.domain import active_pack

    template = active_pack().autolabel_profile.vlm_prompt_template
    return (
        f"{template}\n"
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
        # Named to match the cloud clients so callers that need to ask their own question can duck-type on
        # it. Without this the local path could only ever be asked verify()'s question, which is
        # "what is this?"; the judge needs to ask "the label says X, is that right?" and those are
        # different questions with different failure modes.
        self.model = self.cfg.ollama_tag

    def chat_json(self, prompt: str, *, model: str | None = None, image_jpeg: bytes | None = None,
                  temperature: float = 0.0, schema: dict | None = None) -> dict:
        """One chat turn returning a JSON object, same contract as GroqClient/AnthropicClient.

        Raises on failure rather than returning {}, because an empty object reads downstream as a real but
        empty answer, and for a judge that means "found nothing wrong".

        `schema` constrains the reply the same way verify() constrains a class choice. A judge returning a
        categorical verdict is the case that benefits most: the whole value of the answer is that it lands in
        a known set, and an unparseable "probably not, though it could be" is worth nothing to a caller
        counting agreements.
        """
        payload: dict = {
            "model": model or self.cfg.ollama_tag,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema if schema else "json",
            "options": {"num_ctx": self.cfg.max_context, "temperature": temperature},
        }
        if image_jpeg is not None:
            payload["messages"][0]["images"] = [base64.b64encode(image_jpeg).decode()]
        resp = httpx.post(f"{self.cfg.ollama_url}/api/chat", json=payload, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])

    def verify(
        self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict, temperature: float = 0.0
    ) -> VlmResult:
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return VlmResult()
        b64 = base64.b64encode(buf.tobytes()).decode()
        # `format` takes a JSON schema, not only the string "json". Passing the schema makes the shortlist a
        # constraint on sampling rather than a request in the prompt, so an out-of-vocabulary class can no
        # longer be emitted and then discarded by _validate. The prompt loses the shortlist in exchange and
        # becomes identical across objects, which is what makes prefix caching worth anything here.
        schema = verdict_schema(shortlist, attr_schema)
        payload = {
            "model": self.cfg.ollama_tag,
            "messages": [{"role": "user", "content": static_prompt(attr_schema), "images": [b64]}],
            "stream": False,
            "format": schema,
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
    """The VLM client for the configured providers. Delegates to the router (Groq cloud + Ollama local with
    fallback); when nothing is pointed at the cloud the router returns a plain OllamaVlmClient, identical to
    the previous behaviour. Lazy import breaks the path_c <-> router cycle."""
    from services.llm.router import make_vlm_client as _routed

    return _routed(settings)


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

    # Record which provider served the verdict alongside the model tag, so a gate decision is traceable to
    # ollama vs groq (vs the escalate provider) and the two can be compared on the gold set.
    tag = f"{vlm_tag}@{res.provider}" if res.provider else vlm_tag
    obj.provenance.proposals.append(
        PathProposal(path="path_c_qwen3vl", class_name=res.class_name, conf=None, verdict=verdict, model_version=tag)
    )
    if res.caption:
        obj.provenance.notes.append(f"caption: {res.caption}")
    return obj


class VlmVerifier:
    """Applies a VlmClient to a fused object: builds the shortlist, validates the reply against the
    ontology, and reports the class/attrs to merge back. Pure of DB and GPU so it is unit-testable.
    """

    def __init__(self, client: VlmClient, ontology: Ontology | None = None, settings: Settings | None = None,
                 supported_ids: set[int] | None = None) -> None:
        self.client = client
        self.onto = ontology or get_ontology()
        self.settings = settings or get_settings()
        # M-Q.0: when the grounded set is provided, the shortlist offered to the VLM is restricted to it, so
        # the VLM cannot "confirm" an ungrounded class (e.g. a fixed-class sibling like bus_shelter).
        self.supported_ids = supported_ids

    def _shortlist(self, class_id: int) -> list[str]:
        c = self.onto.by_id(class_id)

        def grounded(name: str) -> bool:
            # with the grounded set, only supported names (plus the object's own class) are offered
            return self.supported_ids is None or not self.onto.has_name(name) or self.onto.by_name(name).id in self.supported_ids

        # current class first, then the cross-superclass anchors (guaranteed presence), then L1
        # siblings for fine within-superclass refinement, then the fallback. All restricted to the
        # grounded set so the VLM cannot confirm an ungrounded class. Anchors come from the active pack.
        from services.domain import active_pack

        anchors = active_pack().autolabel_profile.cross_anchors
        ordered = [c.name]
        ordered += [n for n in anchors if self.onto.has_name(n) and grounded(n)]
        ordered += [k.name for k in self.onto.classes if k.l1 == c.l1 and grounded(k.name)]
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
            # Logged rather than silently nulled. This is the whole verdict being thrown away, crop and
            # prefill and decode included, and until now it left no trace, so the rate at which it happened
            # was unknown and the case for constraining generation could not be made with a number. A
            # constrained backend cannot reach this line at all.
            log.warning("vlm.class_out_of_ontology", emitted=res.class_name[:64], provider=res.provider)
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
            provider=next((r.provider for r in winners if r.provider), ""),
        )


class LlamaServerVlmClient:
    """Path C served by llama.cpp's llama-server, over its OpenAI-compatible endpoint.

    The reason to prefer it over Ollama for this particular job is that it takes the verdict schema in
    `response_format` and compiles it to GBNF server-side, so the shortlist constrains sampling. Ollama's
    `format` does the same thing, so the constraint is not exclusive to llama.cpp; what llama-server adds is
    explicit control of the prefix cache (`--cache-reuse`), which the static prompt was made for, and a
    single static binary with no Python or CUDA dependency matrix, which is the on-prem story.

    Duck-typed against OllamaVlmClient and the cloud clients on purpose: it carries `model`, `chat_json` and
    `verify`, so the router's policy (circuit breaker, escalate, provider attribution) does not change to
    accommodate it. `provider` is recorded on every verdict, so this can be compared against Ollama on the
    gold set rather than asserted to be better.
    """

    provider_name = "llamacpp"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.models.vlm
        self.model = self.cfg.llamacpp_model or "local"

    def _post(self, payload: dict) -> dict:
        resp = httpx.post(f"{self.cfg.llamacpp_url}/v1/chat/completions", json=payload,
                          timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    def _messages(self, prompt: str, b64: str | None) -> list[dict]:
        if b64 is None:
            return [{"role": "user", "content": prompt}]
        # llama-server's multimodal stack takes OpenAI-style content parts with a data URL, not Ollama's
        # separate `images` array.
        return [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}]

    def chat_json(self, prompt: str, *, model: str | None = None, image_jpeg: bytes | None = None,
                  temperature: float = 0.0, schema: dict | None = None) -> dict:
        """One chat turn returning a JSON object, same contract as the other clients.

        Raises rather than returning {}, because an empty object reads downstream as a real but empty
        answer, and for a judge that means "found nothing wrong".
        """
        b64 = base64.b64encode(image_jpeg).decode() if image_jpeg is not None else None
        rf = ({"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}}
              if schema else {"type": "json_object"})
        return self._post({
            "model": model or self.model,
            "messages": self._messages(prompt, b64),
            "temperature": temperature,
            "response_format": rf,
        })

    def verify(self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict,
               temperature: float = 0.0) -> VlmResult:
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return VlmResult()
        schema = verdict_schema(shortlist, attr_schema)
        try:
            data = self._post({
                "model": self.model,
                "messages": self._messages(static_prompt(attr_schema),
                                           base64.b64encode(buf.tobytes()).decode()),
                "temperature": temperature,
                # Compiled to GBNF by the server. `strict` is what makes it a constraint rather than a hint.
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "verdict", "strict": True, "schema": schema}},
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("vlm.call_failed", error=str(exc), provider=self.provider_name)
            return VlmResult()

        return VlmResult(
            class_name=data.get("class"),
            attrs=data.get("attributes", {}) or {},
            caption=data.get("caption", "") or "",
            confident=bool(data.get("confident", False)),
            raw=data,
            provider=self.provider_name,
        )
