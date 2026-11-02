"""Tool-calling chat for the assistant, over the providers this deployment already knows.

One message format inside (OpenAI-style roles; tool calls carry parsed arguments), converted at the edge:
  - ollama and groq speak the OpenAI chat-completions API (Ollama at <ollama_url>/v1), so one backend serves
    both;
  - anthropic speaks the Messages API, where tool calls are content blocks and tool results ride in a user
    message.
Only text goes out: the system prompt, the conversation and tool results, which are counts, class names and
ids. With the default provider (ollama) nothing leaves the machine.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from core.config import Settings, get_settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    name: str

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply: ...


class AssistantUnavailable(RuntimeError):
    """The configured provider cannot be reached or is not configured."""


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


class OpenAICompatible:
    """Ollama (/v1) and Groq."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: float = 90.0):
        self.base_url, self.model, self.api_key, self.timeout_s = base_url.rstrip("/"), model, api_key, timeout_s
        self.name = model

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or "",
                            "tool_calls": [{"id": c["id"], "type": "function",
                                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                           for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        body = {"model": self.model, "temperature": 0, "messages": self._messages(system, messages),
                "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": "auto"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise AssistantUnavailable(f"{self.model}: {exc}") from exc
        msg = r.json()["choices"][0]["message"]
        calls = [ToolCall(id=c.get("id") or f"call_{uuid.uuid4().hex[:8]}", name=c["function"]["name"],
                          arguments=_parse_args(c["function"].get("arguments")))
                 for c in (msg.get("tool_calls") or [])]
        return Reply(content=msg.get("content") or "", tool_calls=calls)


class Anthropic:
    def __init__(self, settings: Settings, model: str, timeout_s: float = 90.0):
        self.cfg, self.model, self.timeout_s = settings.anthropic, model, timeout_s
        self.name = model

    @staticmethod
    def _messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                blocks: list[dict] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                           for c in m.get("tool_calls") or []]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # Results of one round of calls go back together, in one user message.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": "user", "content": m.get("content") or ""})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        if not self.cfg.api_key:
            raise AssistantUnavailable("anthropic: no API key configured")
        body = {"model": self.model, "max_tokens": 2048, "temperature": 0, "system": system,
                "messages": self._messages(messages),
                "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                          for t in tools]}
        headers = {"x-api-key": self.cfg.api_key, "anthropic-version": self.cfg.api_version}
        try:
            r = httpx.post(f"{self.cfg.base_url.rstrip('/')}/messages", json=body, headers=headers,
                           timeout=self.timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise AssistantUnavailable(f"{self.model}: {exc}") from exc
        reply = Reply()
        for block in r.json().get("content", []):
            if block.get("type") == "text":
                reply.content += block.get("text", "")
            elif block.get("type") == "tool_use":
                reply.tool_calls.append(ToolCall(id=block["id"], name=block["name"],
                                                 arguments=_parse_args(block.get("input"))))
        return reply


def make_model(settings: Settings | None = None) -> ChatModel:
    settings = settings or get_settings()
    cfg = settings.assistant
    if cfg.provider == "groq":
        if not settings.groq.api_key:
            raise AssistantUnavailable("groq: no API key configured")
        return OpenAICompatible(settings.groq.base_url, cfg.groq_model or settings.groq.text_model,
                                settings.groq.api_key, cfg.timeout_s)
    if cfg.provider == "anthropic":
        return Anthropic(settings, cfg.anthropic_model, cfg.timeout_s)
    return OpenAICompatible(f"{settings.models.vlm.ollama_url.rstrip('/')}/v1", cfg.ollama_model,
                            timeout_s=cfg.timeout_s)


__all__ = ["Anthropic", "AssistantUnavailable", "ChatModel", "OpenAICompatible", "Reply", "ToolCall", "make_model"]
