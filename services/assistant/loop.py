"""One turn of the assistant: the model calls tools until it can answer, within a step budget.

Reading tools run and their results go back to the model. Proposing tools store a proposal and tell the model
it is waiting for a reviewer, so its answer can say what it proposed but cannot claim it happened. A bad
argument comes back as the tool result, so the model can correct itself (a misspelt class, a missing scope)
instead of the turn failing.
"""

from __future__ import annotations

import asyncio
import json

from core.config import get_settings
from core.logging import get_logger
from services.assistant.llm import ChatModel, make_model
from services.assistant.tools import TOOLS, Context, ToolError

log = get_logger("assistant")

SYSTEM = """You are the assistant inside LabeloxAV, a tool a small team uses to label driving data: recorded \
drives (sessions) of camera frames, with labels (objects) that have a class, a box and a review state. \
Machine labels wait in "review" or "annotate" until a person accepts or rejects them.

Rules:
- Use the tools to answer. Never state a number you did not get from a tool in this conversation.
- Class names must be exact ontology names. If unsure, call list_classes first.
- You cannot change anything yourself. Tools named propose_* create a proposal a reviewer approves or rejects. \
After proposing, say what you proposed and that it waits for approval. Never say a change has been made.
- A change needs a scope: one session or one frame. If the person did not say which, use the session they are \
working in when one is given below, otherwise ask or call find_sessions.
- If a lookup finds nothing but its result lists what does exist (a city code, a class name), retry with the \
matching one before answering.
- Be brief and concrete. Lead with the answer."""


def _system(ctx: Context) -> str:
    where = []
    if ctx.session_id:
        where.append(f"session_id {ctx.session_id}")
    if ctx.frame_id:
        where.append(f"frame_id {ctx.frame_id}")
    return SYSTEM + (f"\n\nThe person is working in {' and '.join(where)}." if where else "")


async def _call_tool(ctx: Context, name: str, args: dict) -> dict:
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"no tool named {name}; available: {sorted(TOOLS)}"}
    try:
        return await tool.handler(ctx, args or {})
    except ToolError as exc:
        return {"error": str(exc)}


def _brief(result: dict) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= 6000 else text[:6000] + '..."(truncated)"'


async def run_turn(ctx: Context, messages: list[dict], model: ChatModel | None = None) -> dict:
    """Answer the last user message. `messages` holds the conversation so far as role/content pairs."""
    model = model or make_model()
    steps = get_settings().assistant.max_steps
    convo: list[dict] = [{"role": m["role"], "content": m["content"]} for m in messages
                         if m.get("role") in ("user", "assistant") and m.get("content")]
    trace: list[dict] = []
    specs = [t.spec() for t in TOOLS.values()]
    for _ in range(steps):
        reply = await asyncio.to_thread(model.chat, _system(ctx), convo, specs)
        if not reply.tool_calls:
            return {"reply": reply.content.strip(), "trace": trace, "proposals": ctx.proposals, "model": model.name}
        convo.append({"role": "assistant", "content": reply.content,
                      "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls]})
        for call in reply.tool_calls:
            result = await _call_tool(ctx, call.name, call.arguments)
            trace.append({"tool": call.name, "arguments": call.arguments,
                          "kind": TOOLS[call.name].kind if call.name in TOOLS else "unknown",
                          "error": result.get("error")})
            convo.append({"role": "tool", "tool_call_id": call.id, "content": _brief(result)})
    log.info("assistant.step_budget_spent", steps=steps)
    return {"reply": "I ran out of steps before finishing. Ask a narrower question, or name the session.",
            "trace": trace, "proposals": ctx.proposals, "model": model.name}


__all__ = ["SYSTEM", "run_turn"]
