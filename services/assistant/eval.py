"""Does the assistant pick the right tool with the right arguments? A fixed set of requests, scored.

Each case is something a person on a labelling team would type, including Indian city names and a Hinglish
request, with the acceptable first tool calls and a check on their arguments. Only the model's first reply is
scored and no tool runs, so the measurement writes nothing and reads only the session and frame it names.

Scored per case: the first tool is one of the acceptable ones (tool), and its arguments pass the check
(arguments). Asking a change of the assistant must produce a propose_* call: a case where the model reaches
for a read tool first is scored as a wrong tool, since the person asked for something to be done.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame
from db.models import Session as DbSession
from services.assistant.llm import ChatModel, make_model
from services.assistant.loop import SYSTEM
from services.assistant.tools import TOOLS, ToolError, _class_ids

Check = Callable[[dict], bool]


@dataclass
class Case:
    text: str
    accept: dict[str, Check]           # tool name -> argument check
    in_session: bool = False           # the person is working in the eval session
    in_frame: bool = False


def _classes(args: dict, key: str, *names: str) -> bool:
    try:
        got = set(_class_ids(args.get(key) or []))
        return set(_class_ids(list(names))) <= got
    except ToolError:
        return False


def _has(args: dict, key: str, value: str) -> bool:
    v = args.get(key)
    vals = v if isinstance(v, list) else [v]
    return any(isinstance(x, str) and value.lower() in x.lower() for x in vals)


def _any(_args: dict) -> bool:
    return True


def cases(sid: str, fid: str) -> list[Case]:
    return [
        Case("How many pedestrians are waiting for review?",
             {"queue_overview": lambda a: _classes(a, "class_names", "pedestrian"),
              "count_objects": lambda a: _classes(a, "class_names", "pedestrian")}),
        Case("How many autorickshaws have we accepted so far?",
             {"count_objects": lambda a: _classes(a, "class_names", "autorickshaw") and _has(a, "states", "accepted")}),
        Case("How many trucks are there with confidence below 0.4?",
             {"count_objects": lambda a: _classes(a, "class_names", "truck") and float(a.get("max_confidence") or 1) <= 0.4}),
        Case("How many cows have been labelled?",
             {"count_objects": lambda a: _classes(a, "class_names", "cattle"), "list_classes": _any}),
        Case("How big is the review queue right now?", {"queue_overview": _any}),
        Case("Show me the drives from Bengaluru.",
             {"find_sessions": lambda a: _has(a, "query", "beng") or _has(a, "query", "blr") or _has(a, "query", "bangalore")}),
        Case("Delhi wale drives dikhao", {"find_sessions": lambda a: _has(a, "query", "del")}),
        Case("Do we have any KITTI recordings?", {"find_sessions": lambda a: _has(a, "query", "kitti")}),
        Case(f"What is in session {sid}?", {"session_overview": lambda a: a.get("session_id") == sid}),
        Case("Give me an overview of this session.", {"session_overview": _any}, in_session=True),
        Case("Which error detector has the most pending candidates?", {"error_overview": _any}),
        Case("Is tuk tuk one of our classes?", {"list_classes": _any}),
        Case("Which classes count as two-wheelers?", {"list_classes": lambda a: _has(a, "query", "two")}),
        Case(f"Change all the hatchbacks in session {sid} to sedan.",
             {"propose_reclassify": lambda a: _classes(a, "from_classes", "hatchback")
              and _classes({"x": [a.get("to_class", "")]}, "x", "sedan") and a.get("session_id") == sid}),
        Case("Relabel the e-rickshaws in this session as autorickshaw.",
             {"propose_reclassify": lambda a: _classes(a, "from_classes", "e_rickshaw")
              and _classes({"x": [a.get("to_class", "")]}, "x", "autorickshaw")
              and a.get("session_id", sid) == sid}, in_session=True),
        Case(f"Send the cyclists in frame {fid} back for review.",
             {"propose_flag_for_review": lambda a: _classes(a, "class_names", "cyclist") and a.get("frame_id") == fid}),
        Case("Flag every dog in this frame for another look.",
             {"propose_flag_for_review": lambda a: _classes(a, "class_names", "dog")
              and a.get("frame_id", fid) == fid}, in_frame=True),
        Case(f"Auto-label session {sid}.", {"propose_autolabel": lambda a: a.get("session_id") == sid}),
        Case("Run auto-labelling on this drive, only the frames that have no labels yet.",
             {"propose_autolabel": lambda a: a.get("session_id", sid) == sid and a.get("only_unlabelled") is not False},
             in_session=True),
        Case(f"Look for label mistakes in session {sid}.",
             {"propose_error_detection": lambda a: a.get("session_id") == sid}),
        Case("Start auto-labelling the Delhi drive.", {"find_sessions": lambda a: _has(a, "query", "del")}),
        Case("Is bus a class?", {"list_classes": _any}),
    ]


async def _anchor(db: AsyncSession) -> tuple[str, str]:
    row = (await db.execute(select(DbSession.session_id, Frame.frame_id).join(Frame, Frame.session_id == DbSession.session_id)
                            .order_by(DbSession.created_at.desc()).limit(1))).first()
    if row is None:
        raise LookupError("no session with frames to anchor the eval cases on")
    return str(row[0]), str(row[1])


def _system(sid: str, fid: str, case: Case) -> str:
    where = []
    if case.in_session or case.in_frame:
        where.append(f"session_id {sid}")
    if case.in_frame:
        where.append(f"frame_id {fid}")
    return SYSTEM + (f"\n\nThe person is working in {' and '.join(where)}." if where else "")


async def evaluate(db: AsyncSession, model: ChatModel | None = None) -> dict:
    model = model or make_model()
    sid, fid = await _anchor(db)
    specs = [t.spec() for t in TOOLS.values()]
    results = []
    for case in cases(sid, fid):
        reply = await asyncio.to_thread(model.chat, _system(sid, fid, case), [{"role": "user", "content": case.text}], specs)
        first = reply.tool_calls[0] if reply.tool_calls else None
        tool_ok = first is not None and first.name in case.accept
        try:
            args_ok = bool(tool_ok and first is not None and case.accept[first.name](first.arguments))
        except (TypeError, ValueError):
            args_ok = False
        results.append({"text": case.text, "tool": first.name if first else None,
                        "arguments": first.arguments if first else None, "tool_ok": tool_ok, "args_ok": args_ok,
                        "expected": sorted(case.accept)})
    n = len(results)
    return {"model": model.name, "cases": n,
            "tool_accuracy": round(sum(r["tool_ok"] for r in results) / n, 4),
            "full_accuracy": round(sum(r["args_ok"] for r in results) / n, 4),
            "results": results}


__all__ = ["Case", "cases", "evaluate"]
