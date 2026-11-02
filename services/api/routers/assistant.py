"""The assistant: ask in words, get answers from the data, approve the changes it proposes.

Chat is open to anyone who labels; approving a proposal is for reviewers, the same role that applies a bulk
edit by hand. See services/assistant for what it can and cannot do.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.models import AssistantProposal
from services.activity import record_activity
from services.api.deps import current_user, db_session, require_role
from services.api.project_guard import caller_hidden_projects
from services.assistant.llm import AssistantUnavailable
from services.assistant.loop import run_turn
from services.assistant.tools import Context, ToolError, execute
from services.project_access import as_hidden, project_of

router = APIRouter()


@router.get("/assistant/status", dependencies=[Depends(require_role("annotator"))])
async def status():
    """Which model answers, and whether it can be reached, so the page can say so before anyone types."""
    s = get_settings()
    cfg = s.assistant
    if cfg.provider == "groq":
        return {"provider": "groq", "model": cfg.groq_model or s.groq.text_model, "ready": bool(s.groq.api_key),
                "local": False, "detail": None if s.groq.api_key else "GROQ_API_KEY is not set"}
    if cfg.provider == "anthropic":
        return {"provider": "anthropic", "model": cfg.anthropic_model, "ready": bool(s.anthropic.api_key),
                "local": False, "detail": None if s.anthropic.api_key else "ANTHROPIC_API_KEY is not set"}
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{s.models.vlm.ollama_url.rstrip('/')}/api/tags")
        names = {m.get("name") for m in r.json().get("models", [])}
        ready = cfg.ollama_model in names
        detail = None if ready else f"run `ollama pull {cfg.ollama_model}` on the server"
    except (httpx.HTTPError, ValueError):
        ready, detail = False, "Ollama is not reachable; start it, or set LBX_ASSISTANT__PROVIDER"
    return {"provider": "ollama", "model": cfg.ollama_model, "ready": ready, "local": True, "detail": detail}


class Message(BaseModel):
    role: str
    content: str = Field(max_length=8000)


class ChatIn(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=40)
    session_id: str | None = None
    frame_id: str | None = None


@router.post("/assistant/chat", dependencies=[Depends(require_role("annotator"))])
async def chat(body: ChatIn, db: AsyncSession = Depends(db_session), user=Depends(current_user),
               hidden=Depends(caller_hidden_projects)):
    last = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
    if not last.strip():
        raise HTTPException(400, "nothing to answer: the last message from the person is empty")
    hidden = as_hidden(hidden)
    for kind, raw in (("session_id", body.session_id), ("frame_id", body.frame_id)):
        if raw:
            found, pid = await project_of(db, kind, raw)
            if not found or (pid is not None and pid in hidden):
                raise HTTPException(404, "not found")
    ctx = Context(db=db, user=user, hidden=hidden, request=last, session_id=body.session_id,
                  frame_id=body.frame_id)
    try:
        return await run_turn(ctx, [m.model_dump() for m in body.messages])
    except AssistantUnavailable as exc:
        raise HTTPException(503, f"the assistant's model is not available: {exc}") from None


def _scope(p: AssistantProposal) -> tuple[str, str] | None:
    args = p.args or {}
    plan = args.get("plan") or {}
    for kind in ("frame_id", "session_id"):
        if plan.get(kind):
            return kind, plan[kind]
        if args.get(kind):
            return kind, args[kind]
    return None


def _row(p: AssistantProposal) -> dict:
    return {"proposal_id": str(p.proposal_id), "tool": p.tool, "summary": p.summary, "preview": p.preview,
            "status": p.status, "result": p.result, "request": p.request,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "decided_at": p.decided_at.isoformat() if p.decided_at else None}


async def _visible(db: AsyncSession, p: AssistantProposal, hidden: frozenset) -> bool:
    if not hidden:
        return True
    scope = _scope(p)
    if scope is None:
        return True
    found, pid = await project_of(db, *scope)
    return not (found and pid is not None and pid in hidden)


@router.get("/assistant/proposals", dependencies=[Depends(require_role("annotator"))])
async def proposals(status: str = "pending", limit: int = 50, db: AsyncSession = Depends(db_session),
                    hidden=Depends(caller_hidden_projects)):
    q = select(AssistantProposal).order_by(AssistantProposal.created_at.desc()).limit(min(max(limit, 1), 200))
    if status != "all":
        q = q.where(AssistantProposal.status == status)
    rows = (await db.execute(q)).scalars().all()
    hidden = as_hidden(hidden)
    return {"proposals": [_row(p) for p in rows if await _visible(db, p, hidden)]}


async def _pending(db: AsyncSession, proposal_id: str, hidden: frozenset) -> AssistantProposal:
    try:
        p = await db.get(AssistantProposal, uuid.UUID(proposal_id))
    except ValueError:
        raise HTTPException(400, "invalid proposal id") from None
    if p is None or not await _visible(db, p, hidden):
        raise HTTPException(404, "proposal not found")
    if p.status != "pending":
        raise HTTPException(409, f"this proposal was already {p.status}")
    return p


@router.post("/assistant/proposals/{proposal_id}/approve")
async def approve(proposal_id: str, db: AsyncSession = Depends(db_session), user=Depends(require_role("reviewer")),
                  hidden=Depends(caller_hidden_projects)):
    """Run the proposal through the path a person would use. A refusal from that path (a busy GPU, a session
    that failed its health checks) marks the proposal failed with the reason, rather than leaving it pending."""
    p = await _pending(db, proposal_id, as_hidden(hidden))
    uid = getattr(user, "user_id", None)
    try:
        result = await execute(db, p, user)
        p.status, p.result = "approved", result if isinstance(result, dict) else {"result": result}
    except (HTTPException, ToolError) as exc:
        # A rollback expires everything the session holds, the signed-in user included, and an expired
        # attribute read later would be a lazy load outside async IO. Reload what is used below.
        await db.rollback()
        if user is not None and user in db:
            await db.refresh(user)
        reloaded = await db.get(AssistantProposal, uuid.UUID(proposal_id), populate_existing=True)
        assert reloaded is not None  # it existed a moment ago, and only this request is deciding it
        p = reloaded
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        p.status, p.result = "failed", {"error": detail}
    p.decided_by, p.decided_at = uid, datetime.now(UTC)
    await record_activity(db, user=user, verb=f"assistant_{p.status}", subject_type="assistant_proposal",
                          subject_id=str(p.proposal_id), summary=p.summary, commit=False)
    await db.commit()
    return _row(p)


@router.post("/assistant/proposals/{proposal_id}/reject")
async def reject(proposal_id: str, db: AsyncSession = Depends(db_session), user=Depends(require_role("reviewer")),
                 hidden=Depends(caller_hidden_projects)):
    p = await _pending(db, proposal_id, as_hidden(hidden))
    p.status, p.decided_by, p.decided_at = "rejected", getattr(user, "user_id", None), datetime.now(UTC)
    await record_activity(db, user=user, verb="assistant_rejected", subject_type="assistant_proposal",
                          subject_id=str(p.proposal_id), summary=p.summary, commit=False)
    await db.commit()
    return _row(p)


@router.post("/assistant/eval", dependencies=[Depends(require_role("admin"))])
async def evaluate_assistant(db: AsyncSession = Depends(db_session)):
    """Score the configured model on the fixed request set (services/assistant/eval.py): does its first tool
    call pick the right tool with the right arguments. Runs no tool and writes nothing."""
    from services.assistant.eval import evaluate

    try:
        return await evaluate(db)
    except AssistantUnavailable as exc:
        raise HTTPException(503, f"the assistant's model is not available: {exc}") from None
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None
