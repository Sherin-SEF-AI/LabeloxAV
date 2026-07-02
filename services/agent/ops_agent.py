"""The Operations Agent ("Ask LabeloxAV"): turns a plain-language request into a plan over the platform's
existing services and runs it, asking only when a step is genuinely ambiguous or mutating.

It plans with the local LLM when available (a strict-JSON tool-selection over a small catalog, under a token
budget) and falls back to a deterministic rule parser otherwise, so it always produces a plan. Read-only
steps (coverage, find sessions, compose/preview a slice) run immediately; mutating steps (autolabel a
session, export a dataset) are gated behind an explicit confirm. Every run is tracked on an AgentRun and
audited. This is the UX multiplier: the platform becomes operable in sentences, and it is cheap because
every tool it calls already exists as a service.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun

log = get_logger("agent.ops_agent")

_KIND = "ops_agent"


# --- tool catalog -----------------------------------------------------------------------------------------

async def _t_coverage(db: AsyncSession, args: dict) -> dict:
    from services.agent.coverage import analyze_coverage

    cov = await analyze_coverage(db)
    return {"gaps": cov.get("gaps", []), "class_balance_median": cov.get("class_balance", {}).get("median")}


async def _t_find_sessions(db: AsyncSession, args: dict) -> dict:
    from db.models import Session as S

    q = select(S.session_id, S.vehicle_id, S.city)
    if args.get("city"):
        q = q.where(S.city.ilike(f"%{args['city']}%"))
    rows = (await db.execute(q.limit(int(args.get("limit", 20))))).all()
    return {"count": len(rows), "sessions": [{"session_id": str(s), "vehicle": v, "city": c} for s, v, c in rows]}


async def _t_create_slice(db: AsyncSession, args: dict) -> dict:
    from services.curation.slices import create_slice

    return await create_slice(args["name"], args.get("predicate", {}), args.get("description"))


async def _t_materialize(db: AsyncSession, args: dict) -> dict:
    from services.curation.slices import materialize_slice

    return await materialize_slice(args["slice_id"], int(args.get("sample", 20)))


async def _t_autolabel(db: AsyncSession, args: dict) -> dict:
    import asyncio

    from db.models import AutolabelJob, TrainingJob
    from db.session import get_sessionmaker

    if (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first():
        return {"error": "GPU reserved for a training job; autolabel paused"}
    if (await db.execute(select(AutolabelJob.job_id).where(AutolabelJob.status == "running").limit(1))).first():
        return {"error": "an autolabel job is already running"}
    sid = uuid.UUID(args["session_id"])
    job_id = uuid.uuid4()
    db.add(AutolabelJob(job_id=job_id, session_id=sid, status="pending"))
    await db.commit()

    async def _run() -> None:
        from services.autolabel.runner import autolabel_session

        try:
            async with get_sessionmaker()() as d:
                j = await d.get(AutolabelJob, job_id)
                j.status = "running"
                await d.commit()
            res = await autolabel_session(sid, args.get("limit"))
            async with get_sessionmaker()() as d:
                j = await d.get(AutolabelJob, job_id)
                j.status, j.counts = "done", res
                await d.commit()
        except Exception as exc:  # noqa: BLE001
            async with get_sessionmaker()() as d:
                j = await d.get(AutolabelJob, job_id)
                if j:
                    j.status, j.error = "error", str(exc)
                    await d.commit()

    asyncio.create_task(_run())
    return {"job_id": str(job_id), "status": "running"}


async def _t_export(db: AsyncSession, args: dict) -> dict:
    import asyncio

    from services.export.dataset import SliceSpec, export_dataset

    spec = SliceSpec(name=args.get("name", "ops-export"), class_names=args.get("class_names"),
                     cities=args.get("cities"), states=args.get("states", ["accepted", "auto_accept"]),
                     formats=args.get("formats", ["coco", "parquet"]), limit=args.get("limit"))

    async def _run() -> None:
        try:
            await export_dataset(spec)
        except Exception as exc:  # noqa: BLE001
            log.error("ops.export_failed", error=str(exc))

    asyncio.create_task(_run())
    return {"status": "export started", "formats": spec.formats, "name": spec.name}


TOOLS: dict[str, dict] = {
    "coverage": {"mutating": False, "run": _t_coverage,
                 "desc": "report class/scene/geo coverage gaps"},
    "find_sessions": {"mutating": False, "run": _t_find_sessions,
                      "desc": "list sessions, optionally filtered by city; args: {city?, limit?}"},
    "create_slice": {"mutating": False, "run": _t_create_slice,
                     "desc": "compose a named curation slice; args: {name, predicate:{scene?,cities?,class_names?,states?,min_conf?}}"},
    "materialize_slice": {"mutating": False, "run": _t_materialize,
                          "desc": "count + sample a slice; args: {slice_id, sample?}"},
    "autolabel": {"mutating": True, "run": _t_autolabel,
                  "desc": "auto-label a session on the GPU; args: {session_id, compute_target?, limit?}"},
    "export": {"mutating": True, "run": _t_export,
               "desc": "export a dataset; args: {name, class_names?, cities?, formats?, limit?}"},
}


# --- planning ---------------------------------------------------------------------------------------------

def _rule_plan(text: str) -> list[dict]:
    """Deterministic fallback: map obvious phrasings to a single tool, so the agent always has a plan."""
    t = text.lower()
    if "coverage" in t or "gap" in t:
        return [{"tool": "coverage", "args": {}}]
    if "export" in t:
        args: dict = {}
        for fmt in ("coco", "yolo", "kitti", "nuscenes", "bdd", "parquet"):
            if fmt in t:
                args.setdefault("formats", []).append(fmt)
        return [{"tool": "export", "args": args}]
    if "auto-label" in t or "autolabel" in t or "label" in t:
        return [{"tool": "autolabel", "args": {}}]
    if "session" in t or "ingest" in t:
        return [{"tool": "find_sessions", "args": {}}]
    return []


def _llm_plan(text: str, budget) -> list[dict] | None:
    try:
        import json

        import httpx

        from core.config import get_settings

        cfg = get_settings().models.vlm
        if not getattr(cfg, "enabled", False) or budget.exhausted:
            return None
        catalog = "\n".join(f"- {n}: {t['desc']}" for n, t in TOOLS.items())
        prompt = (
            "You plan data-platform operations as a sequence of tool calls. Tools:\n" + catalog +
            "\nReturn STRICT JSON only: {\"steps\": [{\"tool\": \"<name>\", \"args\": {..}}]}. "
            "Use only the tools listed. Keep the plan minimal.\n"
            f'Request: "{text}"')
        payload = {"model": cfg.ollama_tag, "stream": False, "format": "json",
                   "messages": [{"role": "user", "content": prompt}], "options": {"temperature": 0.0}}
        resp = httpx.post(f"{cfg.ollama_url}/api/chat", json=payload, timeout=min(getattr(cfg, "timeout_s", 20), 20))
        budget.charge(1)
        resp.raise_for_status()
        data = json.loads(resp.json()["message"]["content"])
        steps = [s for s in data.get("steps", []) if isinstance(s, dict) and s.get("tool") in TOOLS]
        return steps or None
    except Exception:  # noqa: BLE001
        return None


def plan(text: str, budget) -> dict:
    steps = _llm_plan(text, budget)
    source = "llm"
    if not steps:
        steps, source = _rule_plan(text), "rules"
    # normalize
    steps = [{"tool": s["tool"], "args": s.get("args", {}) or {},
              "mutating": TOOLS[s["tool"]]["mutating"]} for s in steps if s.get("tool") in TOOLS]
    return {"steps": steps, "source": source}


async def execute(db: AsyncSession, steps: list[dict], *, confirm: bool = False, created_by: str | None = None) -> dict:
    """Run read steps; stop at the first mutating step unless confirmed. Records the run + audit."""
    run_id = uuid.uuid4()
    results, ran, pending = [], [], None
    for step in steps:
        if step["mutating"] and not confirm:
            pending = step
            break
        try:
            out = await TOOLS[step["tool"]]["run"](db, step["args"])
        except Exception as exc:  # noqa: BLE001
            out = {"error": str(exc)}
        results.append({"tool": step["tool"], "args": step["args"], "result": out})
        ran.append(step["tool"])
    status = "awaiting_confirmation" if pending else "committed"
    db.add(AgentRun(run_id=run_id, kind=_KIND, scope={}, status="committed", policy={"steps": steps},
                    counts={"ran": ran, "status": status}, changes={}, critic={}, created_by=created_by or "ops"))
    await db.commit()
    try:
        from services.govern.audit import record

        await record(db, _KIND, "ops_run", str(run_id), {"ran": ran, "status": status})
    except Exception:  # noqa: BLE001
        pass
    return {"run_id": str(run_id), "status": status, "results": results,
            "pending": pending, "ran": ran}


async def ask(db: AsyncSession, text: str, *, confirm: bool = False, created_by: str | None = None,
              vlm_calls: int = 2) -> dict:
    """Plan from the sentence, then execute the read steps (and mutating steps only if confirmed)."""
    from services.agent.runtime.budget import TokenBudget

    p = plan(text, TokenBudget(vlm_calls))
    if not p["steps"]:
        return {"plan": p, "status": "no_plan", "message": "could not map that request to a known operation"}
    result = await execute(db, p["steps"], confirm=confirm, created_by=created_by)
    return {"plan": p, **result}
