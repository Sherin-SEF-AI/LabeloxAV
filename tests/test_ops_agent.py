"""Operations Agent: the deterministic rule planner, and the execute gating that stops at a mutating step
until confirmed."""

from __future__ import annotations

import asyncio

import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


def test_rule_plan_maps_phrasings():
    from services.agent.ops_agent import _rule_plan

    assert _rule_plan("show me the coverage gaps") == [{"tool": "coverage", "args": {}}]
    exp = _rule_plan("export the night frames to coco")
    assert exp[0]["tool"] == "export" and "coco" in exp[0]["args"]["formats"]
    assert _rule_plan("auto-label this session")[0]["tool"] == "autolabel"
    assert _rule_plan("list sessions in whitefield")[0]["tool"] == "find_sessions"


def _force_rules(monkeypatch=None):
    import services.agent.ops_agent as ops

    ops._llm_plan = lambda text, budget: None   # deterministic: always use the rule fallback


@requires_infra
def test_execute_gates_mutating_step():
    from services.agent.ops_agent import execute, plan
    from db.session import get_sessionmaker

    _force_rules()

    async def _flow():
        # a read step then a mutating step: read runs, mutating pauses for confirmation
        steps = [{"tool": "coverage", "args": {}, "mutating": False},
                 {"tool": "export", "args": {"formats": ["coco"]}, "mutating": True}]
        async with get_sessionmaker()() as db:
            r = await execute(db, steps, confirm=False)
        assert r["status"] == "awaiting_confirmation"
        assert r["pending"]["tool"] == "export"
        assert "coverage" in r["ran"] and "export" not in r["ran"]

    run_async(_flow())


@requires_infra
def test_ask_runs_read_plan():
    from services.agent.ops_agent import ask
    from db.session import get_sessionmaker

    _force_rules()

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await ask(db, "what are the coverage gaps")
        assert r["status"] == "committed"
        assert r["results"][0]["tool"] == "coverage"
        assert "gaps" in r["results"][0]["result"]

    run_async(_flow())
