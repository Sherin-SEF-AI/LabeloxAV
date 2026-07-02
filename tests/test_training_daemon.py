"""Gold-drift monitor: roll back the champion when it regresses on gold beyond tolerance."""

from __future__ import annotations

import asyncio
import uuid

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


async def _seed_champion(baseline_map: float):
    """A champion at baseline_map promoted over a prior version, so a rollback has somewhere to go."""
    from db.models import ModelRegistry
    from db.session import get_sessionmaker
    from sqlalchemy import delete

    tag = f"champ-{uuid.uuid4().hex[:8]}"
    prev = f"prev-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        await db.execute(delete(ModelRegistry).where(ModelRegistry.task == "detection", ModelRegistry.is_champion.is_(True)))
        db.add(ModelRegistry(model_version=prev, task="detection", is_champion=False, gold_metrics={"map": baseline_map},
                             weights_uri="prev.pt"))
        db.add(ModelRegistry(model_version=tag, task="detection", is_champion=True, gold_metrics={"map": baseline_map},
                             promoted_from=prev, weights_uri="champ.pt"))
        await db.commit()
    return tag, prev


@requires_infra
def test_gold_drift_rolls_back_on_regression():
    from db.session import get_sessionmaker
    from services.agent.training_daemon import check_gold_drift
    from services.govern.champion import get_champion

    tag, prev = run_async(_seed_champion(0.80))

    async def regressed(_db, _champ):
        return 0.70   # mAP dropped 0.10, well past tolerance

    async def _flow():
        async with get_sessionmaker()() as db:
            res = await check_gold_drift(db, tolerance=0.03, evaluator=regressed)
            assert res["status"] == "rolled_back" and res["drop"] > 0.03
        async with get_sessionmaker()() as db:
            champ = await get_champion(db, "detection")
            assert champ is not None and champ.model_version == prev   # rolled back to the prior champion

    run_async(_flow())


@requires_infra
def test_gold_drift_healthy_when_stable():
    from db.session import get_sessionmaker
    from services.agent.training_daemon import check_gold_drift
    from services.govern.champion import get_champion

    tag, _prev = run_async(_seed_champion(0.80))

    async def stable(_db, _champ):
        return 0.79   # within tolerance

    async def _flow():
        async with get_sessionmaker()() as db:
            res = await check_gold_drift(db, tolerance=0.03, evaluator=stable)
            assert res["status"] == "healthy"
        async with get_sessionmaker()() as db:
            assert (await get_champion(db, "detection")).model_version == tag   # unchanged

    run_async(_flow())
